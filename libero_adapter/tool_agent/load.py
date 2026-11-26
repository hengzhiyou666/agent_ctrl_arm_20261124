"""Estimate conditional lifting load and pinch force without simulator truth."""

from collections.abc import Mapping
from math import isfinite, prod


# Deliberately broad assumptions for sizing a handling attempt from visible RGB
# categories. These ranges are not measured properties of any LIBERO asset.
# Entries are package mass (kg) and assumed packed-content density (kg/m³).
_PACKAGE_PRIORS = {
    "milk": ((0.010, 0.055), (950.0, 1150.0)),
    "orange_juice": ((0.010, 0.055), (950.0, 1200.0)),
    "salad_dressing": ((0.015, 0.200), (800.0, 1300.0)),
    "bbq_sauce": ((0.015, 0.200), (950.0, 1450.0)),
    "ketchup": ((0.015, 0.200), (950.0, 1450.0)),
    "alphabet_soup": ((0.025, 0.100), (900.0, 1350.0)),
    "tomato_sauce": ((0.025, 0.100), (950.0, 1400.0)),
    "cream_cheese": ((0.005, 0.045), (850.0, 1300.0)),
    "butter": ((0.005, 0.030), (800.0, 1100.0)),
    "chocolate_pudding": ((0.005, 0.050), (850.0, 1450.0)),
}
_ASSUMED_FILLED_FRACTION = (0.5, 1.0)
_ASSUMED_FRICTION = (0.2, 0.6)
_ASSUMED_ACCELERATION_M_S2 = (0.0, 0.5)
_GRAVITY_M_S2 = 9.81
_SAFETY_FACTOR = 1.5
_COULOMB_REFERENCE = "https://modernrobotics.northwestern.edu/nu-gm-book-resource/12-2-1-friction/"


def _bounding_volume(dimensions):
    if isinstance(dimensions, Mapping):
        dimensions = [dimensions.get(axis) for axis in ("width", "depth", "height")]
    try:
        width, depth, height = (float(value) for value in dimensions)
    except (TypeError, ValueError):
        return None
    if not all(isfinite(value) and 0 < value <= 1.0 for value in (width, depth, height)):
        return None
    return prod((width, depth, height))


def estimate_weight(session, object_id: str) -> dict:
    """Describe empty and filled mass scenarios for a localized package."""
    obj = session.objects.get(object_id)
    if obj is None:
        raise RuntimeError("Use an object_id from locate_object.")
    target = obj["target"]
    prior = _PACKAGE_PRIORS.get(target)
    volume = _bounding_volume(obj.get("dimensions_m"))
    load_id = f"load_{len(session.loads) + 1}"
    receipt = {
        "load_id": load_id,
        "object_id": object_id,
        "target": target,
        "status": "unknown",
        "mass_kg": None,
        "source": "RGB category and declared dimensions",
        "dimensions_source": obj.get("dimensions_source", "unknown"),
        "measured": False,
    }
    if prior is not None and volume is not None:
        empty, density = prior
        # An outer bounding box overcounts the internal volume. Keep an
        # explicit interval independent of what the scene appears to contain.
        filled = [tare + volume * density_value * fraction for tare, density_value, fraction
                  in zip(empty, density, _ASSUMED_FILLED_FRACTION)]
        receipt.update(
            status="scenario_estimate",
            mass_kg=[round(empty[0], 4), round(filled[1], 4)],
            empty_package_mass_kg=list(empty),
            filled_package_mass_kg=[round(value, 4) for value in filled],
            bounding_box_volume_l=round(1000 * volume, 3),
            assumed_content_density_kg_m3=list(density),
            assumed_filled_fraction_of_outer_box=list(_ASSUMED_FILLED_FRACTION),
            reason="Fill and material are unknown. These are handling scenarios using broad package assumptions.",
        )
    else:
        receipt["reason"] = "There is no package prior or usable dimension estimate for this object. Use measured mass if available."
    session.loads[load_id] = receipt
    return dict(receipt)


def _force_per_finger(mass_kg, friction, acceleration_m_s2):
    return _SAFETY_FACTOR * mass_kg * (_GRAVITY_M_S2 + acceleration_m_s2) / (2 * friction)


def estimate_grip_force(session, grasp_id: str, load_id: str) -> dict:
    """Estimate an illustrative clamp force interval for a symmetric vertical lift."""
    candidate = session.grasps.get(grasp_id)
    if candidate is None:
        raise RuntimeError("Use a grasp_id from propose_grasps.")
    load = session.loads.get(load_id)
    if load is None or load["object_id"] != candidate["object_id"]:
        raise RuntimeError("Use a load_id from estimate_weight for the same object as this grasp.")
    force_estimate_id = f"force_{len(session.grip_force_estimates) + 1}"
    check = session.grasp_checks.get(grasp_id)
    receipt = {
        "force_estimate_id": force_estimate_id,
        "grasp_id": grasp_id,
        "strategy": candidate["strategy"],
        "load_id": load_id,
        "status": "unknown",
        "normal_force_per_finger_n": None,
        "sum_normal_forces_n": None,
        "grasp_geometry_passed": check["passed"] if check is not None else None,
        "gripper_force_limit_n": None,
        "capacity_comparison": "unknown_force_limit_for_this_backend",
        "force_command_available": False,
        "force_measured": False,
        "reference": _COULOMB_REFERENCE,
    }
    mass = load["mass_kg"]
    if mass is not None:
        low_mass, high_mass = mass
        low_mu, high_mu = _ASSUMED_FRICTION
        low_accel, high_accel = _ASSUMED_ACCELERATION_M_S2
        per_finger = (_force_per_finger(low_mass, high_mu, low_accel),
                      _force_per_finger(high_mass, low_mu, high_accel))
        receipt.update(
            status="conditional_estimate",
            normal_force_per_finger_n=[round(value, 2) for value in per_finger],
            sum_normal_forces_n=[round(2 * value, 2) for value in per_finger],
            mass_kg=mass,
            assumed_static_friction=list(_ASSUMED_FRICTION),
            assumed_upward_acceleration_m_s2=list(_ASSUMED_ACCELERATION_M_S2),
            safety_factor=_SAFETY_FACTOR,
            assumptions=("Two opposing pads share a vertical load evenly. Actual friction, contact areas, "
                         "centre-of-mass torque and package crushing remain unknown."),
        )
    receipt["reason"] = ("This LIBERO interface has open and close commands. It supplies no calibrated clamp "
                         "force limit, measurement or force setpoint. Verify grip retention after a short lift.")
    session.grip_force_estimates[force_estimate_id] = receipt
    return dict(receipt)


TOOLS = {
    "estimate_weight": (
        estimate_weight,
        "Estimate a package's possible empty and filled mass from its RGB category and declared dimensions. "
        "Use after locate_object when payload could affect a grasp. Returns a load_id and the assumptions, or unknown.",
        {"object_id": {"type": "string", "description": "The localized target's object_id."}},
    ),
    "estimate_grip_force": (
        estimate_grip_force,
        "Estimate the normal force each finger would need to resist gravity during a symmetric vertical lift. "
        "Use after proposing a grasp and estimating its load. Returns a range in newtons with friction assumptions. "
        "The LIBERO controller cannot use this range as a force command. Check grip retention independently.",
        {"grasp_id": {"type": "string", "description": "The candidate to assess, from propose_grasps."},
         "load_id": {"type": "string", "description": "The estimated load for the same object, from estimate_weight."}},
    ),
}
