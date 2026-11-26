"""Construct Panda pick-and-place motions from calibrated RGB estimates."""

from dataclasses import dataclass
import math


@dataclass(frozen=True)
class GraspProfile:
    grasp_height_m: float
    jaw_yaw_deg: float = 0.0


# Nominal pad heights from the upright or flat object geometry in LIBERO's
# HOPE assets. The two flat packs and the pudding box need the jaws aligned
# across their short side. The angles describe the jaw closure axis, not the
# object's long axis.
PROFILES = {
    "alphabet_soup": GraspProfile(0.049),
    "bbq_sauce": GraspProfile(0.074),
    "butter": GraspProfile(0.009, 90.0),
    "chocolate_pudding": GraspProfile(0.013, 90.0),
    "cream_cheese": GraspProfile(0.009, 90.0),
    "ketchup": GraspProfile(0.102),
    "milk": GraspProfile(0.079),
    "orange_juice": GraspProfile(0.079),
    "salad_dressing": GraspProfile(0.104),
    "tomato_sauce": GraspProfile(0.049),
}


@dataclass(frozen=True)
class PickPlan:
    plan_id: str
    pick_xy: tuple[float, float]
    place_xy: tuple[float, float]
    waypoints: tuple


def plan_pick_place(plan_id, target, pick_xy, place_xy, floor_z, *, grasp_yaw_deg=None) -> PickPlan:
    """Make a top-down grasp, lift clear of the basket lip, and release in the basket."""
    profile = PROFILES.get(target)
    if profile is None:
        raise RuntimeError(f"no grasp geometry profile for {target}")

    pick_xy = tuple(float(value) for value in pick_xy)
    place_xy = tuple(float(value) for value in place_xy)
    for x, y in (pick_xy, place_xy):
        if not 0.30 <= x <= 0.80 or abs(y) > 0.40:
            raise RuntimeError("RGB estimate is outside the calibrated Franka pick-and-place region")

    yaw_deg = profile.jaw_yaw_deg if grasp_yaw_deg is None else float(grasp_yaw_deg)
    if not math.isfinite(yaw_deg) or not math.isfinite(floor_z):
        raise RuntimeError("grasp orientation and calibrated floor must be finite")
    # Parallel jaw symmetry lets us avoid unnecessary wrist rotation.
    yaw_deg = (yaw_deg + 90.0) % 180.0 - 90.0 if not -90.0 <= yaw_deg <= 90.0 else yaw_deg
    yaw_half = math.radians(yaw_deg) / 2.0
    down = (0.0, math.cos(yaw_half), math.sin(yaw_half), 0.0)

    grasp_height = profile.grasp_height_m
    grasp = (*pick_xy, floor_z + grasp_height)
    pick_hover = (*pick_xy, floor_z + max(0.25, grasp_height + 0.17))
    place_hover = (*place_xy, floor_z + max(0.235, grasp_height + 0.17))
    waypoints = (
        ("approach", pick_hover, down),
        ("descend", grasp, down),
        ("close", None, None),
        ("lift", pick_hover, down),
        ("transfer", place_hover, down),
        ("open", None, None),
    )
    return PickPlan(plan_id, pick_xy, place_xy, waypoints)
