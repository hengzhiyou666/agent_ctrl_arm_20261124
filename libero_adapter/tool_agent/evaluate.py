"""Run and resume the tool-calling agent on the official LIBERO-Object states."""

from __future__ import annotations

import argparse
from concurrent.futures import FIRST_COMPLETED, ProcessPoolExecutor, wait
from contextlib import redirect_stderr, redirect_stdout
import csv
from dataclasses import dataclass
import fcntl
from hashlib import sha256
from importlib import metadata
import json
import multiprocessing
import os
from pathlib import Path
import platform
from time import monotonic
import traceback
from urllib.parse import urlsplit

from libero_adapter.evaluate import (
    INFRASTRUCTURE_STOP_LIMIT,
    STATES_PER_TASK,
    TASK_COUNT,
    Case,
    last_result,
    read_json,
    write_json,
    write_report,
)
from libero_adapter.run import DEFAULT_LIBERO, ROOT


DEFAULT_ENDPOINT = "http://127.0.0.1:11434/v1"
DEFAULT_MODEL = "qwen3.5:4b"
SUITE = "libero_object"
SOURCE_SUFFIXES = {".py", ".json", ".toml", ".yaml", ".yml"}
RUNTIME_PACKAGES = ("mujoco", "robosuite", "torch", "numpy", "Pillow", "scipy", "opencv-python",
                    "opencv-python-headless", "httpx", "openai", "transformers")
CSV_FIELDS = ("task_id", "seed", "status", "official_success", "episode_completed", "control_steps",
              "elapsed_s", "instruction", "failure_reason", "error", "result", "events", "replay",
              "attempt", "log")


@dataclass(frozen=True)
class Settings:
    output: Path
    libero_root: Path
    endpoint: str = DEFAULT_ENDPOINT
    model: str = DEFAULT_MODEL
    record_replay: bool = False
    model_manifest: Path | None = None
    live_monitor: bool = False


def file_sha256(path: Path) -> str:
    digest = sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def file_hashes(root: Path, candidates, *, suffixes: set[str] | None = None) -> dict[str, str]:
    return {path.relative_to(root).as_posix(): file_sha256(path)
            for path in sorted(set(candidates))
            if path.is_file() and "__pycache__" not in path.parts
            and (suffixes is None or path.suffix in suffixes)}


def tree_sha256(hashes: dict[str, str]) -> str:
    return sha256(json.dumps(hashes, sort_keys=True, separators=(",", ":")).encode("utf-8")).hexdigest()


def source_manifest() -> dict:
    adapter = ROOT / "libero_adapter"
    captain = ROOT / "CaPtain-Harness" / "src" / "captain_harness"
    candidates = [
        *adapter.glob("*.py"),
        *(adapter / "tool_agent").rglob("*"),
        *(adapter / "skills").rglob("*"),
        *captain.rglob("*.py"),
        *(ROOT / "simulation").rglob("*.py"),
    ]
    hashes = file_hashes(ROOT, candidates, suffixes=SOURCE_SUFFIXES)
    return {"sha256": tree_sha256(hashes), "files": hashes}


def scene_manifest(libero_root: Path) -> dict:
    import yaml

    source = libero_root / "libero" / "libero"
    config_dir = Path(os.environ.get("LIBERO_CONFIG_PATH", libero_root / ".libero-local")).expanduser()
    config_file = config_dir / "config.yaml"
    if not source.is_dir() or not config_file.is_file():
        raise ValueError(f"Initialised LIBERO source and config required at {source} and {config_file}")
    with config_file.open(encoding="utf-8") as stream:
        config = yaml.safe_load(stream)
    if not isinstance(config, dict):
        raise ValueError(f"Expected a LIBERO path mapping in {config_file}")
    directories = {"bddl_files": SUITE, "init_states": SUITE, "assets": ""}
    groups = {}
    for name, subdirectory in directories.items():
        if not isinstance(config.get(name), str):
            raise ValueError(f"Missing {name} path in {config_file}")
        directory = (Path(config[name]).expanduser() / subdirectory).resolve()
        hashes = file_hashes(directory, directory.rglob("*"))
        if not hashes:
            raise ValueError(f"Missing LIBERO {name} files in {directory}")
        groups[name] = {"path": str(directory), "sha256": tree_sha256(hashes), "files": len(hashes)}
    code = file_hashes(source, source.rglob("*.py"))
    groups["benchmark_code"] = {"path": str(source), "sha256": tree_sha256(code), "files": len(code)}
    return {"config_path": str(config_file.resolve()), "groups": groups,
            "sha256": tree_sha256({name: group["sha256"] for name, group in groups.items()})}


def local_ollama_manifest(settings: Settings) -> Path | None:
    endpoint = urlsplit(settings.endpoint)
    if endpoint.hostname not in {"localhost", "127.0.0.1", "::1"} or endpoint.port != 11434:
        return None
    model = settings.model
    name, sep, tag = model.rpartition(":") if ":" in model.rsplit("/", 1)[-1] else (model, "", "")
    entries = name.split("/")
    if len(entries) == 1:
        entries = ["registry.ollama.ai", "library", *entries]
    elif len(entries) == 2:
        entries = ["registry.ollama.ai", *entries]
    if any(entry in {"", ".", ".."} for entry in [*entries, tag if sep else "latest"]):
        return None
    model_path = Path("manifests", *entries, tag if sep else "latest")
    roots = (Path.home() / ".ollama" / "models", Path("/usr/share/ollama/.ollama/models"),
             Path("/var/lib/ollama/.ollama/models"))
    if model_dir := os.environ.get("OLLAMA_MODELS"):
        roots = (Path(model_dir).expanduser(),)
    return next((root / model_path for root in roots if (root / model_path).is_file()), None)


def model_fingerprint(settings: Settings) -> dict:
    path = settings.model_manifest or local_ollama_manifest(settings)
    if path is None:
        return {"sha256": None, "provenance": "unverified_endpoint_model"}
    value = read_json(path)
    if value is None or not isinstance(value.get("layers"), list) or not isinstance(value.get("config"), dict):
        raise ValueError(f"Invalid Ollama model manifest at {path}")
    return {"sha256": file_sha256(path), "path": str(path.resolve()),
            "provenance": "provided_manifest" if settings.model_manifest else "local_ollama_manifest",
            "layers": value["layers"], "config": value["config"]}


def manifest_for(settings: Settings, cases: list[Case]) -> dict:
    packages = {}
    for name in RUNTIME_PACKAGES:
        try:
            packages[name] = metadata.version(name)
        except metadata.PackageNotFoundError:
            continue
    rendering_env = {name: os.environ.get(name) for name in
                     ("MUJOCO_GL", "PYOPENGL_PLATFORM", "CAPTAIN_LAZY_RENDER",
                      "__EGL_VENDOR_LIBRARY_FILENAMES", "CUDA_VISIBLE_DEVICES")}
    return {
        "suite": SUITE,
        "libero_root": str(settings.libero_root),
        "mode": "tool_agent",
        "model": settings.model,
        "endpoint": settings.endpoint,
        "record_replay": settings.record_replay,
        "live_monitor": settings.live_monitor,
        "tasks": sorted({case.task_id for case in cases}),
        "initial_state_indices": sorted({case.seed for case in cases}),
        "source": source_manifest(),
        "scene": scene_manifest(settings.libero_root),
        "model_fingerprint": model_fingerprint(settings),
        "runtime": {"python": platform.python_version(), "packages": packages,
                    "environment": rendering_env},
    }


def failure_reason(outcome: dict, events_path: Path) -> str:
    if reason := outcome.get("failure_reason") or outcome.get("error"):
        return str(reason)
    agent_report = outcome.get("agent_report")
    if isinstance(agent_report, dict) and (reason := agent_report.get("reason")):
        return str(reason)
    if events_path.is_file():
        for line in reversed(events_path.read_text(encoding="utf-8").splitlines()):
            try:
                event = json.loads(line)
            except ValueError:
                continue
            receipt = event.get("receipt", {}) if isinstance(event, dict) else {}
            if isinstance(receipt, dict) and receipt.get("ok") is False:
                if reason := receipt.get("reason") or receipt.get("error") or receipt.get("status"):
                    return f"{event.get('tool', 'tool')}: {reason}"
    return "LIBERO placement criterion was not satisfied at the final check"


def scored_fields(outcome: dict, attempt: Path, output: Path) -> dict:
    if outcome.get("infrastructure_error") or type(outcome.get("official_success")) is not bool:
        raise RuntimeError(outcome.get("error") or "Episode did not return an official success verdict")
    events = attempt / "events.jsonl"
    gif = next(iter(sorted(attempt.glob("*.gif"))), None)
    return {
        "status": "scored", "official_success": outcome["official_success"],
        "episode_completed": outcome.get("episode_completed"), "control_steps": outcome.get("control_steps"),
        "instruction": outcome.get("instruction"), "error": outcome.get("error"),
        "failure_reason": None if outcome["official_success"] else failure_reason(outcome, events),
        "result": str((attempt / "result.json").relative_to(output)),
        "events": str(events.relative_to(output)) if events.is_file() else None,
        "replay": str(gif.relative_to(output)) if gif else None,
    }


def attempt_fields(case: Case, attempt: Path, output: Path) -> dict:
    return {"task_id": case.task_id, "seed": case.seed, "attempt": str(attempt.relative_to(output)),
            "log": str(attempt.with_suffix(".log").relative_to(output))}


def recover_scored(case: Case, settings: Settings) -> dict | None:
    folder = settings.output / case.folder
    for attempt in sorted(folder.glob("attempt_*"), reverse=True):
        if not attempt.is_dir():
            continue
        result = read_json(attempt / "result.json")
        if (result is None or result.get("infrastructure_error") or type(result.get("official_success")) is not bool
                or result.get("task_id", result.get("task_index")) != case.task_id
                or result.get("seed") != case.seed or result.get("model") != settings.model):
            continue
        trial = {**attempt_fields(case, attempt, settings.output), **scored_fields(result, attempt, settings.output)}
        write_json(folder / "trial.json", trial)
        return trial
    return None


def evaluate_trial(case: Case, settings: Settings) -> dict:
    folder = settings.output / case.folder
    folder.mkdir(parents=True, exist_ok=True)
    attempt_number = 1
    attempt = folder / f"attempt_{attempt_number:03d}"
    while attempt.exists() or attempt.with_suffix(".log").exists():
        attempt_number += 1
        attempt = folder / f"attempt_{attempt_number:03d}"
    trial = attempt_fields(case, attempt, settings.output)
    started = monotonic()
    with attempt.with_suffix(".log").open("x", encoding="utf-8", buffering=1) as log:
        with redirect_stdout(log), redirect_stderr(log):
            try:
                from .agent import evaluate

                outcome = evaluate(task_id=case.task_id, seed=case.seed, output=attempt,
                                   model=settings.model, endpoint=settings.endpoint, libero_root=settings.libero_root,
                                   record_replay=settings.record_replay, live_monitor=settings.live_monitor)
                result_path = attempt / "result.json"
                fields = scored_fields(outcome, attempt, settings.output)
                if not result_path.is_file():
                    attempt.mkdir(parents=True, exist_ok=True)
                    write_json(result_path, {**outcome, "task_id": case.task_id, "seed": case.seed,
                                             "model": settings.model})
                trial.update(fields)
            except Exception as error:
                traceback.print_exc(file=log)
                trial.update(status="infrastructure_error", error=f"{type(error).__name__}: {error}")
    trial["elapsed_s"] = round(monotonic() - started, 3)
    write_json(folder / "trial.json", trial)
    return trial


def write_csv(output: Path, cases: list[Case], trials: dict[Case, dict]) -> None:
    path = output / "trials.csv"
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    with temporary.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=CSV_FIELDS, extrasaction="ignore")
        writer.writeheader()
        for case in cases:
            writer.writerow(trials.get(case, {"task_id": case.task_id, "seed": case.seed, "status": "pending"}))
    temporary.replace(path)


def read_trials(settings: Settings, cases: list[Case]) -> dict[Case, dict]:
    trials = {}
    for case in cases:
        trial = last_result(case, settings.output)
        if not trial or trial["status"] != "scored":
            trial = recover_scored(case, settings) or trial
        if trial:
            trials[case] = trial
    return trials


def run_pending(settings: Settings, cases: list[Case], trials: dict[Case, dict], workers: int) -> str | None:
    pending = iter(case for case in cases if trials.get(case, {}).get("status") != "scored")
    stop_reason = None
    consecutive_errors = 0
    with ProcessPoolExecutor(max_workers=workers, mp_context=multiprocessing.get_context("spawn")) as pool:
        active = {pool.submit(evaluate_trial, case, settings): case
                  for _ in range(workers) if (case := next(pending, None)) is not None}
        while active:
            done, _ = wait(active, return_when=FIRST_COMPLETED)
            for future in done:
                case = active.pop(future)
                try:
                    trial = future.result()
                except Exception as error:
                    trial = last_result(case, settings.output)
                    if trial is None or trial["status"] != "scored":
                        trial = recover_scored(case, settings)
                    if trial is None:
                        folder = settings.output / case.folder
                        folder.mkdir(parents=True, exist_ok=True)
                        trial = {"task_id": case.task_id, "seed": case.seed, "status": "infrastructure_error",
                                 "error": f"{type(error).__name__}: {error}"}
                        write_json(folder / "trial.json", trial)
                trials[case] = trial
                if trial["status"] == "scored":
                    consecutive_errors = 0
                    verdict = "成功" if trial["official_success"] else f"失败（{trial['failure_reason']}）"
                else:
                    consecutive_errors += 1
                    verdict = f"运行故障，待补跑（{trial.get('error')}）"
                summary = write_report(settings.output, cases, trials)
                write_csv(settings.output, cases, trials)
                overall = summary["overall"]
                print(f"[{overall['scored']}/{len(cases)}，成功 {overall['successes']}] "
                      f"任务 {case.task_id} 状态 {case.seed}　{verdict}", flush=True)
                if consecutive_errors >= INFRASTRUCTURE_STOP_LIMIT:
                    stop_reason = "Repeated infrastructure failures. Resume after fixing the reported errors."
                if stop_reason is None and (case := next(pending, None)) is not None:
                    try:
                        active[pool.submit(evaluate_trial, case, settings)] = case
                    except Exception as error:
                        stop_reason = f"Worker pool unavailable: {error}"
    return stop_reason


def run_trials(settings: Settings, cases: list[Case], workers: int) -> int:
    trials = read_trials(settings, cases)
    summary = write_report(settings.output, cases, trials)
    write_csv(settings.output, cases, trials)
    print(f"已有 {summary['overall']['scored']}/{len(cases)} 局完成官方判定。", flush=True)
    stop_reason = None
    try:
        if not summary["complete"]:
            stop_reason = run_pending(settings, cases, trials, workers)
    except KeyboardInterrupt:
        stop_reason = "Interrupted. Resume with the same output directory and options."
    except Exception as error:
        stop_reason = f"Evaluation runner unavailable: {type(error).__name__}: {error}"
    finally:
        trials = read_trials(settings, cases)
        summary = write_report(settings.output, cases, trials, stop_reason=stop_reason)
        write_csv(settings.output, cases, trials)
    if stop_reason:
        print(stop_reason, flush=True)
    overall = summary["overall"]
    print(f"已判定 {overall['scored']}/{len(cases)} 局，成功 {overall['successes']} 局。", flush=True)
    return 0 if summary["complete"] else 1


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--libero", type=Path, default=DEFAULT_LIBERO)
    parser.add_argument("--workers", type=int, default=6)
    parser.add_argument("--tasks", nargs="+", type=int, choices=range(TASK_COUNT),
                        help="Select tasks for a pilot or replay. Default: all 10 tasks.")
    state_selection = parser.add_mutually_exclusive_group()
    state_selection.add_argument("--states-per-task", type=int, default=STATES_PER_TASK,
                                 help="Prefix of official initial states. Default: all 50.")
    state_selection.add_argument("--seeds", nargs="+", type=int, choices=range(STATES_PER_TASK),
                                 metavar="STATE", help="Select official initial-state indices, starting at 0.")
    parser.add_argument("--endpoint", default=DEFAULT_ENDPOINT)
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--model-manifest", type=Path, help="Optional exact Ollama manifest for the chosen model.")
    parser.add_argument("--record-replay", action="store_true", help="Record a two-camera GIF per trial.")
    parser.add_argument("--live-monitor", action="store_true", help="Include asynchronous Qwen motion supervision.")
    args = parser.parse_args()
    if args.workers < 1 or not 1 <= args.states_per_task <= STATES_PER_TASK:
        parser.error("workers must be positive and states per task must be 1–50")
    settings = Settings(args.output.resolve(), args.libero.resolve(), args.endpoint, args.model,
                        args.record_replay, args.model_manifest.resolve() if args.model_manifest else None,
                        args.live_monitor)
    task_ids = sorted(set(args.tasks) if args.tasks else range(TASK_COUNT))
    states = sorted(set(args.seeds) if args.seeds else range(args.states_per_task))
    cases = [Case(task_id, seed) for seed in states for task_id in task_ids]
    settings.output.mkdir(parents=True, exist_ok=True)
    with (settings.output / ".evaluate.lock").open("a", encoding="utf-8") as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            parser.error(f"evaluation already running in {settings.output}")
        try:
            manifest = manifest_for(settings, cases)
        except (OSError, ValueError, ImportError) as error:
            parser.error(str(error))
        manifest_path = settings.output / "manifest.json"
        if manifest_path.exists():
            if read_json(manifest_path) != manifest:
                parser.error("output uses different settings, source, scene, or model revision; choose a new output")
        elif any(settings.output.glob("task_??")):
            parser.error("existing attempts lack an evaluation manifest; choose a new output directory")
        else:
            write_json(manifest_path, manifest)
        if manifest["model_fingerprint"]["sha256"] is None:
            print("当前模型服务没有可核验的本地清单，评测只记录模型名称与服务地址。", flush=True)
        return run_trials(settings, cases, min(len(cases), args.workers))


if __name__ == "__main__":
    raise SystemExit(main())
