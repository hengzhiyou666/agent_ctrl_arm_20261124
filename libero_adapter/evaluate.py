"""Run and resume the LIBERO-Object evaluation over official initial states."""

from __future__ import annotations

import argparse
from concurrent.futures import FIRST_COMPLETED, ProcessPoolExecutor, wait
from contextlib import redirect_stderr, redirect_stdout
import csv
from dataclasses import dataclass
import fcntl
from hashlib import sha256
import json
import multiprocessing
import os
from pathlib import Path
from time import monotonic
import traceback

from libero_adapter.run import DEFAULT_LIBERO


TASK_COUNT = 10
STATES_PER_TASK = 50
INFRASTRUCTURE_STOP_LIMIT = 3


@dataclass(frozen=True)
class Case:
    task_id: int
    seed: int

    @property
    def folder(self) -> Path:
        return Path(f"task_{self.task_id:02d}") / f"state_{self.seed:02d}"


@dataclass(frozen=True)
class Settings:
    output: Path
    libero_root: Path
    mode: str
    endpoint: str
    model: str
    record_replay: bool


def write_json(path: Path, data: dict) -> None:
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_text(json.dumps(data, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    temporary.replace(path)


def read_json(path: Path) -> dict | None:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None
    return value if isinstance(value, dict) else None


def last_result(case: Case, output: Path) -> dict | None:
    value = read_json(output / case.folder / "trial.json")
    if value is None or (value.get("task_id"), value.get("seed")) != (case.task_id, case.seed):
        return None
    if value.get("status") == "scored" and type(value.get("official_success")) is bool:
        return value
    if value.get("status") == "infrastructure_error":
        return value
    return None


def evaluate_trial(case: Case, settings: Settings) -> dict:
    folder = settings.output / case.folder
    folder.mkdir(parents=True, exist_ok=True)
    attempt_number = 1
    while (folder / f"attempt_{attempt_number:03d}.log").exists():
        attempt_number += 1
    attempt = folder / f"attempt_{attempt_number:03d}"
    log_path = attempt.with_suffix(".log")
    trial = {"task_id": case.task_id, "seed": case.seed,
             "attempt": str(attempt.relative_to(settings.output)),
             "log": str(log_path.relative_to(settings.output))}
    started = monotonic()
    with log_path.open("x", encoding="utf-8", buffering=1) as log, redirect_stdout(log), redirect_stderr(log):
        try:
            from libero_adapter.run import evaluate_case

            outcome = evaluate_case(task_id=case.task_id, seed=case.seed, output=attempt, mode=settings.mode,
                                    endpoint=settings.endpoint, model=settings.model,
                                    record_replay=settings.record_replay, libero_root=settings.libero_root)
            if outcome.get("infrastructure_error") or type(outcome.get("official_success")) is not bool:
                raise RuntimeError(outcome.get("error") or "runner did not return an official success verdict")
            trial.update(status="scored", official_success=outcome["official_success"],
                         episode_completed=outcome.get("episode_completed"),
                         instruction=outcome.get("instruction"), error=outcome.get("error"),
                         result=str((attempt / "result.json").relative_to(settings.output)))
        except Exception as error:
            traceback.print_exc(file=log)
            trial.update(status="infrastructure_error", error=f"{type(error).__name__}: {error}")
    trial["elapsed_s"] = round(monotonic() - started, 3)
    write_json(folder / "trial.json", trial)
    return trial


def counts(cases: list[Case], trials: dict[Case, dict]) -> dict:
    recorded = [trials[case] for case in cases if case in trials]
    scored = [trial for trial in recorded if trial["status"] == "scored"]
    successes = sum(trial["official_success"] for trial in scored)
    return {
        "planned": len(cases),
        "scored": len(scored),
        "successes": successes,
        "failures": len(scored) - successes,
        "agent_loops_completed": sum(trial.get("episode_completed") is True for trial in scored),
        "infrastructure_errors": len(recorded) - len(scored),
        "pending": len(cases) - len(recorded),
        "success_rate": successes / len(scored) if scored else None,
    }


def write_report(output: Path, cases: list[Case], trials: dict[Case, dict], *, stop_reason: str | None = None) -> dict:
    overall = counts(cases, trials)
    tasks = []
    for task_id in sorted({case.task_id for case in cases}):
        task_cases = [case for case in cases if case.task_id == task_id]
        instruction = next((trials[case]["instruction"] for case in task_cases
                            if case in trials and trials[case].get("instruction")), None)
        tasks.append({"task_id": task_id, "instruction": instruction, **counts(task_cases, trials)})
    complete = overall["scored"] == overall["planned"]
    summary = {
        "suite": "libero_object",
        "complete": complete,
        "full_evaluation": complete and len(cases) == TASK_COUNT * STATES_PER_TASK,
        "stop_reason": stop_reason,
        "overall": overall,
        "tasks": tasks,
    }
    write_json(output / "summary.json", summary)
    return summary


def write_csv(output: Path, cases: list[Case], trials: dict[Case, dict]) -> None:
    path = output / "trials.csv"
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    fields = ("task_id", "seed", "status", "official_success", "episode_completed",
              "elapsed_s", "instruction", "error", "result", "log")
    with temporary.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        for case in cases:
            writer.writerow(trials.get(case, {"task_id": case.task_id, "seed": case.seed, "status": "pending"}))
    temporary.replace(path)


def manifest_for(settings: Settings, cases: list[Case]) -> dict:
    sources = sorted(Path(__file__).resolve().parent.glob("*.py"))
    return {
        "suite": "libero_object",
        "libero_root": str(settings.libero_root),
        "mode": settings.mode,
        "endpoint": settings.endpoint,
        "model": settings.model,
        "record_replay": settings.record_replay,
        "tasks": sorted({case.task_id for case in cases}),
        "initial_state_indices": sorted({case.seed for case in cases}),
        "sources": {source.name: sha256(source.read_bytes()).hexdigest() for source in sources},
    }


def run_trials(settings: Settings, cases: list[Case], workers: int) -> int:
    trials = {case: trial for case in cases if (trial := last_result(case, settings.output)) is not None}
    pending = iter(case for case in cases if trials.get(case, {}).get("status") != "scored")
    summary = write_report(settings.output, cases, trials)
    print(f"已有 {summary['overall']['scored']}/{len(cases)} 局官方判定。", flush=True)
    stop_reason = None
    consecutive_errors = 0
    try:
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
                        trial = {"task_id": case.task_id, "seed": case.seed, "status": "infrastructure_error",
                                 "error": f"{type(error).__name__}: {error}"}
                        folder = settings.output / case.folder
                        folder.mkdir(parents=True, exist_ok=True)
                        write_json(folder / "trial.json", trial)
                    trials[case] = trial
                    if trial["status"] == "scored":
                        consecutive_errors = 0
                        verdict = "成功" if trial["official_success"] else "失败"
                    else:
                        consecutive_errors += 1
                        verdict = f"运行故障，待补跑 ({trial.get('error')})"
                    summary = write_report(settings.output, cases, trials)
                    scored = summary["overall"]["scored"]
                    won = summary["overall"]["successes"]
                    print(f"[{scored}/{len(cases)}，成功 {won}] 任务 {case.task_id} 状态 {case.seed}: {verdict}",
                          flush=True)
                    if consecutive_errors >= INFRASTRUCTURE_STOP_LIMIT:
                        stop_reason = "Repeated infrastructure errors. Resume this output after resolving them."
                    if stop_reason is None and (case := next(pending, None)) is not None:
                        try:
                            active[pool.submit(evaluate_trial, case, settings)] = case
                        except Exception as error:
                            stop_reason = f"Worker pool unavailable: {error}"
    finally:
        # Recover durable worker results if a process was interrupted before its reply reached the parent.
        for case in cases:
            if (trial := last_result(case, settings.output)) is not None:
                trials[case] = trial
        summary = write_report(settings.output, cases, trials, stop_reason=stop_reason)
        write_csv(settings.output, cases, trials)
    overall = summary["overall"]
    print(f"已判定 {overall['scored']}/{len(cases)} 局，成功 {overall['successes']} 局。", flush=True)
    return 0 if summary["complete"] else 1


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--libero", type=Path, default=DEFAULT_LIBERO)
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--tasks", nargs="+", type=int, choices=range(TASK_COUNT),
                        help="Optional subset for validation. Default: all 10 tasks.")
    parser.add_argument("--states-per-task", type=int, default=STATES_PER_TASK,
                        help="Optional prefix of official initial states. Default: all 50.")
    parser.add_argument("--mode", choices=("agent", "scripted"), default="agent")
    parser.add_argument("--endpoint", default="http://127.0.0.1:11434/v1")
    parser.add_argument("--model", default="qwen3.5:4b")
    parser.add_argument("--record-replay", action="store_true", help="Save a GIF for each trial.")
    args = parser.parse_args()
    if args.workers < 1 or not 1 <= args.states_per_task <= STATES_PER_TASK:
        parser.error("workers must be positive and states per task must be 1–50")
    settings = Settings(args.output.resolve(), args.libero.resolve(), args.mode,
                        args.endpoint, args.model, args.record_replay)
    if not (settings.libero_root / "libero" / "libero").is_dir():
        parser.error(f"LIBERO source not found at {settings.libero_root}")
    task_ids = sorted(set(args.tasks) if args.tasks else range(TASK_COUNT))
    cases = [Case(task_id, seed) for seed in range(args.states_per_task) for task_id in task_ids]
    settings.output.mkdir(parents=True, exist_ok=True)
    with (settings.output / ".evaluate.lock").open("a", encoding="utf-8") as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            parser.error(f"evaluation already running in {settings.output}")
        manifest_path = settings.output / "manifest.json"
        manifest = manifest_for(settings, cases)
        if manifest_path.exists() and read_json(manifest_path) != manifest:
            parser.error("output uses a different evaluation protocol or code revision; choose a new output directory")
        if not manifest_path.exists():
            write_json(manifest_path, manifest)
        return run_trials(settings, cases, min(len(cases), args.workers))


if __name__ == "__main__":
    raise SystemExit(main())
