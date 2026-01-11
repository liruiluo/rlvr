from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path


@dataclass
class EvalPoint:
    step: int
    reward_mean_at_1: float
    log_path: str
    elapsed_sec: float


@dataclass
class TaskResult:
    task: str
    experiment_name: str
    eval_points: list[EvalPoint]
    converged_step: int | None
    converged_step_threshold: int | None
    converged_step_plateau: int | None
    best_step: int | None
    best_reward_mean_at_1: float | None


def parse_last_val_reward(log_text: str, task: str) -> tuple[int, float] | None:
    pattern = re.compile(
        rf"step:(\d+).*?val-core/reasoning_gym/{re.escape(task)}/reward/mean@1:(?:np\.float64\()?"
        r"([0-9eE.+-]+)\)?"
    )
    matches = list(pattern.finditer(log_text))
    if not matches:
        return None
    last = matches[-1]
    return int(last.group(1)), float(last.group(2))


def parse_first_val_reward(log_text: str, task: str) -> tuple[int, float] | None:
    pattern = re.compile(
        rf"step:(\d+).*?val-core/reasoning_gym/{re.escape(task)}/reward/mean@1:(?:np\.float64\()?"
        r"([0-9eE.+-]+)\)?"
    )
    matches = list(pattern.finditer(log_text))
    if not matches:
        return None
    first = matches[0]
    return int(first.group(1)), float(first.group(2))


def write_json(path: Path, obj) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(description="Run per-task convergence sweeps for ReasoningGym + veRL.")
    parser.add_argument("--config-name", default="algo/rgym/grpo_moe_lora_sphere_hf_single_gpu_perf")
    parser.add_argument("--project-name", default="rgym_convergence")
    parser.add_argument("--tasks", nargs="+", default=["chain_sum", "gcd", "base_conversion", "spell_backward"])
    parser.add_argument("--max-steps", type=int, default=500)
    parser.add_argument("--chunk-steps", type=int, default=20)
    parser.add_argument("--threshold", type=float, default=0.9)
    parser.add_argument("--patience", type=int, default=3)
    parser.add_argument("--plateau-eps", type=float, default=0.02)

    parser.add_argument("--model", default="Qwen/Qwen2.5-0.5B-Instruct")
    parser.add_argument("--dataset-size", type=int, default=256)
    parser.add_argument("--train-batch-size", type=int, default=16)
    parser.add_argument("--val-batch-size", type=int, default=32)
    parser.add_argument("--max-response-length", type=int, default=16)
    parser.add_argument("--rollout-n", type=int, default=2)
    parser.add_argument("--logprob-micro-bsz", type=int, default=8)

    parser.add_argument("--reports-dir", default="experiments/verl_rgym/reports")
    parser.add_argument("--logs-dir", default="experiments/verl_rgym/logs/convergence_sweep")
    args = parser.parse_args()

    repo_root = Path(__file__).resolve().parents[3]
    runner = repo_root / "experiments" / "verl_rgym" / "grpo_train_local.py"
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    run_root = repo_root / args.logs_dir / timestamp
    reports_root = repo_root / args.reports_dir / f"convergence_sweep_{timestamp}"
    run_root.mkdir(parents=True, exist_ok=True)
    reports_root.mkdir(parents=True, exist_ok=True)

    results: list[TaskResult] = []

    for task in args.tasks:
        exp_name = f"conv_{task}_bs{args.train_batch_size}_n{args.rollout_n}_r{args.max_response_length}"
        print(f"[sweep] task={task} exp={exp_name}", flush=True)
        task_result = TaskResult(
            task=task,
            experiment_name=exp_name,
            eval_points=[],
            converged_step=None,
            converged_step_threshold=None,
            converged_step_plateau=None,
            best_step=None,
            best_reward_mean_at_1=None,
        )

        window: list[float] = []
        start_step = 0
        while start_step < args.max_steps:
            target_step = min(args.max_steps, start_step + args.chunk_steps)
            log_path = run_root / f"{task}_to_step_{target_step}.log"
            print(f"[sweep] task={task} run steps {start_step}->{target_step} log={log_path}", flush=True)

            cmd = [
                sys.executable,
                str(runner),
                "--config-name",
                args.config_name,
                f"trainer.project_name={args.project_name}",
                f"trainer.experiment_name={exp_name}",
                f"actor_rollout_ref.model.path={args.model}",
                f"task={task}",
                f"reasoning_gym.dataset_size={args.dataset_size}",
                f"data.train_batch_size={args.train_batch_size}",
                f"data.val_batch_size={args.val_batch_size}",
                f"data.max_response_length={args.max_response_length}",
                f"actor_rollout_ref.rollout.n={args.rollout_n}",
                f"actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu={args.logprob_micro_bsz}",
                f"trainer.total_epochs=999999",
                f"trainer.total_training_steps={target_step}",
                f"trainer.save_freq={args.chunk_steps}",
                f"trainer.test_freq={args.chunk_steps}",
                "trainer.logger=[console]",
            ]

            t0 = time.perf_counter()
            proc = subprocess.run(cmd, capture_output=True, text=True)
            elapsed = time.perf_counter() - t0
            log_path.write_text(proc.stdout + proc.stderr, encoding="utf-8")
            if proc.returncode != 0:
                raise RuntimeError(f"Task={task} failed at target_step={target_step}. See {log_path}")

            merged_log = proc.stdout + proc.stderr
            if start_step == 0 and not task_result.eval_points:
                first = parse_first_val_reward(merged_log, task=task)
                if first is not None:
                    first_step, first_reward = first
                    task_result.eval_points.append(
                        EvalPoint(step=0, reward_mean_at_1=first_reward, log_path=str(log_path), elapsed_sec=0.0)
                    )
                    if task_result.best_reward_mean_at_1 is None or first_reward > task_result.best_reward_mean_at_1:
                        task_result.best_reward_mean_at_1 = first_reward
                        task_result.best_step = 0
                    print(
                        f"[sweep] task={task} initial eval logged_step={first_step} recorded_step=0 mean@1={first_reward}",
                        flush=True,
                    )

            parsed = parse_last_val_reward(merged_log, task=task)
            if parsed is not None:
                step, reward = parsed
                print(f"[sweep] task={task} eval step={step} mean@1={reward}", flush=True)
                task_result.eval_points.append(
                    EvalPoint(step=step, reward_mean_at_1=reward, log_path=str(log_path), elapsed_sec=elapsed)
                )
                if task_result.best_reward_mean_at_1 is None or reward > task_result.best_reward_mean_at_1:
                    task_result.best_reward_mean_at_1 = reward
                    task_result.best_step = step

                window.append(reward)
                if len(window) > args.patience:
                    window.pop(0)
                if len(window) == args.patience:
                    if task_result.converged_step_threshold is None and all(v >= args.threshold for v in window):
                        task_result.converged_step_threshold = step

                    best = task_result.best_reward_mean_at_1 or reward
                    if (max(window) - min(window)) <= args.plateau_eps and (best - reward) <= args.plateau_eps:
                        task_result.converged_step = step
                        task_result.converged_step_plateau = step
                        print(
                            f"[sweep] task={task} converged at step={step} (plateau_eps={args.plateau_eps}, patience={args.patience})",
                            flush=True,
                        )
                        break

            write_json(
                reports_root / "partial.json",
                {"args": vars(args), "results": [asdict(r) for r in results] + [asdict(task_result)]},
            )

            start_step = target_step

        results.append(task_result)
        write_json(reports_root / "partial.json", {"args": vars(args), "results": [asdict(r) for r in results]})

    write_json(reports_root / "results.json", {"args": vars(args), "results": [asdict(r) for r in results]})

    lines = ["# Convergence sweep", "", f"- config: `{args.config_name}`", f"- timestamp: `{timestamp}`", ""]
    for r in results:
        last = r.eval_points[-1].reward_mean_at_1 if r.eval_points else None
        lines.append(
            f"- task `{r.task}`: converged_step={r.converged_step}, threshold_step={r.converged_step_threshold}, best={r.best_reward_mean_at_1}@{r.best_step}, last={last}"
        )
    (reports_root / "results.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
