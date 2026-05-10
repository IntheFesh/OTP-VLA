"""
Stage 5 sim evaluation rollout loop — paper-ready SR producer.

Eval protocol (matches OpenVLA-OFT eval convention for fair comparison):
  - Suite: LIBERO-Spatial (10 tasks)
  - Per-task: N episodes (default 5 for quick eval; 50 for full paper run)
  - Per-episode: deterministic seed = seed_base * 1e4 + task_id * 1e2 + episode_idx
                 max_steps = 600 (OFT default)
  - Success: env.check_success() polled after each env.step
  - Action chunk: predictor produces 8 actions per call; env steps through
                  them sequentially (no replanning until chunk consumed)
  - Termination: success / done flag / max_steps reached

Predictor RNG isolation (per E1.4 design):
  - predictor.reset(episode_seed) before env loop
  - sim env's RNG (env.seed) and predictor's RNG (counter-based) are
    independent; predict_chunk saves/restores torch+np state internally.

Output (in --output-dir):
  - results.json: per-episode {task_id, episode_idx, success, steps,
                  chunks, episode_seed, instruction}
  - summary.md: paper-table-ready per-task SR + overall mean
  - rollout_log.txt: high-level progress log

Usage:
  # Quick eval (1 task x 5 episodes):
  python scripts/eval/eval_standard.py \
      --predictor-class otp_soft \
      --ckpt $(cat .paper_ready_ckpt) \
      --task-ids 0 \
      --n-episodes-per-task 5

  # Full paper run (10 tasks x 50 episodes; ~5h on RTX PRO 6000):
  python scripts/eval/eval_standard.py \
      --predictor-class otp_soft \
      --ckpt $(cat .paper_ready_ckpt) \
      --n-episodes-per-task 50 \
      --num-samples 20 --reduction mean \
      --output-dir results/eval_full_$(date +%Y%m%d_%H%M%S)
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

# MUJOCO_GL must be set BEFORE importing libero
os.environ.setdefault("MUJOCO_GL", "egl")

import numpy as np
import torch

# Patch torch.load to default weights_only=False (LIBERO init_states are pickle)
_orig_torch_load = torch.load
def _patched_load(*args, **kwargs):
    kwargs.setdefault("weights_only", False)
    return _orig_torch_load(*args, **kwargs)
torch.load = _patched_load

from otp.eval.predictor import (
    ActionPredictor,
    OTPSoftPredictor,
    OpenVLAOFTPredictor,
)

logger = logging.getLogger(__name__)


# ===========================================================================
# Predictor factory
# ===========================================================================
def build_predictor(args: argparse.Namespace) -> ActionPredictor:
    """Construct predictor based on --predictor-class arg."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type != "cuda":
        logger.warning(
            "CUDA not available; sim eval on CPU is impractical "
            "(forward ~100x slower than GPU)."
        )

    if args.predictor_class == "otp_soft":
        return OTPSoftPredictor(
            ckpt_path=args.ckpt,
            config_path=args.config,
            grasp_affordance_dir=args.grasp_affordance_dir,
            device=device,
            amp_dtype=torch.bfloat16,
            log_diagnostics=args.log_diagnostics,
        )
    elif args.predictor_class == "oft":
        return OpenVLAOFTPredictor(device=device)  # not yet implemented
    else:
        raise ValueError(f"Unknown predictor class: {args.predictor_class!r}")


# ===========================================================================
# Episode rollout
# ===========================================================================
def rollout_one_episode(
    predictor: ActionPredictor,
    env,
    instruction: str,
    episode_seed: int,
    init_state: Optional[np.ndarray] = None,
    max_steps: int = 600,
    num_samples: int = 1,
    reduction: str = "none",
    log_diagnostics: bool = False,
) -> Dict[str, Any]:
    """Run one episode rollout to success or max_steps.

    Args:
        predictor:    ActionPredictor instance
        env:          LIBERO OffScreenRenderEnv
        instruction:  task language string
        episode_seed: seed for this (task, episode) pair
        max_steps:    OFT-standard 600
        num_samples:  CFM samples per predict_chunk call
        reduction:    'none' / 'mean' / 'median'

    Returns dict with:
        success:      bool (final env.check_success())
        steps:        int (total env.step() calls)
        chunks:       int (predict_chunk calls)
        success_step: int (first step at which success became True; -1 if never)
        episode_seed: int (echoed for results.json)
        wall_time:   float (seconds)
    """
    env.seed(episode_seed)
    env.reset()
    if init_state is not None:
        env.set_init_state(init_state)
        # take a no-op step to settle physics + get fresh obs
        obs, _, _, _ = env.step(np.zeros(7, dtype=np.float32))
    else:
        obs = env.reset()  # second reset returns obs
    predictor.reset(episode_seed)

    success = False
    success_step = -1
    step = 0
    chunks = 0
    t0 = time.time()

    while step < max_steps and not success:
        chunk = predictor.predict_chunk(
            obs, instruction,
            num_samples=num_samples,
            reduction=reduction,
        )  # (H=8, 7) np.float32

        # Step through chunk action-by-action
        for h_idx in range(chunk.shape[0]):
            action = chunk[h_idx].astype(np.float32)
            obs, rew, done, info = env.step(action)
            step += 1

            success = bool(env.check_success())
            if success:
                success_step = step
                break  # exit chunk-inner loop
            if done or step >= max_steps:
                break

        chunks += 1
        if log_diagnostics and chunks % 10 == 0:
            logger.debug(
                f"  rollout step={step} chunks={chunks} success={success}"
            )

    wall_time = time.time() - t0
    return {
        "success":      success,
        "steps":        step,
        "chunks":       chunks,
        "success_step": success_step,
        "episode_seed": episode_seed,
        "wall_time":    wall_time,
    }


# ===========================================================================
# Per-task eval
# ===========================================================================
def eval_one_task(
    predictor: ActionPredictor,
    task_suite,
    task_id: int,
    n_episodes: int,
    seed_base: int,
    max_steps: int = 600,
    num_samples: int = 1,
    reduction: str = "none",
    log_diagnostics: bool = False,
) -> List[Dict[str, Any]]:
    """Evaluate one task across N episodes."""
    from libero.libero import get_libero_path
    from libero.libero.envs import OffScreenRenderEnv

    task = task_suite.get_task(task_id)
    bddl_path = (
        Path(get_libero_path("bddl_files"))
        / task.problem_folder
        / task.bddl_file
    )
    if not bddl_path.exists():
        raise FileNotFoundError(f"BDDL file missing: {bddl_path}")

    instruction = task.language
    logger.info(f"  [task {task_id}] '{instruction}'")
    logger.info(f"  bddl: {bddl_path.name}")

    # Load LIBERO standard demo init states (50 per task)
    init_states_arr = task_suite.get_task_init_states(task_id)
    n_inits = init_states_arr.shape[0]
    logger.info(f"  loaded {n_inits} init_states for task {task_id}")

    # One env per task (re-seeded between episodes)
    env = OffScreenRenderEnv(
        bddl_file_name=str(bddl_path),
        camera_heights=128,
        camera_widths=128,
    )

    episode_results = []
    try:
        for ep_idx in range(n_episodes):
            episode_seed = seed_base * 10000 + task_id * 100 + ep_idx
            init_state = init_states_arr[ep_idx % n_inits]
            t_episode_start = time.time()
            result = rollout_one_episode(
                predictor=predictor,
                env=env,
                instruction=instruction,
                episode_seed=episode_seed,
                init_state=init_state,
                max_steps=max_steps,
                num_samples=num_samples,
                reduction=reduction,
                log_diagnostics=log_diagnostics,
            )
            result["task_id"] = task_id
            result["episode_idx"] = ep_idx
            result["instruction"] = instruction
            episode_results.append(result)

            flag = "✓" if result["success"] else "✗"
            logger.info(
                f"  [task {task_id} ep {ep_idx}] {flag} "
                f"steps={result['steps']:3d} chunks={result['chunks']:3d} "
                f"wall={result['wall_time']:5.1f}s success_step={result['success_step']}"
            )
    finally:
        env.close()

    return episode_results


# ===========================================================================
# Aggregation
# ===========================================================================
def aggregate_results(
    all_results: List[Dict[str, Any]],
    task_suite,
    task_ids: List[int],
) -> Dict[str, Any]:
    """Compute per-task and overall SR + summary stats."""
    by_task: Dict[int, List[Dict[str, Any]]] = {tid: [] for tid in task_ids}
    for r in all_results:
        by_task[r["task_id"]].append(r)

    per_task_summary = []
    for tid in task_ids:
        task_eps = by_task[tid]
        if not task_eps:
            continue
        n = len(task_eps)
        n_success = sum(1 for r in task_eps if r["success"])
        sr = n_success / n if n > 0 else 0.0
        avg_steps = float(np.mean([r["steps"] for r in task_eps]))
        avg_steps_success = float(np.mean(
            [r["steps"] for r in task_eps if r["success"]]
        )) if n_success > 0 else float("nan")
        per_task_summary.append({
            "task_id":           tid,
            "instruction":       task_suite.get_task(tid).language,
            "n_episodes":        n,
            "n_success":         n_success,
            "success_rate":      sr,
            "avg_steps":         avg_steps,
            "avg_steps_success": avg_steps_success,
        })

    total_n = sum(t["n_episodes"] for t in per_task_summary)
    total_success = sum(t["n_success"] for t in per_task_summary)
    overall_sr = total_success / total_n if total_n > 0 else 0.0
    return {
        "overall": {
            "n_episodes_total": total_n,
            "n_success_total":  total_success,
            "success_rate":     overall_sr,
        },
        "per_task": per_task_summary,
    }


def write_summary_md(summary: Dict[str, Any], output_path: Path) -> None:
    """Paper-table-ready markdown summary."""
    lines = []
    lines.append("# Sim eval summary\n")
    o = summary["overall"]
    lines.append(f"## Overall")
    lines.append(f"- **Success rate**: {o['success_rate']:.1%}  "
                 f"({o['n_success_total']} / {o['n_episodes_total']})\n")

    lines.append(f"## Per-task SR")
    lines.append(f"| Task ID | Instruction | N | Success | SR | Avg steps (success) |")
    lines.append(f"|---|---|---|---|---|---|")
    for t in summary["per_task"]:
        avg_s = t["avg_steps_success"]
        avg_s_str = f"{avg_s:.1f}" if not np.isnan(avg_s) else "—"
        lines.append(
            f"| {t['task_id']} "
            f"| {t['instruction'][:60]}{'…' if len(t['instruction']) > 60 else ''} "
            f"| {t['n_episodes']} | {t['n_success']} "
            f"| {t['success_rate']:.1%} | {avg_s_str} |"
        )
    output_path.write_text("\n".join(lines))


# ===========================================================================
# Main
# ===========================================================================
def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)

    # Predictor
    p.add_argument("--predictor-class", type=str, default="otp_soft",
                   choices=["otp_soft", "oft"],
                   help="Which predictor adapter to use")
    p.add_argument("--ckpt", type=Path, required=True,
                   help="Path to ckpt (.pt) for otp_soft predictor")
    p.add_argument("--config", type=Path,
                   default=Path("configs/otp_soft_4e.yaml"),
                   help="Model config yaml")
    p.add_argument("--grasp-affordance-dir", type=Path,
                   default=Path("data/grasp_affordances"),
                   help="Mesh + grasp affordance npz dir")

    # Eval scope
    p.add_argument("--suite", type=str, default="libero_spatial",
                   help="LIBERO benchmark suite name")
    p.add_argument("--task-ids", type=int, nargs="+", default=None,
                   help="Specific task IDs to eval (default: all 10)")
    p.add_argument("--n-episodes-per-task", type=int, default=5,
                   help="Episodes per task (5 for quick eval, 50 for full paper run)")
    p.add_argument("--max-steps", type=int, default=600,
                   help="Max env steps per episode (OFT default 600)")
    p.add_argument("--seed-base", type=int, default=0,
                   help="Episode seed = seed_base*1e4 + task_id*1e2 + episode_idx")

    # Predictor inference settings
    p.add_argument("--num-samples", type=int, default=1,
                   help="CFM samples per predict_chunk (>=1)")
    p.add_argument("--reduction", type=str, default="none",
                   choices=["none", "mean", "median"],
                   help="Multi-sample reduction (only matters if num-samples > 1)")

    # Output
    p.add_argument("--output-dir", type=Path, required=True,
                   help="Where to dump results.json + summary.md")
    p.add_argument("--log-diagnostics", action="store_true",
                   help="Verbose per-call DEBUG log")

    return p.parse_args()


def main() -> None:
    args = parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)

    # Logger setup
    log_path = args.output_dir / "rollout_log.txt"
    logging.basicConfig(
        level=logging.DEBUG if args.log_diagnostics else logging.INFO,
        format="%(asctime)s | %(message)s",
        datefmt="%H:%M:%S",
        handlers=[
            logging.FileHandler(log_path),
            logging.StreamHandler(sys.stdout),
        ],
    )
    logger.info("=" * 70)
    logger.info(f"Sim eval starting")
    logger.info(f"  predictor:  {args.predictor_class}")
    logger.info(f"  ckpt:       {args.ckpt}")
    logger.info(f"  config:     {args.config}")
    logger.info(f"  suite:      {args.suite}")
    logger.info(f"  episodes/task: {args.n_episodes_per_task}")
    logger.info(f"  num_samples: {args.num_samples} reduction: {args.reduction}")
    logger.info(f"  seed_base:  {args.seed_base}")
    logger.info(f"  max_steps:  {args.max_steps}")
    logger.info(f"  output:     {args.output_dir}")
    logger.info("=" * 70)

    # Build predictor
    logger.info("Building predictor...")
    predictor = build_predictor(args)
    logger.info("Predictor ready.")

    # Load benchmark
    from libero.libero import benchmark
    benchmark_dict = benchmark.get_benchmark_dict()
    if args.suite not in benchmark_dict:
        raise ValueError(
            f"Unknown suite {args.suite!r}; available: {list(benchmark_dict.keys())}"
        )
    task_suite = benchmark_dict[args.suite]()
    n_tasks_total = task_suite.n_tasks if hasattr(task_suite, "n_tasks") else 10

    if args.task_ids is None:
        args.task_ids = list(range(n_tasks_total))
    logger.info(f"Evaluating tasks: {args.task_ids}")

    # Eval loop
    all_results: List[Dict[str, Any]] = []
    t0 = time.time()
    for task_id in args.task_ids:
        logger.info("-" * 70)
        task_results = eval_one_task(
            predictor=predictor,
            task_suite=task_suite,
            task_id=task_id,
            n_episodes=args.n_episodes_per_task,
            seed_base=args.seed_base,
            max_steps=args.max_steps,
            num_samples=args.num_samples,
            reduction=args.reduction,
            log_diagnostics=args.log_diagnostics,
        )
        all_results.extend(task_results)

    total_wall = time.time() - t0

    # Aggregate
    summary = aggregate_results(all_results, task_suite, args.task_ids)
    summary["meta"] = {
        "predictor_class":      args.predictor_class,
        "ckpt":                 str(args.ckpt),
        "config":               str(args.config),
        "suite":                args.suite,
        "task_ids":             args.task_ids,
        "n_episodes_per_task":  args.n_episodes_per_task,
        "max_steps":            args.max_steps,
        "seed_base":            args.seed_base,
        "num_samples":          args.num_samples,
        "reduction":            args.reduction,
        "total_wall_seconds":   total_wall,
    }

    # Write JSON
    results_path = args.output_dir / "results.json"
    with open(results_path, "w") as f:
        json.dump({
            "summary":  summary,
            "episodes": all_results,
        }, f, indent=2, default=str)
    logger.info(f"Wrote: {results_path}")

    # Write summary.md
    summary_md_path = args.output_dir / "summary.md"
    write_summary_md(summary, summary_md_path)
    logger.info(f"Wrote: {summary_md_path}")

    # Final print
    logger.info("=" * 70)
    logger.info(f"DONE in {total_wall:.1f}s")
    logger.info(f"Overall SR: {summary['overall']['success_rate']:.1%} "
                f"({summary['overall']['n_success_total']} "
                f"/ {summary['overall']['n_episodes_total']})")
    for t in summary["per_task"]:
        logger.info(
            f"  task {t['task_id']:2d}: SR {t['success_rate']:.1%} "
            f"({t['n_success']}/{t['n_episodes']})"
        )
    logger.info("=" * 70)


if __name__ == "__main__":
    main()
