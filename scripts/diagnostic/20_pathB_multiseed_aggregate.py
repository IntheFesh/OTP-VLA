"""
Path B Step 6: Multi-seed retrain result aggregation.

After 3 seeds × 50 epoch retrain completes, this script:
  1. Loads each seed's final ckpt
  2. Runs LIBERO sim eval (10 tasks × 50 episodes per seed)
  3. Reports mean ± std SR across seeds, per-task and overall

Required input: 3 trained checkpoints in
  results/pathB_step6_seed42_*/ckpt_step*.pt   (final ckpt of each)
  results/pathB_step6_seed137_*/
  results/pathB_step6_seed2026_*/

Output:
  results/phase0/pathB_step6_multiseed.json
  Console: per-seed SR table + aggregate stats
"""

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Dict, List

import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

OUT_DIR = REPO_ROOT / "results" / "phase0"
OUT_DIR.mkdir(parents=True, exist_ok=True)


def find_latest_ckpt_for_seed(seed: int) -> Path:
    """Locate the most-recent run directory for a given seed and its final ckpt."""
    pattern = f"pathB_step6_seed{seed}_*"
    matches = sorted(REPO_ROOT.glob(f"results/{pattern}"))
    if not matches:
        raise FileNotFoundError(f"No results/{pattern}/ directory found")
    run_dir = matches[-1]  # most-recent timestamp

    # Find largest-step ckpt
    ckpts = sorted(run_dir.glob("ckpt_step*.pt"))
    if not ckpts:
        raise FileNotFoundError(f"No ckpt_step*.pt in {run_dir}")
    return ckpts[-1]


def run_sim_eval(ckpt_path: Path, seed: int, n_episodes_per_task: int = 50) -> Dict:
    """Run LIBERO-Spatial 10 tasks × n_episodes eval on a Path B ckpt.

    Returns:
        dict with per_task SR + overall_sr + episode-level data.
    """
    from otp.eval.predictor import OTPSoftPredictor
    from libero.libero import benchmark

    # Find config file for this seed
    config_path = REPO_ROOT / f"configs/otp_soft_pathB_seed{seed}.yaml"
    if not config_path.exists():
        raise FileNotFoundError(f"Config not found: {config_path}")

    print(f"\n--- Seed {seed} ---")
    print(f"  ckpt: {ckpt_path}")
    print(f"  config: {config_path}")

    predictor = OTPSoftPredictor(
        ckpt_path=str(ckpt_path),
        config_path=str(config_path),
        grasp_affordance_dir="data/grasp_affordances",
        device=torch.device("cuda"),
        log_diagnostics=False,
    )

    task_suite = benchmark.get_benchmark_dict()["libero_spatial"]()

    per_task_results = {}
    total_succ = 0
    total_ep = 0
    t_start = time.time()

    for task_id in range(10):
        task = task_suite.get_task(task_id)
        # We need a LIBERO env runner here. OTPSoftPredictor.predict_chunk is the
        # action source; LIBERO env handles environment.
        from experiments.robot.libero.libero_utils import get_libero_env
        env, task_description = get_libero_env(task, "openvla", resolution=256)
        init_states = task_suite.get_task_init_states(task_id)

        # Per-task SR loop using OTPSoftPredictor
        task_succ = 0
        for ep_idx in range(n_episodes_per_task):
            episode_seed = 5000000 + seed * 100000 + task_id * 1000 + ep_idx
            predictor.reset(episode_seed=episode_seed)

            success = _run_one_episode(
                env, init_states[ep_idx % len(init_states)],
                predictor, task_description, max_steps=220,
            )
            task_succ += int(success)
            total_ep += 1
            if success:
                total_succ += 1

        per_task_results[task_id] = {
            "n_episodes": n_episodes_per_task,
            "successes": task_succ,
            "sr": task_succ / n_episodes_per_task,
            "task_description": task_description,
        }
        elapsed = time.time() - t_start
        eta = elapsed * (10 - task_id - 1) / (task_id + 1) / 60
        print(f"  task {task_id}: {task_succ}/{n_episodes_per_task} "
              f"({task_succ/n_episodes_per_task:.0%}), ETA {eta:.1f} min")

    overall_sr = total_succ / total_ep
    print(f"  → seed {seed} overall: {total_succ}/{total_ep} = {overall_sr:.1%}")

    return {
        "seed": seed,
        "ckpt": str(ckpt_path),
        "per_task": per_task_results,
        "overall_sr": overall_sr,
        "total_successes": total_succ,
        "total_episodes": total_ep,
        "wall_clock_min": (time.time() - t_start) / 60,
    }


def _run_one_episode(env, initial_state, predictor, task_description, max_steps=220):
    """Minimal episode loop matching OTPSoftPredictor's expected obs schema."""
    from collections import deque

    env.reset()
    obs = env.set_init_state(initial_state)
    action_queue = deque(maxlen=8)
    num_steps_wait = 10

    for t in range(max_steps + num_steps_wait):
        if t < num_steps_wait:
            # Dummy action during warm-up (OFT convention)
            dummy = np.zeros(7, dtype=np.float32)
            dummy[-1] = -1.0  # gripper open
            obs, _, done, _ = env.step(dummy.tolist())
            continue

        if len(action_queue) == 0:
            chunk = predictor.predict_chunk(obs, task_description)
            # chunk shape: (1, H, 7) — use action sequence
            for a in chunk[0]:
                action_queue.append(a)

        action = action_queue.popleft()
        obs, _, done, _ = env.step(action.tolist())
        if done:
            return True

    return False


def aggregate(seed_results: List[Dict]) -> Dict:
    """Aggregate SR across seeds: mean ± std per-task and overall."""
    n_seeds = len(seed_results)
    per_task_aggregated = {}
    for task_id in range(10):
        srs = [r["per_task"][task_id]["sr"] for r in seed_results]
        per_task_aggregated[task_id] = {
            "mean": float(np.mean(srs)),
            "std": float(np.std(srs)),
            "min": float(np.min(srs)),
            "max": float(np.max(srs)),
            "per_seed": {str(r["seed"]): r["per_task"][task_id]["sr"] for r in seed_results},
        }

    overall_srs = [r["overall_sr"] for r in seed_results]
    overall = {
        "mean": float(np.mean(overall_srs)),
        "std": float(np.std(overall_srs)),
        "min": float(np.min(overall_srs)),
        "max": float(np.max(overall_srs)),
        "per_seed": {str(r["seed"]): r["overall_sr"] for r in seed_results},
    }

    return {
        "n_seeds": n_seeds,
        "seeds_evaluated": [r["seed"] for r in seed_results],
        "per_task": per_task_aggregated,
        "overall": overall,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seeds", nargs="+", type=int, default=[42, 137, 2026])
    parser.add_argument("--n_episodes", type=int, default=50,
                       help="Episodes per task per seed")
    args = parser.parse_args()

    print("=" * 70)
    print("Path B Step 6: Multi-seed eval aggregation")
    print("=" * 70)
    print(f"Seeds: {args.seeds}")
    print(f"Episodes per (task, seed): {args.n_episodes}")
    print(f"Total episodes: {len(args.seeds) * 10 * args.n_episodes}")

    # Patch torch.load (LIBERO files use numpy reconstructor)
    _orig = torch.load
    def _patched(*a, **k):
        k.setdefault("weights_only", False)
        return _orig(*a, **k)
    torch.load = _patched

    # Sys path setup for OFT repo (LIBERO utils)
    OFT_REPO = Path("/root/autodl-tmp/third_party/openvla-oft")
    sys.path.insert(0, str(OFT_REPO))
    sys.path.insert(0, str(OFT_REPO / "experiments" / "robot" / "libero"))

    seed_results = []
    for seed in args.seeds:
        try:
            ckpt_path = find_latest_ckpt_for_seed(seed)
            result = run_sim_eval(ckpt_path, seed, args.n_episodes)
            seed_results.append(result)
        except FileNotFoundError as e:
            print(f"\nSKIP seed {seed}: {e}")
            continue

    if len(seed_results) == 0:
        print("\nERROR: no seeds successfully evaluated")
        sys.exit(1)

    aggregate_result = aggregate(seed_results)

    # Print summary
    print("\n" + "=" * 70)
    print("Aggregate results")
    print("=" * 70)
    print(f"\nOverall SR: {aggregate_result['overall']['mean']:.1%} ± "
          f"{aggregate_result['overall']['std']:.1%}")
    print(f"  Range: [{aggregate_result['overall']['min']:.1%}, "
          f"{aggregate_result['overall']['max']:.1%}]")
    print(f"  Per-seed: {aggregate_result['overall']['per_seed']}")

    print(f"\nPer-task SR (mean ± std):")
    for task_id, stats in aggregate_result["per_task"].items():
        print(f"  task {task_id}: {stats['mean']:.1%} ± {stats['std']:.1%}  "
              f"(min {stats['min']:.0%}, max {stats['max']:.0%})")

    # Save
    out = {
        "version": "pathB_step6_multiseed_v1",
        "n_episodes_per_task_per_seed": args.n_episodes,
        "per_seed_results": seed_results,
        "aggregate": aggregate_result,
    }
    out_path = OUT_DIR / "pathB_step6_multiseed.json"
    with open(out_path, "w") as f:
        json.dump(out, f, indent=2)
    print(f"\nWritten: {out_path}")


if __name__ == "__main__":
    main()
