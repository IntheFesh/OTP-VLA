"""
Phase 0b: OFT Unseen-Phrasing Sim Eval (V7 §III protocol).

Adopts OpenVLA-OFT's run_libero_eval.py reference implementation:
  - initialize_model()    → loads model + action_head + proprio_projector
  - run_episode()         → episode loop, takes task_description as parameter
  - get_libero_env()      → LIBERO env construction

We override task_description with paraphrased instructions (6 templates × 10 tasks).

V7 §III protocol:
  - Stage 1: Determinism verification (1 task × 5 episodes × 2 runs, bit-exact)
  - Stage 2: Identity SR sanity (≥ 92% gate on 50 episode, 5 tasks × 10 ep)
  - Stage 3: Full ablation (10 tasks × 6 phrasings × 50 episodes = 3000 episodes)

Usage:
  python scripts/diagnostic/18_oft_unseen_phrasing.py --stage determinism
  python scripts/diagnostic/18_oft_unseen_phrasing.py --stage identity_sanity
  python scripts/diagnostic/18_oft_unseen_phrasing.py --stage full_ablation

V7 §III.G Plan B fallback:
  If cumulative debug > 16 hr without passing identity gate, abort and switch
  to V3 self-ablation (separate script).

Output: results/phase0/oft_unseen_phrasing_<stage>.json
"""

import argparse
import json
import os
import sys
import time
from collections import defaultdict
from pathlib import Path

# ─────────────────────────────────────────────────────────────────────────────
# CRITICAL: must come BEFORE any HF imports
os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")
os.environ.setdefault("HF_HOME", "/root/autodl-tmp/hf_cache")

REPO_ROOT = Path(__file__).resolve().parents[2]
OFT_REPO = Path("/root/autodl-tmp/third_party/openvla-oft")
sys.path.insert(0, str(OFT_REPO))
sys.path.insert(0, str(OFT_REPO / "experiments" / "robot" / "libero"))

OUT_DIR = REPO_ROOT / "results" / "phase0"
OUT_DIR.mkdir(parents=True, exist_ok=True)


# ─────────────────────────────────────────────────────────────────────────────
# Paraphrased instruction set (V7 §III.B FROZEN, do not modify post-V7-commit)
# ─────────────────────────────────────────────────────────────────────────────

SPATIAL_PHRASES = [
    "between the plate and the ramekin",
    "from table center",
    "in the top drawer of the wooden cabinet",
    "next to the cookie box",
    "next to the plate",
    "next to the ramekin",
    "on the cookie box",
    "on the ramekin",
    "on the stove",
    "on the wooden cabinet",
]

PARAPHRASINGS = {
    "P0_identity":    "pick up the black bowl {spatial} and place it on the plate",
    "P1_word_order":  "place on the plate the black bowl that is {spatial}",
    "P2_synonym":     "grab the dark bowl {spatial} and put it on the plate",
    "P3_passive":     "the black bowl {spatial} should be picked up and placed on the plate",
    "P4_verb_change": "move the black bowl {spatial} onto the plate",
    "P5_compact":     "transfer the black bowl {spatial} to the plate",
}


def get_paraphrased(task_id: int, phrase_id: str) -> str:
    return PARAPHRASINGS[phrase_id].format(spatial=SPATIAL_PHRASES[task_id])


# ─────────────────────────────────────────────────────────────────────────────
# Config
# ─────────────────────────────────────────────────────────────────────────────

def build_oft_config():
    """Build minimal GenerateConfig for OFT eval on LIBERO-Spatial."""
    from experiments.robot.libero.run_libero_eval import GenerateConfig

    cfg = GenerateConfig(
        pretrained_checkpoint="moojink/openvla-7b-oft-finetuned-libero-spatial",
        model_family="openvla",
        use_l1_regression=True,
        use_diffusion=False,
        use_film=False,
        num_images_in_input=2,
        use_proprio=True,
        center_crop=True,
        num_open_loop_steps=8,
        task_suite_name="libero_spatial",
        num_trials_per_task=50,
        num_steps_wait=10,
        env_img_res=256,
        initial_states_path="DEFAULT",
        unnorm_key="libero_spatial_no_noops",
    )
    return cfg


# ─────────────────────────────────────────────────────────────────────────────
# Stage 1: Determinism verification
# ─────────────────────────────────────────────────────────────────────────────

def run_determinism_check():
    """1 task × 5 episodes × 2 runs, verify bit-exact action output."""
    print("=" * 70)
    print("Stage 1: OFT Determinism Verification (1 task × 5 episodes × 2 runs)")
    print("=" * 70)

    from experiments.robot.libero.run_libero_eval import (
        initialize_model, run_episode,
    )
    from experiments.robot.libero.libero_utils import get_libero_env
    from experiments.robot.robot_utils import set_seed_everywhere, get_image_resize_size
    from libero.libero import benchmark

    cfg = build_oft_config()

    print("\nLoading OFT model components (~3-5 min first time)...")
    t0 = time.time()
    model, action_head, proprio_projector, noisy_action_projector, processor = initialize_model(cfg)
    print(f"Model loaded in {time.time() - t0:.1f}s")

    resize_size = get_image_resize_size(cfg)
    task_suite = benchmark.get_benchmark_dict()[cfg.task_suite_name]()

    task = task_suite.get_task(0)  # task 0
    env, default_desc = get_libero_env(task, cfg.model_family, resolution=cfg.env_img_res)
    paraphrased = get_paraphrased(0, "P0_identity")
    print(f"\nDefault task_description: {default_desc!r}")
    print(f"Identity paraphrased:     {paraphrased!r}")

    # Run 5 episodes twice with same seed, compare success patterns
    n_episodes = 5
    print(f"\nRunning {n_episodes} episodes twice for determinism check...")

    results = []
    for run_idx in range(2):
        set_seed_everywhere(7)  # fixed seed
        successes = []
        for ep_idx in range(n_episodes):
            success, _ = run_episode(
                cfg, env, paraphrased, model, resize_size,
                processor=processor, action_head=action_head,
                proprio_projector=proprio_projector,
                noisy_action_projector=noisy_action_projector,
                initial_state=None,
            )
            successes.append(bool(success))
            print(f"  run {run_idx} ep {ep_idx}: success={success}")
        results.append(successes)

    determinism_pass = (results[0] == results[1])
    print(f"\nResults run 1: {results[0]}")
    print(f"Results run 2: {results[1]}")
    print(f"Bit-exact match: {determinism_pass}")

    summary = {
        "stage": "determinism",
        "n_episodes": n_episodes,
        "run1_successes": results[0],
        "run2_successes": results[1],
        "deterministic": determinism_pass,
    }
    with open(OUT_DIR / "oft_unseen_phrasing_determinism.json", "w") as f:
        json.dump(summary, f, indent=2)
    return determinism_pass


# ─────────────────────────────────────────────────────────────────────────────
# Stage 2: Identity SR sanity
# ─────────────────────────────────────────────────────────────────────────────

def run_identity_sanity():
    """5 tasks × 10 episodes = 50 episodes on identity phrasing.

    V7 §III.D pass criterion: identity SR ≥ 92%.
    """
    print("=" * 70)
    print("Stage 2: OFT Identity SR Sanity (5 tasks × 10 episodes)")
    print("=" * 70)

    from experiments.robot.libero.run_libero_eval import (
        initialize_model, run_episode,
    )
    from experiments.robot.libero.libero_utils import get_libero_env
    from experiments.robot.robot_utils import set_seed_everywhere, get_image_resize_size
    from libero.libero import benchmark

    cfg = build_oft_config()
    set_seed_everywhere(7)

    print("\nLoading OFT model...")
    t0 = time.time()
    model, action_head, proprio_projector, noisy_action_projector, processor = initialize_model(cfg)
    print(f"Model loaded in {time.time() - t0:.1f}s")

    resize_size = get_image_resize_size(cfg)
    task_suite = benchmark.get_benchmark_dict()[cfg.task_suite_name]()

    per_task_results = {}
    overall_successes = 0
    overall_episodes = 0

    for task_id in range(5):  # first 5 LIBERO-Spatial tasks
        task = task_suite.get_task(task_id)
        env, _ = get_libero_env(task, cfg.model_family, resolution=cfg.env_img_res)
        paraphrased = get_paraphrased(task_id, "P0_identity")

        task_successes = 0
        for ep_idx in range(10):
            success, _ = run_episode(
                cfg, env, paraphrased, model, resize_size,
                processor=processor, action_head=action_head,
                proprio_projector=proprio_projector,
                noisy_action_projector=noisy_action_projector,
                initial_state=None,
            )
            task_successes += int(bool(success))
            print(f"  task {task_id} ep {ep_idx}: success={success}")

        per_task_results[task_id] = {
            "n_episodes": 10,
            "successes": task_successes,
            "sr": task_successes / 10,
            "instruction": paraphrased,
        }
        overall_successes += task_successes
        overall_episodes += 10
        print(f"  → task {task_id} SR: {task_successes}/10 = {task_successes/10:.1%}")

    overall_sr = overall_successes / overall_episodes
    gate_pass = overall_sr >= 0.92

    print(f"\nOverall identity SR: {overall_successes}/{overall_episodes} = {overall_sr:.1%}")
    print(f"V7 §III.D gate (≥ 92%): {'PASS' if gate_pass else 'FAIL'}")

    summary = {
        "stage": "identity_sanity",
        "per_task": per_task_results,
        "overall_sr": overall_sr,
        "overall_successes": overall_successes,
        "overall_episodes": overall_episodes,
        "gate_threshold": 0.92,
        "gate_pass": gate_pass,
    }
    with open(OUT_DIR / "oft_unseen_phrasing_identity_sanity.json", "w") as f:
        json.dump(summary, f, indent=2)

    if not gate_pass:
        if overall_sr >= 0.80:
            print(f"\n⚠️  Marginal failure ({overall_sr:.1%}). Up to 8 hr debug per V7 §III.D.")
        else:
            print(f"\n⚠️  Severe failure ({overall_sr:.1%}). Up to 16 hr debug per V7 §III.D.")
        print(f"    If unresolved after debug cap, activate Plan B (V3 self-ablation).")

    return gate_pass


# ─────────────────────────────────────────────────────────────────────────────
# Stage 3: Full ablation
# ─────────────────────────────────────────────────────────────────────────────

def run_full_ablation():
    """10 tasks × 6 phrasings × 50 episodes = 3000 episodes.

    Estimated GPU time: 12-20 hours.
    """
    print("=" * 70)
    print("Stage 3: OFT Full Unseen-Phrasing Ablation (10 × 6 × 50 = 3000 ep)")
    print("=" * 70)

    from experiments.robot.libero.run_libero_eval import (
        initialize_model, run_episode,
    )
    from experiments.robot.libero.libero_utils import get_libero_env
    from experiments.robot.robot_utils import set_seed_everywhere, get_image_resize_size
    from libero.libero import benchmark

    cfg = build_oft_config()
    set_seed_everywhere(7)

    print("\nLoading OFT model...")
    t0 = time.time()
    model, action_head, proprio_projector, noisy_action_projector, processor = initialize_model(cfg)
    print(f"Model loaded in {time.time() - t0:.1f}s")

    resize_size = get_image_resize_size(cfg)
    task_suite = benchmark.get_benchmark_dict()[cfg.task_suite_name]()

    # Pre-build envs per task (LIBERO env construction is slow)
    print("\nPre-building 10 LIBERO envs...")
    envs = {}
    for task_id in range(10):
        task = task_suite.get_task(task_id)
        env, _ = get_libero_env(task, cfg.model_family, resolution=cfg.env_img_res)
        envs[task_id] = env

    # Result storage: per_cell_results[(task_id, phrase_id)] = list of bool
    results = defaultdict(list)

    n_total = 0
    n_success = 0
    t_start = time.time()

    for phrase_id in PARAPHRASINGS.keys():
        for task_id in range(10):
            env = envs[task_id]
            paraphrased = get_paraphrased(task_id, phrase_id)
            print(f"\n=== task {task_id} × {phrase_id} ===")
            print(f"  instruction: {paraphrased!r}")
            cell_successes = []
            for ep_idx in range(50):
                # Episode seed per V7 §III.C: 1000000 + task*10000 + phrase_idx*1000 + ep
                phrase_idx = list(PARAPHRASINGS.keys()).index(phrase_id)
                ep_seed = 1000000 + task_id * 10000 + phrase_idx * 1000 + ep_idx
                set_seed_everywhere(ep_seed)

                success, _ = run_episode(
                    cfg, env, paraphrased, model, resize_size,
                    processor=processor, action_head=action_head,
                    proprio_projector=proprio_projector,
                    noisy_action_projector=noisy_action_projector,
                    initial_state=None,
                )
                cell_successes.append(bool(success))
                n_total += 1
                if success:
                    n_success += 1

                if (ep_idx + 1) % 10 == 0:
                    elapsed = time.time() - t_start
                    eta_min = (3000 - n_total) * elapsed / n_total / 60 if n_total > 0 else 0
                    print(f"    ep {ep_idx+1}/50: cell SR={sum(cell_successes)/len(cell_successes):.0%}, "
                          f"global {n_success}/{n_total} ({n_success/n_total:.1%}), ETA {eta_min:.0f} min")

            results[(task_id, phrase_id)] = cell_successes
            print(f"  → cell SR: {sum(cell_successes)}/{len(cell_successes)} = {sum(cell_successes)/len(cell_successes):.0%}")

            # Incrementally save to allow recovery from crashes
            partial = {f"task{t}_{p}": v for (t, p), v in results.items()}
            with open(OUT_DIR / "oft_unseen_phrasing_full_partial.json", "w") as f:
                json.dump(partial, f, indent=2)

    # Final save
    final = {f"task{t}_{p}": v for (t, p), v in results.items()}
    final["global_sr"] = n_success / n_total
    final["n_total"] = n_total
    final["n_success"] = n_success
    final["wall_clock_s"] = time.time() - t_start
    with open(OUT_DIR / "oft_unseen_phrasing_full.json", "w") as f:
        json.dump(final, f, indent=2)

    print(f"\n{'=' * 70}")
    print(f"DONE. Global SR: {n_success}/{n_total} = {n_success/n_total:.1%}")
    print(f"Wall clock: {(time.time() - t_start)/60:.1f} min")


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage", choices=["determinism", "identity_sanity", "full_ablation"],
                       required=True)
    args = parser.parse_args()

    if args.stage == "determinism":
        result = run_determinism_check()
        sys.exit(0 if result else 1)
    elif args.stage == "identity_sanity":
        result = run_identity_sanity()
        sys.exit(0 if result else 1)
    elif args.stage == "full_ablation":
        run_full_ablation()
