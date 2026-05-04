#!/usr/bin/env python
"""
scripts/02_validate_controller_with_gt.py

Stage 2 critical gate (§5.1 — A1 empirical validation):
Validate the FixedManipulationController end-to-end by feeding it
GROUND-TRUTH object trajectories (extracted from LIBERO demos) and
checking whether it can complete each task in sim.

This is NOT a test of the OTP head.  It is a test of φ_T — does the
fixed controller, given a perfect trajectory prediction, actually
manipulate the object successfully?  If GT replay fails, the OTP head
cannot rescue it; if GT replay succeeds, then OTP-head failures
downstream are head-prediction problems, not controller problems.

PASS criterion (§REPO_LAYOUT § validation):
    ≥ 8 / 10 tasks reach success rate ≥ 80%.

Usage:
    python scripts/02_validate_controller_with_gt.py \\
        --suite libero_spatial \\
        --object-poses-dir data/object_poses/ \\
        --num-episodes-per-task 5 \\
        --max-control-steps 600 \\
        --horizon 8 \\
        --output results/gt_replay_validation.json
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import re
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch

logging.basicConfig(
    level=logging.INFO,
    format="[%(levelname)s] %(asctime)s %(name)s | %(message)s",
)
logger = logging.getLogger("gt_replay")


# ---------------------------------------------------------------------------
# Setup
# ---------------------------------------------------------------------------

def _print_startup(args: argparse.Namespace) -> None:
    print("=" * 72)
    print(f"[ENV]  python={sys.version.split()[0]}  cwd={os.getcwd()}")
    print(f"[CFG]  suite={args.suite}  episodes_per_task={args.num_episodes_per_task}")
    print(f"[CFG]  max_control_steps={args.max_control_steps}  horizon={args.horizon}")
    print(f"[DATA] object_poses_dir={args.object_poses_dir}")
    print(f"[OUT]  output={args.output}")
    print("=" * 72)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _parse_target_object(
    instruction: str,
    object_names: List[str],
    object_poses: np.ndarray,   # (N_obj, T, 4, 4)
) -> int:
    """
    Identify the *target* object — the one being manipulated in this demo.

    Instead of name-matching (which fails when multiple bowls exist with
    arbitrary _1/_2 numbering), we pick the object whose XYZ position
    changes the most over the demo.  In a LIBERO pick-and-place demo,
    only the target object moves substantially; receptacles (plate) and
    distractors (other bowl) are static.

    Returns:
        index into object_names
    """
    # Compute total path length per object: sum of frame-to-frame deltas.
    pos = object_poses[:, :, :3, 3]                        # (N_obj, T, 3)
    deltas = np.linalg.norm(np.diff(pos, axis=1), axis=-1)  # (N_obj, T-1)
    total_motion = deltas.sum(axis=1)                       # (N_obj,)

    # Pick the object with the largest motion.  Tie-break: lower index.
    target_idx = int(np.argmax(total_motion))

    # Sanity check: the target should have moved at least 5cm cumulatively.
    if total_motion[target_idx] < 0.05:
        # Fall back to name-based heuristic.
        for i, name in enumerate(object_names):
            if "black_bowl" in name:
                return i
    return target_idx


def _parse_release_object(instruction: str, object_names: List[str]) -> int:
    """For libero_spatial all tasks release on the plate."""
    instr = instruction.lower()
    for i, name in enumerate(object_names):
        if name == "plate_1":
            return i
    return len(object_names) - 1


def _se3_from_pos_quat(pos: np.ndarray, quat: np.ndarray) -> np.ndarray:
    """Build a 4×4 SE(3) matrix from position + quaternion."""
    from otp.utils.lie_algebra import quat_to_so3
    R = quat_to_so3(torch.from_numpy(quat.astype(np.float32)).unsqueeze(0)).squeeze(0).numpy()
    T = np.eye(4, dtype=np.float64)
    T[:3, :3] = R
    T[:3, 3] = pos.astype(np.float64)
    return T


def _build_xi_window(
    object_poses_full: np.ndarray,   # (N_obj, T, 4, 4)
    t_start: int,
    horizon: int,
) -> torch.Tensor:
    """
    Slice the GT object poses [t_start : t_start+horizon] and convert to
    Lie algebra (N_obj, H, 6) tensor.  Pads with the last frame if the
    trajectory ends within the horizon window.
    """
    from otp.utils.lie_algebra import se3_to_lie

    N_obj, T_total, _, _ = object_poses_full.shape
    t_end = t_start + horizon
    if t_end <= T_total:
        window = object_poses_full[:, t_start:t_end]    # (N_obj, H, 4, 4)
    else:
        # Pad with last frame.
        valid = object_poses_full[:, t_start:T_total]
        pad_n = t_end - T_total
        pad = np.repeat(valid[:, -1:], pad_n, axis=1)
        window = np.concatenate([valid, pad], axis=1)

    window_t = torch.from_numpy(window.astype(np.float32))     # (N_obj, H, 4, 4)
    flat = window_t.reshape(-1, 4, 4)
    xi_flat = se3_to_lie(flat)                                  # (N_obj*H, 6)
    return xi_flat.reshape(N_obj, horizon, 6)


def _extract_obs_state(env_obs, target_object_name: str) -> Tuple[np.ndarray, np.ndarray, float]:
    """
    Pull out (ee_pose, target_object_pose, gripper_state) from a LIBERO env obs.

    gripper_state convention: LIBERO 'robot0_gripper_qpos' is (2,) with
      [+0.04, -0.04] when fully open and [0.0, 0.0] when fully closed.
    We map to a scalar in [-1, +1] where +1 = closed.
    """
    ee_pos = np.asarray(env_obs["robot0_eef_pos"], dtype=np.float64)
    ee_quat = np.asarray(env_obs["robot0_eef_quat"], dtype=np.float32)
    ee_pose = _se3_from_pos_quat(ee_pos, ee_quat)

    obj_pos = np.asarray(env_obs[f"{target_object_name}_pos"], dtype=np.float64)
    obj_quat = np.asarray(env_obs[f"{target_object_name}_quat"], dtype=np.float32)
    obj_pose = _se3_from_pos_quat(obj_pos, obj_quat)

    g_qpos = np.asarray(env_obs["robot0_gripper_qpos"], dtype=np.float64)
    # gripper qpos width in [0, 0.08]; closed → 0, open → 0.08.
    width = float(g_qpos[0] - g_qpos[1])
    # Map width=0.08 → -1, width=0 → +1.
    gripper_scalar = float(np.clip(1.0 - 2.0 * width / 0.08, -1.0, 1.0))

    return ee_pose, obj_pose, gripper_scalar


# ---------------------------------------------------------------------------
# Per-episode runner
# ---------------------------------------------------------------------------

def _run_one_episode(
    env,
    init_state: np.ndarray,
    object_poses_full: np.ndarray,    # (N_obj, T, 4, 4)
    target_object_idx: int,
    target_object_name: str,
    release_target_xy: np.ndarray,
    horizon: int,
    max_control_steps: int,
    controller_kwargs: Optional[dict] = None,
) -> Dict[str, object]:
    """Run one GT-replay rollout and return success + diagnostic info."""
    from otp.controllers.fixed_manipulation_controller import FixedManipulationController

    ctrl = FixedManipulationController(**(controller_kwargs or {}))

    env.reset()
    obs = env.set_init_state(init_state)

    T_gt = object_poses_full.shape[1]
    success = False
    final_phase = "APPROACH"
    timeout_reason = None

    # Set trajectory ONCE — controller is stateful (phase machine, gripper
    # history).  Calling set_trajectory each step would reset() everything
    # and the phase machine would never advance past APPROACH.
    #
    # We feed the *full* GT object trajectory as the controller's xi.
    # During TRANSPORT, the controller advances trajectory_step internally
    # and reads waypoints[trajectory_step] from the H-axis.
    xi_full = _build_xi_window(object_poses_full, t_start=0, horizon=T_gt)
    target_obj_init = object_poses_full[target_object_idx, 0]
    ctrl.set_trajectory(
        xi_full,
        object_pose=target_obj_init,
        release_target_xy=release_target_xy,
    )

    for t in range(max_control_steps):
        ee_pose, obj_pose, gripper = _extract_obs_state(obs, target_object_name)
        action = ctrl.step(ee_pose, obj_pose, gripper)

        obs, reward, done, info = env.step(action)
        final_phase = ctrl.current_phase

        # LIBERO success: reward >= 1 means task complete.
        if float(reward) >= 1.0:
            success = True
            break
        if done:
            timeout_reason = "env_done_no_reward"
            break
        if final_phase == "DONE":
            timeout_reason = "phase_done_no_reward"
            break

    if not success and timeout_reason is None:
        timeout_reason = f"max_steps_at_phase_{final_phase}"

    # Pull last-N phase trace entries for debugging (avoid huge JSON output).
    phase_trace_tail = ctrl.phase_trace[-30:] if ctrl.phase_trace else []

    return {
        "success":            success,
        "final_phase":        final_phase,
        "timeout_reason":     timeout_reason,
        "control_steps":      t + 1,
        "phase_trace_tail":   phase_trace_tail,
    }


# ---------------------------------------------------------------------------
# Per-task runner
# ---------------------------------------------------------------------------

def _run_task(
    suite: str,
    task_dir: Path,
    bm_index: Optional[int],
    bddl_path: str,
    num_episodes: int,
    horizon: int,
    max_control_steps: int,
    controller_kwargs: Optional[dict] = None,
) -> Dict[str, object]:
    """
    Run num_episodes rollouts for a single task using its extracted demos.

    Each episode uses a different demo's init state + GT trajectory.
    """
    from libero.libero.envs import OffScreenRenderEnv

    npz_paths = sorted(task_dir.glob("demo_*.npz"))
    if not npz_paths:
        return {
            "success_rate": 0.0,
            "num_episodes": 0,
            "error": f"no demo .npz files in {task_dir}",
        }
    npz_paths = npz_paths[:num_episodes]

    # Build env once for the task — reused across episodes.
    env = OffScreenRenderEnv(
        bddl_file_name=bddl_path,
        camera_heights=128,
        camera_widths=128,
    )

    successes = 0
    episode_results = []

    try:
        for ep_idx, npz_path in enumerate(npz_paths):
            data = np.load(npz_path, allow_pickle=True)
            object_names = list(data["object_names"])
            object_poses = data["object_poses"]          # (N_obj, T, 4, 4)
            states = data["states"]                       # (T, 92)
            instruction = str(data["instruction"])

            target_idx = _parse_target_object(instruction, object_names, object_poses)
            release_idx = _parse_release_object(instruction, object_names)
            target_name = object_names[target_idx]

            # Log target/distractor disambiguation result.
            pos = object_poses[:, :, :3, 3]
            motions = np.linalg.norm(np.diff(pos, axis=1), axis=-1).sum(axis=1)
            motion_str = ", ".join(
                f"{name}={motions[i]:.2f}m" for i, name in enumerate(object_names)
            )
            logger.info("    target=%s   motions=[%s]", target_name, motion_str)

            # Release target = plate's xy from final frame.
            release_xy = object_poses[release_idx, -1, :2, 3].astype(np.float64)

            ep = _run_one_episode(
                env=env,
                init_state=states[0],
                object_poses_full=object_poses,
                target_object_idx=target_idx,
                target_object_name=target_name,
                release_target_xy=release_xy,
                horizon=horizon,
                max_control_steps=max_control_steps,
                controller_kwargs=controller_kwargs,
            )
            ep["demo_id"]     = npz_path.stem
            ep["target_obj"]  = target_name
            ep["release_xy"]  = release_xy.tolist()
            episode_results.append(ep)
            if ep["success"]:
                successes += 1

            logger.info(
                "    ep %d/%d demo=%s success=%s phase=%s steps=%d reason=%s",
                ep_idx + 1, len(npz_paths), npz_path.stem,
                ep["success"], ep["final_phase"], ep["control_steps"],
                ep["timeout_reason"],
            )
    finally:
        env.close()

    return {
        "success_rate": successes / len(npz_paths),
        "successes":    successes,
        "num_episodes": len(npz_paths),
        "episodes":     episode_results,
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def _resolve_bddl_paths(suite: str, task_dirs: List[Path]) -> Dict[str, str]:
    """For each task dir name, find the matching bddl file path."""
    from libero.libero.benchmark import get_benchmark_dict
    bm = get_benchmark_dict()[suite]()
    mapping: Dict[str, str] = {}
    for task_dir in task_dirs:
        task_stem = task_dir.name
        matched = None
        for i in range(bm.get_num_tasks()):
            bddl_full = bm.get_task_bddl_file_path(i)
            bddl_stem = Path(bddl_full).stem
            if bddl_stem == task_stem or bddl_stem in task_stem or task_stem in bddl_stem:
                matched = bddl_full
                break
        if matched is None:
            raise FileNotFoundError(
                f"No BDDL file matched task dir: {task_stem}"
            )
        mapping[task_stem] = matched
    return mapping


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--suite", type=str, default="libero_spatial")
    parser.add_argument("--object-poses-dir", type=Path, required=True,
                        help="Output of scripts/03_extract_object_poses.py")
    parser.add_argument("--num-episodes-per-task", type=int, default=5)
    parser.add_argument("--max-control-steps", type=int, default=600)
    parser.add_argument("--horizon", type=int, default=8)
    parser.add_argument("--output", type=Path,
                        default=Path("results/gt_replay_validation.json"))
    parser.add_argument("--task-pass-threshold", type=float, default=0.8,
                        help="Per-task success rate ≥ this counts as PASS.")
    parser.add_argument("--num-tasks-pass-threshold", type=int, default=8,
                        help="Need this many tasks above threshold for global PASS.")
    args = parser.parse_args()

    _print_startup(args)
    args.output.parent.mkdir(parents=True, exist_ok=True)

    os.environ.setdefault("MUJOCO_GL", "egl")

    task_dirs = sorted(p for p in args.object_poses_dir.iterdir()
                       if p.is_dir())
    if not task_dirs:
        logger.error("No task subdirectories under %s", args.object_poses_dir)
        sys.exit(1)

    bddl_map = _resolve_bddl_paths(args.suite, task_dirs)

    t0 = time.time()
    per_task_results: Dict[str, dict] = {}
    tasks_passed = 0

    for task_idx, task_dir in enumerate(task_dirs):
        task_name = task_dir.name
        logger.info("[%d/%d] task: %s", task_idx + 1, len(task_dirs), task_name)
        result = _run_task(
            suite=args.suite,
            task_dir=task_dir,
            bm_index=None,
            bddl_path=bddl_map[task_name],
            num_episodes=args.num_episodes_per_task,
            horizon=args.horizon,
            max_control_steps=args.max_control_steps,
            controller_kwargs={
                # APPROACH thresholds: looser xy because OSC PD overshoots
                # and a 5cm-step controller cannot stably stay <5cm.
                "xy_approach_threshold":  0.10,
                # PRE_GRASP → GRASP based on xy alignment with grasp target,
                # not z-descent (z descends in GRASP phase).
                "z_pregrasp_threshold":   0.03,
                # GRASP gripper-stability minimum.
                "grasp_min_steps":        5,
                # TRANSPORT → RELEASE: gripper-carry distance precision.
                "transport_xy_threshold": 0.08,
                "trajectory_done_threshold": 0.85,
                # Looser per-phase timeout — APPROACH alone takes ~140 steps
                # to converge to xy<0.10 with LIBERO OSC's slow PD response.
                "max_phase_steps":        300,
                # Position clip — allow up to LIBERO's max OSC step.
                "pos_clip":               0.05,
                # CRITICAL: zero out rotation delta. LIBERO Panda's home pose
                # already has gripper pointing down; trying to rotate to
                # _R_TOPDOWN gives ±π rad axis-angle deltas that severely
                # couple-perturb the OSC position controller, making EE
                # oscillate randomly instead of converging to target.
                # Tested: with ori_clip=0, P-controller converges from 0.22m
                # to 0.05m in 201 steps with zero oscillation.
                "ori_clip":               0.0,
                # Slight grasp height offset so EE doesn't penetrate object
                # geometry — bowl table-z is ~0.90, grasp at z=0.92 works.
                "grasp_height_offset":    0.04,
                # Approach height offset matches PRE_GRASP hover height.
                "approach_height_offset": 0.10,
            },
        )
        per_task_results[task_name] = result
        if result.get("success_rate", 0.0) >= args.task_pass_threshold:
            tasks_passed += 1
            verdict = "PASS"
        else:
            verdict = "FAIL"
        logger.info(
            "  -> %s  success_rate=%.2f  (%d/%d episodes)",
            verdict,
            result.get("success_rate", 0.0),
            result.get("successes", 0),
            result.get("num_episodes", 0),
        )

    elapsed = round(time.time() - t0, 2)
    global_pass = tasks_passed >= args.num_tasks_pass_threshold

    output = {
        "suite":                   args.suite,
        "num_episodes_per_task":   args.num_episodes_per_task,
        "horizon":                 args.horizon,
        "max_control_steps":       args.max_control_steps,
        "task_pass_threshold":     args.task_pass_threshold,
        "tasks_passed":            tasks_passed,
        "tasks_total":             len(task_dirs),
        "global_verdict":          "PASS" if global_pass else "FAIL",
        "elapsed_sec":             elapsed,
        "per_task_results":        per_task_results,
    }
    with open(args.output, "w") as f:
        json.dump(output, f, indent=2, default=lambda o: str(o))

    print()
    print("=" * 72)
    print(f"GT REPLAY VALIDATION — {output['global_verdict']}")
    print(f"  tasks passed:  {tasks_passed} / {len(task_dirs)}  "
          f"(threshold ≥ {args.num_tasks_pass_threshold})")
    print(f"  per-task success rates:")
    for name, res in per_task_results.items():
        print(f"    {res.get('success_rate', 0.0):.2f}  "
              f"{name}")
    print(f"  elapsed: {elapsed}s")
    print(f"  full output: {args.output}")
    print("=" * 72)
    sys.exit(0 if global_pass else 1)


if __name__ == "__main__":
    main()
