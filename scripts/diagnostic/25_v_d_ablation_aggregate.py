"""
scripts/diagnostic/25_v_d_ablation_aggregate.py

§V.D ablation aggregate analysis script.

Reads results from 21 (or 18) ablation retrains and produces:
  - Per-cell × per-task SR table (3 seeds aggregated)
  - Pairwise McNemar tests (BH-FDR adjusted)
  - Wilson 95% CI per cell × task
  - Welch's t-test for seed-level SR (cell pair comparisons)
  - LaTeX-ready table for §V.D
  - JSON detail report

Reuses existing codebase machinery:
  - eval_standard.py / 20_pathB_multiseed_aggregate.py for SR rollout
  - 21_stage3_mcnemar_analysis.py for McNemar (continuity-corrected)
  - 22_stage3_main_analysis.py for Benjamini-Hochberg FDR
  - scipy.stats for Welch's t-test and Wilson CI (via norm)

Usage:
  # Step 1: Collect SR for all 21 (or 18) retrains
  python scripts/diagnostic/25_v_d_ablation_aggregate.py \
      --phase collect \
      --results-root results \
      --output-dir results/phase0/v_d \
      --n-episodes-per-task 50

  # Step 2: Run stats + LaTeX table (after Phase A complete)
  python scripts/diagnostic/25_v_d_ablation_aggregate.py \
      --phase analyze \
      --sr-data results/phase0/v_d/all_sr_data.json \
      --output-dir results/phase0/v_d

Pre-registration link: notes/v_d_ablation_preregistration_2026-05-13.md
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
import time
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from scipy.stats import chi2, norm, ttest_ind

# MUJOCO_GL must be set BEFORE importing libero (per eval_standard.py)
os.environ.setdefault("MUJOCO_GL", "egl")

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

logger = logging.getLogger("v_d_aggregate")
logging.basicConfig(
    format="%(asctime)s [%(levelname)s] %(message)s", level=logging.INFO,
)


# =============================================================================
# Cell definitions (matches preregistration §1.1 + §1.2)
# =============================================================================

# 21-cell version (variant c passes mini E2E gate)
CELLS_FULL = {
    "A_pathB":       {"description": "Path B base (det head + CFM dec)",
                      "config_prefix": "otp_soft_pathB_seed",
                      "run_dir_prefix": "pathB_step6_seed"},  # legacy naming
    "B_pathB_cocos": {"description": "Path B + Cocos source",
                      "config_prefix": "otp_soft_ablation_pathB_cocos_seed",
                      "run_dir_prefix": "ablation_pathB_cocos_seed"},
    "C_variantc":    {"description": "Variant (c) CFM head + det decoder",
                      "config_prefix": "otp_soft_ablation_variantc_seed",
                      "run_dir_prefix": "ablation_variantc_seed"},
    "D_v3":          {"description": "V3 baseline (CFM head + CFM dec)",
                      "config_prefix": "otp_soft_ablation_v3_seed",
                      "run_dir_prefix": "ablation_v3_seed"},
    "E_drop_mesh":   {"description": "V3 + drop mesh",
                      "config_prefix": "otp_soft_ablation_drop_mesh_seed",
                      "run_dir_prefix": "ablation_drop_mesh_seed"},
    "F_drop_grasp":  {"description": "V3 + drop grasp",
                      "config_prefix": "otp_soft_ablation_drop_grasp_seed",
                      "run_dir_prefix": "ablation_drop_grasp_seed"},
    "G_drop_proprio":{"description": "V3 + drop proprio",
                      "config_prefix": "otp_soft_ablation_drop_proprio_seed",
                      "run_dir_prefix": "ablation_drop_proprio_seed"},
}

# Fallback if variant c dropped (18-cell version)
CELLS_NO_VARIANTC = {k: v for k, v in CELLS_FULL.items() if k != "C_variantc"}

SEEDS = [42, 137, 2026]
TASK_IDS = list(range(10))  # LIBERO-Spatial 10 tasks
ALPHA = 0.05
SR_PRACTICAL_EQUIVALENCE_PP = 0.02  # ±2pp


# =============================================================================
# Stats — Reused from 21_stage3 / 22_stage3
# =============================================================================

def mcnemar_pvalue(outcomes_a: List[bool], outcomes_b: List[bool]) -> Tuple[float, float, int, int]:
    """
    Continuity-corrected McNemar test on paired binary outcomes.
    Copied from scripts/diagnostic/21_stage3_mcnemar_analysis.py.
    """
    assert len(outcomes_a) == len(outcomes_b), "Paired McNemar requires equal-length"
    n_01 = sum(1 for (a, b) in zip(outcomes_a, outcomes_b) if a and not b)
    n_10 = sum(1 for (a, b) in zip(outcomes_a, outcomes_b) if not a and b)
    if n_01 + n_10 == 0:
        return 0.0, 1.0, n_01, n_10
    chi2_stat = (abs(n_01 - n_10) - 1) ** 2 / (n_01 + n_10)
    p_value = 1.0 - chi2.cdf(chi2_stat, df=1)
    return float(chi2_stat), float(p_value), n_01, n_10


def benjamini_hochberg(pvalues: List[float], alpha: float = 0.05) -> Tuple[List[float], List[bool]]:
    """
    Benjamini-Hochberg FDR correction.
    Returns (q_values, reject_flags) of same length as input.
    Adapted from scripts/diagnostic/22_stage3_main_analysis.py.
    """
    n = len(pvalues)
    if n == 0:
        return [], []
    p_arr = np.array(pvalues)
    order = np.argsort(p_arr)
    ranked = p_arr[order]
    q_raw = ranked * n / np.arange(1, n + 1)
    # Enforce monotonicity (reverse cummin)
    q_mono = np.minimum.accumulate(q_raw[::-1])[::-1]
    q_clip = np.minimum(q_mono, 1.0)
    # Restore original order
    q_final = np.empty(n)
    q_final[order] = q_clip
    reject = q_final < alpha
    return q_final.tolist(), reject.tolist()


def wilson_ci(n_success: int, n_total: int, alpha: float = 0.05) -> Tuple[float, float]:
    """Wilson score interval for binomial proportion. n_success ≤ n_total."""
    if n_total == 0:
        return 0.0, 1.0
    z = norm.ppf(1 - alpha / 2)
    p_hat = n_success / n_total
    denom = 1 + z**2 / n_total
    center = (p_hat + z**2 / (2 * n_total)) / denom
    half = z * np.sqrt(p_hat * (1 - p_hat) / n_total + z**2 / (4 * n_total**2)) / denom
    return max(0.0, center - half), min(1.0, center + half)


def welchs_ttest(sample_a: List[float], sample_b: List[float]) -> Tuple[float, float]:
    """Welch's t-test (unequal variance). Returns (t_stat, p_value)."""
    if len(sample_a) < 2 or len(sample_b) < 2:
        return float("nan"), 1.0
    res = ttest_ind(sample_a, sample_b, equal_var=False)
    return float(res.statistic), float(res.pvalue)


def practical_equivalence_via_ci(
    sr_a: float, sr_b: float, n_episodes: int, threshold_pp: float = SR_PRACTICAL_EQUIVALENCE_PP
) -> Tuple[bool, Tuple[float, float]]:
    """
    Check if the 95% CI of (SR_a - SR_b) lies entirely within ±threshold_pp.
    Approximate Wald CI for paired SR diff.
    Returns (is_equivalent, ci).
    """
    diff = sr_a - sr_b
    # Approximate variance for difference in proportions (assuming independent)
    var_a = sr_a * (1 - sr_a) / n_episodes
    var_b = sr_b * (1 - sr_b) / n_episodes
    se = np.sqrt(var_a + var_b)
    z = norm.ppf(0.975)
    lo, hi = diff - z * se, diff + z * se
    equiv = (-threshold_pp <= lo) and (hi <= threshold_pp)
    return equiv, (float(lo), float(hi))


# =============================================================================
# Phase A — SR Collection
# =============================================================================

def find_latest_ckpt(run_dir: Path) -> Optional[Path]:
    ckpts = sorted(run_dir.glob("ckpt_step*.pt"))
    return ckpts[-1] if ckpts else None


def find_run_dir(results_root: Path, run_dir_prefix: str, seed: int) -> Optional[Path]:
    """Find the most-recent run dir matching pattern: {prefix}{seed}_*."""
    pattern = f"{run_dir_prefix}{seed}_*"
    matches = sorted(results_root.glob(pattern))
    return matches[-1] if matches else None


def collect_sr_for_cell_seed(
    cell_id: str, cell_spec: Dict, seed: int,
    results_root: Path, n_episodes_per_task: int, output_dir: Path,
) -> Optional[Dict]:
    """
    Run LIBERO eval for one (cell, seed) combination.

    Reuses logic from scripts/diagnostic/20_pathB_multiseed_aggregate.py.
    For each task: produces list of bool outcomes (success/fail per episode).
    """
    run_dir = find_run_dir(results_root, cell_spec["run_dir_prefix"], seed)
    if run_dir is None:
        logger.warning(f"  [{cell_id} seed={seed}] run dir not found")
        return None

    ckpt = find_latest_ckpt(run_dir)
    if ckpt is None:
        logger.warning(f"  [{cell_id} seed={seed}] no ckpt in {run_dir}")
        return None

    config_path = REPO_ROOT / "configs" / f"{cell_spec['config_prefix']}{seed}.yaml"
    if not config_path.exists():
        logger.error(f"  [{cell_id} seed={seed}] config not found: {config_path}")
        return None

    logger.info(f"  [{cell_id} seed={seed}] ckpt={ckpt.name}")

    # Check if SR already computed (resumable)
    cell_seed_out = output_dir / f"sr_{cell_id}_seed{seed}.json"
    if cell_seed_out.exists():
        logger.info(f"    cached: {cell_seed_out}")
        return json.loads(cell_seed_out.read_text())

    # Run LIBERO eval (mirror pathB_multiseed_aggregate logic)
    try:
        from otp.eval.predictor import OTPSoftPredictor
        from libero.libero import benchmark
        from experiments.robot.libero.libero_utils import get_libero_env
        import torch
    except Exception as e:
        logger.error(f"    Import failure: {e}")
        return None

    predictor = OTPSoftPredictor(
        ckpt_path=str(ckpt),
        config_path=str(config_path),
        grasp_affordance_dir="data/grasp_affordances",
        device=torch.device("cuda"),
        log_diagnostics=False,
    )

    task_suite = benchmark.get_benchmark_dict()["libero_spatial"]()
    per_task_outcomes: Dict[int, List[bool]] = {}
    t_start = time.time()

    for task_id in TASK_IDS:
        task = task_suite.get_task(task_id)
        env, task_description = get_libero_env(task, "openvla", resolution=256)
        init_states = task_suite.get_task_init_states(task_id)
        outcomes: List[bool] = []
        for ep_idx in range(n_episodes_per_task):
            # Deterministic seed (matches eval_standard.py convention)
            episode_seed = 5000000 + seed * 100000 + task_id * 1000 + ep_idx
            predictor.reset(episode_seed=episode_seed)
            success = _run_one_episode(
                env, init_states[ep_idx % len(init_states)],
                predictor, task_description, max_steps=220,
            )
            outcomes.append(bool(success))
        per_task_outcomes[task_id] = outcomes
        sr = sum(outcomes) / len(outcomes)
        elapsed = time.time() - t_start
        eta = elapsed * (10 - task_id - 1) / (task_id + 1) / 60
        logger.info(f"    task {task_id}: SR={sr:.1%} ({sum(outcomes)}/{n_episodes_per_task}) "
                    f"ETA {eta:.1f}min")

    result = {
        "cell_id": cell_id,
        "seed": seed,
        "ckpt": str(ckpt),
        "config": str(config_path),
        "n_episodes_per_task": n_episodes_per_task,
        "per_task_outcomes": {str(k): v for k, v in per_task_outcomes.items()},
        "wall_clock_min": (time.time() - t_start) / 60,
    }
    cell_seed_out.write_text(json.dumps(result, indent=2))
    return result


def _run_one_episode(env, initial_state, predictor, task_description, max_steps=220) -> bool:
    """Mirror of pathB_multiseed_aggregate._run_one_episode."""
    from collections import deque
    import numpy as np_local
    env.reset()
    obs = env.set_init_state(initial_state)
    action_queue = deque(maxlen=8)
    num_steps_wait = 10
    for t in range(max_steps + num_steps_wait):
        if t < num_steps_wait:
            dummy = np_local.zeros(7, dtype=np_local.float32)
            dummy[-1] = -1.0
            obs, _, done, _ = env.step(dummy.tolist())
            continue
        if len(action_queue) == 0:
            chunk = predictor.predict_chunk(obs, task_description)
            for a in chunk:
                action_queue.append(a)
        action = action_queue.popleft()
        obs, _, done, info = env.step(action.tolist() if hasattr(action, "tolist") else list(action))
        if env.check_success():
            return True
        if done:
            return False
    return False


def phase_a_collect(
    results_root: Path, output_dir: Path, n_episodes_per_task: int,
    cells: Dict[str, Dict],
) -> Dict:
    """Collect SR for all cells × seeds. Returns aggregated dict."""
    output_dir.mkdir(parents=True, exist_ok=True)
    all_data = {}
    for cell_id, cell_spec in cells.items():
        all_data[cell_id] = {}
        for seed in SEEDS:
            logger.info(f"--- {cell_id} seed={seed} ---")
            result = collect_sr_for_cell_seed(
                cell_id, cell_spec, seed, results_root,
                n_episodes_per_task, output_dir,
            )
            if result:
                all_data[cell_id][str(seed)] = result

    # Write aggregated SR data
    sr_data_path = output_dir / "all_sr_data.json"
    sr_data_path.write_text(json.dumps(all_data, indent=2))
    logger.info(f"All SR data written: {sr_data_path}")
    return all_data


# =============================================================================
# Phase B-F — Statistical Analysis + LaTeX
# =============================================================================

def aggregate_per_cell_per_task(all_data: Dict) -> Dict:
    """
    For each (cell, task), aggregate outcomes across 3 seeds → 150 outcomes.
    Returns: {cell_id: {task_id: {outcomes: [bool×150], n_success, n_total, sr}}}
    """
    out = {}
    for cell_id, seed_data in all_data.items():
        out[cell_id] = {}
        for task_id in TASK_IDS:
            combined = []
            for seed in SEEDS:
                seed_str = str(seed)
                if seed_str not in seed_data:
                    continue
                task_outcomes = seed_data[seed_str].get("per_task_outcomes", {}).get(str(task_id), [])
                combined.extend(task_outcomes)
            if combined:
                out[cell_id][task_id] = {
                    "outcomes": combined,
                    "n_success": sum(combined),
                    "n_total": len(combined),
                    "sr": sum(combined) / len(combined),
                }
    return out


def aggregate_per_cell_per_seed(all_data: Dict) -> Dict:
    """For each (cell, seed), return aggregate SR. Used for Welch's t-test."""
    out = {}
    for cell_id, seed_data in all_data.items():
        out[cell_id] = {}
        for seed in SEEDS:
            seed_str = str(seed)
            if seed_str not in seed_data:
                continue
            total_succ = 0
            total_ep = 0
            for task_id in TASK_IDS:
                outcomes = seed_data[seed_str].get("per_task_outcomes", {}).get(str(task_id), [])
                total_succ += sum(outcomes)
                total_ep += len(outcomes)
            if total_ep > 0:
                out[cell_id][seed] = total_succ / total_ep
    return out


def pairwise_mcnemar(per_cell_per_task: Dict, cell_pairs: List[Tuple[str, str]]) -> List[Dict]:
    """Run McNemar test for each (cell_pair × task)."""
    results = []
    for cell_a, cell_b in cell_pairs:
        for task_id in TASK_IDS:
            if (task_id not in per_cell_per_task.get(cell_a, {})
                or task_id not in per_cell_per_task.get(cell_b, {})):
                continue
            outc_a = per_cell_per_task[cell_a][task_id]["outcomes"]
            outc_b = per_cell_per_task[cell_b][task_id]["outcomes"]
            # Need paired: if same n, treat as paired (deterministic seed alignment)
            n = min(len(outc_a), len(outc_b))
            chi2_stat, p, n_01, n_10 = mcnemar_pvalue(outc_a[:n], outc_b[:n])
            results.append({
                "cell_a": cell_a, "cell_b": cell_b, "task_id": task_id,
                "chi2_stat": chi2_stat, "p_value": p,
                "n_01": n_01, "n_10": n_10,
                "sr_a": per_cell_per_task[cell_a][task_id]["sr"],
                "sr_b": per_cell_per_task[cell_b][task_id]["sr"],
            })
    return results


def latex_table_v_d(per_cell_per_task: Dict, per_cell_per_seed: Dict, cells: Dict) -> str:
    """Generate §V.D LaTeX table."""
    lines = [
        "\\begin{table}[t]",
        "\\centering",
        "\\caption{\\textsc{\\S V.D Ablation Results.} "
        "Mean SR ($\\pm$ std across 3 seeds) per cell × task on LIBERO-Spatial. "
        "n=150 episodes per cell × task (3 seeds $\\times$ 50 episodes). "
        "$^\\dagger$ q$<$0.05 vs V3 baseline (BH-FDR adjusted McNemar).}",
        "\\label{tab:v_d_ablation}",
        "\\small",
        "\\begin{tabular}{l|" + "c" * len(TASK_IDS) + "|c}",
        "\\toprule",
        "Cell & " + " & ".join(f"T{t}" for t in TASK_IDS) + " & Mean \\\\",
        "\\midrule",
    ]
    for cell_id, cell_spec in cells.items():
        if cell_id not in per_cell_per_task:
            continue
        task_srs = []
        for task_id in TASK_IDS:
            if task_id in per_cell_per_task[cell_id]:
                task_srs.append(per_cell_per_task[cell_id][task_id]["sr"])
            else:
                task_srs.append(None)
        # Per-seed overall SR for std
        seed_srs = [per_cell_per_seed.get(cell_id, {}).get(s) for s in SEEDS]
        seed_srs = [s for s in seed_srs if s is not None]
        mean_sr = np.mean(seed_srs) if seed_srs else float("nan")
        std_sr = np.std(seed_srs, ddof=1) if len(seed_srs) > 1 else 0.0

        row = [cell_id.replace("_", "\\_")]
        for sr in task_srs:
            row.append(f"{sr*100:.0f}" if sr is not None else "-")
        row.append(f"{mean_sr*100:.1f}$\\pm${std_sr*100:.1f}")
        lines.append(" & ".join(row) + " \\\\")
    lines.extend([
        "\\bottomrule",
        "\\end{tabular}",
        "\\end{table}",
    ])
    return "\n".join(lines)


def phase_b_to_f_analyze(sr_data_path: Path, output_dir: Path, cells: Dict):
    """Run all statistical analyses on collected SR data."""
    output_dir.mkdir(parents=True, exist_ok=True)
    all_data = json.loads(Path(sr_data_path).read_text())

    # Aggregate
    per_cell_per_task = aggregate_per_cell_per_task(all_data)
    per_cell_per_seed = aggregate_per_cell_per_seed(all_data)

    # Define cell pairs (per preregistration §2.2)
    cell_ids = list(cells.keys())
    architecture_pairs = []
    drop_vs_v3_pairs = []
    for i, ca in enumerate(cell_ids):
        for cb in cell_ids[i+1:]:
            if ca.startswith(("E_drop", "F_drop", "G_drop")) or cb.startswith(("E_drop", "F_drop", "G_drop")):
                # Only compare drop cells vs D_v3
                if ca == "D_v3" and cb.startswith(("E_drop", "F_drop", "G_drop")):
                    drop_vs_v3_pairs.append((ca, cb))
                elif cb == "D_v3" and ca.startswith(("E_drop", "F_drop", "G_drop")):
                    drop_vs_v3_pairs.append((cb, ca))
            else:
                architecture_pairs.append((ca, cb))

    logger.info(f"Architecture pairs: {len(architecture_pairs)}")
    logger.info(f"Drop-vs-V3 pairs: {len(drop_vs_v3_pairs)}")

    # Phase B + C: Pairwise McNemar with BH-FDR (separate families per preregistration)
    arch_results = pairwise_mcnemar(per_cell_per_task, architecture_pairs)
    drop_results = pairwise_mcnemar(per_cell_per_task, drop_vs_v3_pairs)

    arch_pvals = [r["p_value"] for r in arch_results]
    drop_pvals = [r["p_value"] for r in drop_results]
    arch_qvals, arch_reject = benjamini_hochberg(arch_pvals, ALPHA)
    drop_qvals, drop_reject = benjamini_hochberg(drop_pvals, ALPHA)
    for r, q, rej in zip(arch_results, arch_qvals, arch_reject):
        r["q_value"] = q
        r["bh_reject"] = bool(rej)
    for r, q, rej in zip(drop_results, drop_qvals, drop_reject):
        r["q_value"] = q
        r["bh_reject"] = bool(rej)

    # Phase D: Wilson CI per cell × task
    cell_task_cis = {}
    for cell_id, task_data in per_cell_per_task.items():
        cell_task_cis[cell_id] = {}
        for task_id, td in task_data.items():
            lo, hi = wilson_ci(td["n_success"], td["n_total"])
            cell_task_cis[cell_id][str(task_id)] = {"lo": lo, "hi": hi}

    # Phase E: Welch's t-test on per-seed SR (cell pair level)
    welch_results = []
    for ca, cb in architecture_pairs + drop_vs_v3_pairs:
        sa = list(per_cell_per_seed.get(ca, {}).values())
        sb = list(per_cell_per_seed.get(cb, {}).values())
        if len(sa) >= 2 and len(sb) >= 2:
            t, p = welchs_ttest(sa, sb)
            welch_results.append({
                "cell_a": ca, "cell_b": cb,
                "t_stat": t, "p_value": p,
                "mean_a": float(np.mean(sa)), "mean_b": float(np.mean(sb)),
                "n_seeds_a": len(sa), "n_seeds_b": len(sb),
            })

    # Phase F: LaTeX table
    latex = latex_table_v_d(per_cell_per_task, per_cell_per_seed, cells)
    (output_dir / "v_d_table.tex").write_text(latex)

    # JSON detail report
    report = {
        "timestamp": time.time(),
        "n_cells": len(cells),
        "cells": cells,
        "per_cell_per_task_sr": {
            ci: {str(ti): {"sr": td["sr"], "n_success": td["n_success"],
                            "n_total": td["n_total"]}
                  for ti, td in tdata.items()}
            for ci, tdata in per_cell_per_task.items()
        },
        "per_cell_per_seed_sr": per_cell_per_seed,
        "wilson_cis": cell_task_cis,
        "architecture_pairs_mcnemar": arch_results,
        "drop_vs_v3_pairs_mcnemar": drop_results,
        "welch_ttest_seed_level": welch_results,
    }
    out_json = output_dir / "v_d_analysis.json"
    out_json.write_text(json.dumps(report, indent=2, default=str))

    # Console summary
    print("\n" + "=" * 72)
    print("§V.D ABLATION ANALYSIS SUMMARY")
    print("=" * 72)
    for cell_id in cells:
        if cell_id in per_cell_per_task:
            cell_total_succ = sum(td["n_success"] for td in per_cell_per_task[cell_id].values())
            cell_total_ep = sum(td["n_total"] for td in per_cell_per_task[cell_id].values())
            overall_sr = cell_total_succ / max(cell_total_ep, 1)
            print(f"  {cell_id}: overall SR = {overall_sr:.1%} "
                  f"({cell_total_succ}/{cell_total_ep})")
    print()
    print(f"Architecture comparisons: {sum(arch_reject)}/{len(arch_reject)} "
          f"reject H0 at BH-FDR q<{ALPHA}")
    print(f"Drop-vs-V3 comparisons:   {sum(drop_reject)}/{len(drop_reject)} "
          f"reject H0 at BH-FDR q<{ALPHA}")
    print(f"\nLaTeX table: {output_dir / 'v_d_table.tex'}")
    print(f"Full report: {out_json}")
    print("=" * 72 + "\n")


# =============================================================================
# Main
# =============================================================================

def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--phase", choices=["collect", "analyze", "both"], default="both",
                        help="collect: run SR rollouts. analyze: stats only.")
    parser.add_argument("--results-root", type=Path,
                        default=REPO_ROOT / "results",
                        help="Root containing ablation_*_seed{seed}_<timestamp>/ dirs")
    parser.add_argument("--output-dir", type=Path,
                        default=REPO_ROOT / "results" / "phase0" / "v_d")
    parser.add_argument("--sr-data", type=Path,
                        help="JSON file with collected SR (for --phase analyze)")
    parser.add_argument("--n-episodes-per-task", type=int, default=50)
    parser.add_argument("--cells-config", choices=["full", "no_variantc"], default="full",
                        help="Whether variant (c) is included (21 vs 18 retrains)")
    args = parser.parse_args(argv)

    cells = CELLS_FULL if args.cells_config == "full" else CELLS_NO_VARIANTC
    args.output_dir.mkdir(parents=True, exist_ok=True)
    logger.info(f"Using {len(cells)}-cell configuration: {args.cells_config}")

    if args.phase in ("collect", "both"):
        all_data = phase_a_collect(
            args.results_root, args.output_dir,
            args.n_episodes_per_task, cells,
        )
        sr_data_path = args.output_dir / "all_sr_data.json"
    else:
        sr_data_path = args.sr_data or (args.output_dir / "all_sr_data.json")
        if not sr_data_path.exists():
            logger.error(f"SR data not found: {sr_data_path}")
            return 1

    if args.phase in ("analyze", "both"):
        phase_b_to_f_analyze(sr_data_path, args.output_dir, cells)

    return 0


if __name__ == "__main__":
    sys.exit(main())
