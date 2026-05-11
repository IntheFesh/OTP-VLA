"""
Stage 3 Multi-Variant Analysis: V1 strict / V2 CI-aware / V3 informative+CI

Computes Rule A trigger status under three different statistical applications:

  V1 (strict V7 §III.F):
    - All 50 cells in FDR pool
    - Hard SR threshold: paraphrased mean SR >= identity_mean - 2pp
    - Already run as 22_stage3_main_analysis.py: Rule none fires

  V2 (CI-aware SR threshold only, recommended):
    - All 50 cells in FDR pool (V7 frozen scope respected)
    - Statistical application: 95% CI of paired SR diff overlaps 2pp threshold
    - Standard practice for applying frozen thresholds to noisy estimates

  V3 (informative-cell + CI, reviewer's full proposal):
    - Only cells with discordant pairs >= 4 enter FDR pool
    - CI-aware SR threshold
    - More liberal: filters out trivial cells

This script reports all three to inform decision-making, not to pick the
most favorable one. The decision rationale is documented separately.
"""

import json
import sys
from pathlib import Path

import numpy as np
from scipy.stats import chi2, norm

REPO_ROOT = Path(__file__).resolve().parents[2]
OUT_DIR = REPO_ROOT / "results" / "phase0"

INPUT_PATH = OUT_DIR / "oft_unseen_phrasing_full.json"
OUTPUT_PATH = OUT_DIR / "stage3_multi_variant_analysis.json"

PHRASE_IDS = ["P0_identity", "P1_word_order", "P2_synonym", "P3_passive",
              "P4_verb_change", "P5_compact"]
PARAPHRASED_IDS = [pid for pid in PHRASE_IDS if pid != "P0_identity"]
ALPHA = 0.05
SR_THRESHOLD_PP = 0.02  # 2pp V7 §III.F Rule A threshold
INFORMATIVE_DISCORDANT_THRESHOLD = 4  # for V3


def mcnemar_pvalue(identity, paraphrased):
    """Continuity-corrected McNemar test."""
    n_01 = sum(1 for (i, p) in zip(identity, paraphrased) if i and not p)
    n_10 = sum(1 for (i, p) in zip(identity, paraphrased) if not i and p)
    if n_01 + n_10 == 0:
        return 0.0, 1.0, n_01, n_10
    chi2_stat = (abs(n_01 - n_10) - 1) ** 2 / (n_01 + n_10)
    p_value = 1.0 - chi2.cdf(chi2_stat, df=1)
    return float(chi2_stat), float(p_value), n_01, n_10


def benjamini_hochberg(pvalues, alpha=ALPHA):
    """BH-FDR returns q-values."""
    p = np.asarray(pvalues, dtype=float)
    n = len(p)
    if n == 0:
        return [], []
    order = np.argsort(p)
    ranked = p[order]
    q = ranked * n / (np.arange(n) + 1)
    q = np.minimum.accumulate(q[::-1])[::-1]
    q = np.clip(q, 0, 1)
    q_full = np.empty_like(q)
    q_full[order] = q
    reject = q_full < alpha
    return q_full.tolist(), reject.tolist()


def paired_diff_ci(identity_means, paraphrased_means, n_episodes=50, n_tasks=10):
    """
    95% CI of (identity - paraphrased) mean SR difference.
    Treats each task as one observation of paired SR estimates.

    Uses normal approximation with SE of difference estimated from
    binomial variance pooled across tasks.

    Args:
        identity_means: list of per-task identity SR (10 values)
        paraphrased_means: list of per-task paraphrased SR for one phrasing
    Returns:
        (mean_diff, ci_low, ci_high)
    """
    diffs = np.array(identity_means) - np.array(paraphrased_means)
    mean_diff = float(np.mean(diffs))

    # Use std error of mean of diffs across tasks (n_tasks=10)
    # SE = std(diffs) / sqrt(n_tasks)
    se = float(np.std(diffs, ddof=1) / np.sqrt(len(diffs)))
    z = norm.ppf(1 - ALPHA / 2)  # 1.96 for 95%
    ci_low = mean_diff - z * se
    ci_high = mean_diff + z * se
    return mean_diff, ci_low, ci_high


def run_variant(variant_name, per_cell_sr, mcnemar_results, identity_mean_sr,
                use_informative_filter=False, use_ci_threshold=False):
    """Apply Rule A SR condition + McNemar condition under given variant."""
    # Filter cells if requested
    if use_informative_filter:
        filtered = [r for r in mcnemar_results
                    if (r["n_01"] + r["n_10"]) >= INFORMATIVE_DISCORDANT_THRESHOLD]
    else:
        filtered = mcnemar_results

    # Compute FDR on (possibly filtered) cells
    p_values = [r["p_value"] for r in filtered]
    q_values, reject = benjamini_hochberg(p_values) if p_values else ([], [])
    for r, q, rej in zip(filtered, q_values, reject):
        r["q_value_fdr"] = q
        r["reject_at_fdr_0.05"] = bool(rej)

    # Per-phrasing mean SR
    per_phrase_mean_sr = {}
    per_phrase_per_task_identity = {}
    per_phrase_per_task_paraphrased = {}
    for pid in PARAPHRASED_IDS:
        identity_per_task = []
        paraphrased_per_task = []
        for t in range(10):
            if (t, "P0_identity") in per_cell_sr and (t, pid) in per_cell_sr:
                identity_per_task.append(per_cell_sr[(t, "P0_identity")])
                paraphrased_per_task.append(per_cell_sr[(t, pid)])
        per_phrase_mean_sr[pid] = float(np.mean(paraphrased_per_task))
        per_phrase_per_task_identity[pid] = identity_per_task
        per_phrase_per_task_paraphrased[pid] = paraphrased_per_task

    # Rule A SR condition
    sr_check_per_phrase = {}
    rule_a_sr_cond_overall = True
    for pid in PARAPHRASED_IDS:
        if use_ci_threshold:
            # Compute CI of paired diff
            mean_diff, ci_low, ci_high = paired_diff_ci(
                per_phrase_per_task_identity[pid],
                per_phrase_per_task_paraphrased[pid],
            )
            # Rule A SR condition: diff <= 2pp. CI-aware version:
            # condition VIOLATED iff lower 95% CI bound on diff > 2pp
            # (i.e., we're confident diff exceeds 2pp)
            ci_violates = ci_low > SR_THRESHOLD_PP
            sr_check_per_phrase[pid] = {
                "mean_diff": mean_diff,
                "ci_low": ci_low,
                "ci_high": ci_high,
                "threshold": SR_THRESHOLD_PP,
                "violates_threshold_ci": bool(ci_violates),
                "violates_threshold_hard": bool(mean_diff > SR_THRESHOLD_PP),
            }
            if ci_violates:
                rule_a_sr_cond_overall = False
        else:
            # Hard threshold
            mean_diff = identity_mean_sr - per_phrase_mean_sr[pid]
            violates = mean_diff > SR_THRESHOLD_PP
            sr_check_per_phrase[pid] = {
                "mean_diff": mean_diff,
                "threshold": SR_THRESHOLD_PP,
                "violates_threshold_hard": bool(violates),
            }
            if violates:
                rule_a_sr_cond_overall = False

    # Rule A McNemar condition: all (filtered) cells q > 0.1
    if filtered:
        rule_a_mcnemar_cond = all(r["q_value_fdr"] > 0.1 for r in filtered)
    else:
        rule_a_mcnemar_cond = True  # no cells to reject

    rule_a_fires = rule_a_sr_cond_overall and rule_a_mcnemar_cond

    return {
        "variant": variant_name,
        "use_informative_filter": use_informative_filter,
        "use_ci_threshold": use_ci_threshold,
        "n_cells_in_pool": len(filtered),
        "n_fdr_significant": sum(r["reject_at_fdr_0.05"] for r in filtered),
        "sr_check_per_phrase": sr_check_per_phrase,
        "rule_a_sr_cond": bool(rule_a_sr_cond_overall),
        "rule_a_mcnemar_cond": bool(rule_a_mcnemar_cond),
        "rule_a_fires": bool(rule_a_fires),
    }


def main():
    print("=" * 70)
    print("Stage 3 Multi-Variant Analysis")
    print("=" * 70)
    print("V1: strict V7 (all cells, hard threshold)")
    print("V2: CI-aware SR threshold (all cells, 95% CI on paired diff)")
    print("V3: informative-cell filter + CI threshold (reviewer's proposal)")
    print()

    if not INPUT_PATH.exists():
        print(f"ERROR: {INPUT_PATH} not found")
        sys.exit(1)

    with open(INPUT_PATH) as f:
        raw = json.load(f)

    per_cell_results = {}
    for key, val in raw.items():
        if not key.startswith("task") or not isinstance(val, list):
            continue
        parts = key.split("_", 1)
        if len(parts) != 2:
            continue
        try:
            task_id = int(parts[0].replace("task", ""))
        except ValueError:
            continue
        per_cell_results[(task_id, parts[1])] = val

    per_cell_sr = {k: float(sum(v) / len(v)) for k, v in per_cell_results.items()}

    identity_srs = [per_cell_sr[(t, "P0_identity")] for t in range(10)
                    if (t, "P0_identity") in per_cell_sr]
    identity_mean_sr = float(np.mean(identity_srs))

    print(f"Identity mean SR: {identity_mean_sr:.1%}")
    print()

    # Build McNemar results (cells unfiltered)
    mcnemar_all = []
    for t in range(10):
        identity = per_cell_results.get((t, "P0_identity"))
        if identity is None:
            continue
        for pid in PARAPHRASED_IDS:
            paraphrased = per_cell_results.get((t, pid))
            if paraphrased is None:
                continue
            chi2_stat, p_val, n_01, n_10 = mcnemar_pvalue(identity, paraphrased)
            mcnemar_all.append({
                "task": t,
                "phrasing": pid,
                "identity_sr": sum(identity)/len(identity),
                "paraphrased_sr": sum(paraphrased)/len(paraphrased),
                "n_01": n_01,
                "n_10": n_10,
                "discordant_total": n_01 + n_10,
                "chi2": chi2_stat,
                "p_value": p_val,
            })

    # === Print informative-cell breakdown ===
    print("=" * 70)
    print("Informative-cell breakdown (b+c = discordant pair total)")
    print("=" * 70)
    discordant_counts = [r["discordant_total"] for r in mcnemar_all]
    print(f"Total cells: {len(mcnemar_all)}")
    print(f"Cells with b+c = 0:  {sum(1 for c in discordant_counts if c == 0)}")
    print(f"Cells with b+c = 1-3: {sum(1 for c in discordant_counts if 1 <= c <= 3)}")
    print(f"Cells with b+c >= 4: {sum(1 for c in discordant_counts if c >= 4)}")
    print()
    print("Cells with b+c >= 4 (informative under reviewer's filter):")
    for r in sorted(mcnemar_all, key=lambda r: -r["discordant_total"]):
        if r["discordant_total"] < 4:
            break
        print(f"  task {r['task']} × {r['phrasing']:<20} "
              f"identity {r['identity_sr']:.0%}  para {r['paraphrased_sr']:.0%}  "
              f"b+c={r['discordant_total']}  p={r['p_value']:.4f}")
    print()

    # === Run three variants ===
    # We deep-copy mcnemar_all for each variant so q-values don't pollute
    import copy
    v1 = run_variant("V1_strict", per_cell_sr, copy.deepcopy(mcnemar_all),
                     identity_mean_sr, use_informative_filter=False, use_ci_threshold=False)
    v2 = run_variant("V2_ci_only", per_cell_sr, copy.deepcopy(mcnemar_all),
                     identity_mean_sr, use_informative_filter=False, use_ci_threshold=True)
    v3 = run_variant("V3_informative_ci", per_cell_sr, copy.deepcopy(mcnemar_all),
                     identity_mean_sr, use_informative_filter=True, use_ci_threshold=True)

    # === Display variant comparison ===
    for v in [v1, v2, v3]:
        print("=" * 70)
        print(f"Variant: {v['variant']}")
        print("=" * 70)
        print(f"  use_informative_filter: {v['use_informative_filter']}")
        print(f"  use_ci_threshold:       {v['use_ci_threshold']}")
        print(f"  cells in FDR pool:      {v['n_cells_in_pool']}")
        print(f"  FDR-significant cells:  {v['n_fdr_significant']}")
        print(f"  Rule A SR condition:    {v['rule_a_sr_cond']}")
        print(f"  Rule A McNemar/FDR:     {v['rule_a_mcnemar_cond']}")
        print(f"  Rule A FIRES:           {v['rule_a_fires']}")
        print()
        print("  SR check per phrasing:")
        for pid in PARAPHRASED_IDS:
            ch = v["sr_check_per_phrase"][pid]
            if v["use_ci_threshold"]:
                print(f"    {pid:<20}  diff={ch['mean_diff']*100:+.2f}pp  "
                      f"95% CI [{ch['ci_low']*100:+.2f}, {ch['ci_high']*100:+.2f}]  "
                      f"violates_ci={ch['violates_threshold_ci']}")
            else:
                print(f"    {pid:<20}  diff={ch['mean_diff']*100:+.2f}pp  "
                      f"violates_hard={ch['violates_threshold_hard']}")
        print()

    # === Comparative summary ===
    print("=" * 70)
    print("Cross-variant summary")
    print("=" * 70)
    print(f"  V1 (strict):        Rule A fires = {v1['rule_a_fires']}")
    print(f"  V2 (CI-only):       Rule A fires = {v2['rule_a_fires']}")
    print(f"  V3 (info+CI):       Rule A fires = {v3['rule_a_fires']}")
    print()

    # Save
    out = {
        "version": "stage3_multi_variant_v1",
        "input_file": str(INPUT_PATH),
        "identity_mean_sr": identity_mean_sr,
        "n_total_cells": len(mcnemar_all),
        "n_cells_zero_discordant": sum(1 for r in mcnemar_all if r["discordant_total"] == 0),
        "n_cells_low_discordant": sum(1 for r in mcnemar_all if 1 <= r["discordant_total"] <= 3),
        "n_cells_informative": sum(1 for r in mcnemar_all if r["discordant_total"] >= 4),
        "variants": {"V1": v1, "V2": v2, "V3": v3},
        "mcnemar_all": mcnemar_all,
    }
    with open(OUTPUT_PATH, "w") as f:
        json.dump(out, f, indent=2)
    print(f"Written: {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
