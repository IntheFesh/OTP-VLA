"""
Stage 3 Post-hoc Analysis: McNemar test + V7 §III.F decision rules.

Loads results/phase0/oft_unseen_phrasing_full.json (Stage 3 output) and
applies V7 §III.E McNemar test + §III.F conditional framing rules.

Outputs:
  - Per-(task, phrasing) cell SR table
  - Per-(task, phrasing) McNemar p-values vs identity
  - V7 §III.F rule application (A / B / C / D)
  - results/phase0/stage3_analysis.json with all numbers
  - Console: human-readable verdict + paper §V.B framing implication

Usage:
  python scripts/diagnostic/21_stage3_mcnemar_analysis.py
"""

import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
from scipy.stats import chi2

REPO_ROOT = Path(__file__).resolve().parents[2]
OUT_DIR = REPO_ROOT / "results" / "phase0"

INPUT_PATH = OUT_DIR / "oft_unseen_phrasing_full.json"
OUTPUT_PATH = OUT_DIR / "stage3_analysis.json"

PHRASE_IDS = ["P0_identity", "P1_word_order", "P2_synonym", "P3_passive",
              "P4_verb_change", "P5_compact"]
PARAPHRASED_IDS = [pid for pid in PHRASE_IDS if pid != "P0_identity"]


def mcnemar_pvalue(identity_outcomes, paraphrased_outcomes):
    """Compute McNemar test p-value on paired binary outcomes.

    Args:
        identity_outcomes:    list of bool (50 episodes under P0_identity)
        paraphrased_outcomes: list of bool (50 episodes under paraphrasing)

    Returns:
        (chi2_stat, p_value, n_01, n_10)
        n_01: succeed in identity but fail in paraphrased
        n_10: fail in identity but succeed in paraphrased
    """
    assert len(identity_outcomes) == len(paraphrased_outcomes), \
        "Paired McNemar requires equal-length sequences"

    n_01 = sum(1 for (i, p) in zip(identity_outcomes, paraphrased_outcomes) if i and not p)
    n_10 = sum(1 for (i, p) in zip(identity_outcomes, paraphrased_outcomes) if not i and p)

    if n_01 + n_10 == 0:
        # Identical outcomes → no evidence of difference
        return 0.0, 1.0, n_01, n_10

    # Continuity correction
    chi2_stat = (abs(n_01 - n_10) - 1) ** 2 / (n_01 + n_10)
    p_value = 1.0 - chi2.cdf(chi2_stat, df=1)
    return float(chi2_stat), float(p_value), n_01, n_10


def apply_v7_rules(per_cell_sr, mcnemar_results, identity_overall_sr):
    """Apply V7 §III.F conditional rules A/B/C/D.

    Rule A (visual-grounded):
        Paraphrased SR ≥ identity − 2pp for all 5 paraphrasings (mean SR)
        AND McNemar p > 0.1 for all 50 (task × phrasing) cells

    Rule B (language-conditioned, brittle):
        Mean paraphrased SR ≤ identity − 15pp
        AND McNemar p < 0.01 in ≥ 3 paraphrased phrasings (averaged across tasks)

    Rule C (mixed, per-task heterogeneous):
        Per-task SR drop variance > 20pp² across 10 tasks
        AND neither A nor B applies

    Rule D (thesis falsification):
        For ≥ 8 of 10 tasks, mean paraphrased SR drop ≥ 50pp on all 5 paraphrasings
        AND OFT identity SR ≥ 95%

    Returns dict with each rule's status and triggered rule (A/B/C/D/none).
    """
    # Compute mean paraphrased SR per phrasing (averaged across 10 tasks)
    per_phrase_mean_sr = {}
    for pid in PARAPHRASED_IDS:
        srs = [per_cell_sr[(t, pid)] for t in range(10) if (t, pid) in per_cell_sr]
        per_phrase_mean_sr[pid] = float(np.mean(srs)) if srs else None

    # Per-task SR drops (mean across paraphrased phrasings)
    per_task_mean_drop = {}
    per_task_drops_all_phrases = {}  # for Rule D
    for t in range(10):
        identity_sr = per_cell_sr.get((t, "P0_identity"))
        if identity_sr is None:
            continue
        drops = []
        for pid in PARAPHRASED_IDS:
            p_sr = per_cell_sr.get((t, pid))
            if p_sr is not None:
                drops.append(identity_sr - p_sr)
        per_task_mean_drop[t] = float(np.mean(drops)) if drops else None
        per_task_drops_all_phrases[t] = drops

    # --- Rule A ---
    # All 5 paraphrasings: mean SR ≥ identity − 2pp
    identity_mean_sr = np.mean([per_cell_sr.get((t, "P0_identity"), 0) for t in range(10)])
    rule_a_sr_cond = all(
        (per_phrase_mean_sr[pid] or 0) >= identity_mean_sr - 0.02
        for pid in PARAPHRASED_IDS
    )
    # McNemar: p > 0.1 for ALL 50 cells
    rule_a_mcnemar_cond = all(r["p_value"] > 0.1 for r in mcnemar_results)
    rule_a_fires = rule_a_sr_cond and rule_a_mcnemar_cond

    # --- Rule B ---
    # Mean paraphrased SR ≤ identity − 15pp
    mean_paraphrased_sr = float(np.mean(list(per_phrase_mean_sr.values())))
    rule_b_sr_cond = mean_paraphrased_sr <= identity_mean_sr - 0.15
    # McNemar p < 0.01 in ≥ 3 paraphrased phrasings
    phrase_sig_count = 0
    for pid in PARAPHRASED_IDS:
        ps = [r["p_value"] for r in mcnemar_results if r["phrasing"] == pid]
        if ps and np.mean(ps) < 0.01:
            phrase_sig_count += 1
    rule_b_mcnemar_cond = phrase_sig_count >= 3
    rule_b_fires = rule_b_sr_cond and rule_b_mcnemar_cond

    # --- Rule C ---
    drops = list(per_task_mean_drop.values())
    drop_var_pp2 = float(np.var(drops) * 10000) if drops else 0.0  # to pp²
    rule_c_var_cond = drop_var_pp2 > 20.0
    rule_c_fires = rule_c_var_cond and not rule_a_fires and not rule_b_fires

    # --- Rule D ---
    # ≥ 8 tasks have mean drop ≥ 50pp on all 5 paraphrasings
    n_tasks_severe = 0
    for t in range(10):
        drops_t = per_task_drops_all_phrases.get(t, [])
        if drops_t and all(d >= 0.50 for d in drops_t):
            n_tasks_severe += 1
    rule_d_severe_cond = n_tasks_severe >= 8
    rule_d_identity_cond = identity_mean_sr >= 0.95
    rule_d_fires = rule_d_severe_cond and rule_d_identity_cond

    # Determine triggered rule (priority: D > B > A > C as written; mutual exclusion)
    if rule_d_fires:
        triggered = "D"
    elif rule_b_fires:
        triggered = "B"
    elif rule_a_fires:
        triggered = "A"
    elif rule_c_fires:
        triggered = "C"
    else:
        triggered = "none"

    return {
        "identity_mean_sr": float(identity_mean_sr),
        "per_phrase_mean_sr": per_phrase_mean_sr,
        "mean_paraphrased_sr": mean_paraphrased_sr,
        "per_task_mean_drop": per_task_mean_drop,
        "rule_a_sr_cond": bool(rule_a_sr_cond),
        "rule_a_mcnemar_cond": bool(rule_a_mcnemar_cond),
        "rule_a_fires": bool(rule_a_fires),
        "rule_b_sr_cond": bool(rule_b_sr_cond),
        "rule_b_mcnemar_phrase_count": int(phrase_sig_count),
        "rule_b_mcnemar_cond": bool(rule_b_mcnemar_cond),
        "rule_b_fires": bool(rule_b_fires),
        "rule_c_drop_var_pp2": drop_var_pp2,
        "rule_c_var_cond": bool(rule_c_var_cond),
        "rule_c_fires": bool(rule_c_fires),
        "rule_d_severe_tasks": int(n_tasks_severe),
        "rule_d_severe_cond": bool(rule_d_severe_cond),
        "rule_d_identity_cond": bool(rule_d_identity_cond),
        "rule_d_fires": bool(rule_d_fires),
        "triggered_rule": triggered,
    }


def framing_implication(rule_triggered):
    framings = {
        "A": ("Paper §V.B sells the THREE-LAYER DIAGNOSTIC + Theorem 3 story:\n"
              "  - Backbone faithful (Gate 1: r=0.76)\n"
              "  - Demonstrations language-orthogonal (M3: r∈[-0.27, -0.20])\n"
              "  - V3 policy unfaithful (C1: r=-0.19)\n"
              "  - OFT is robust to paraphrase (this Rule A data)\n"
              "  - Theorem 3 explains supervision-collapse\n"
              "  Sell point: STRONGEST. Original M3 thesis fully supported."),
        "B": ("Paper §V.B narrows scope to ARCHITECTURAL COMPARISON:\n"
              "  - OFT brittleness under paraphrase suggests OFT recovers\n"
              "    language conditioning through inductive bias\n"
              "  - OTP-Soft's stacked CFM head fails to do this\n"
              "  Sell point: NARROWER. Architectural comparison framing."),
        "C": ("Paper §V.B per-task NUANCED framing:\n"
              "  - Some tasks visual-grounded, others language-conditioned\n"
              "  - Mechanism is task-conditional (likely scene-query)\n"
              "  - Per-task breakdown table is the central exhibit\n"
              "  Sell point: NUANCED but defensible."),
        "D": ("Paper main thesis FAILS. Fallback to Tier 3:\n"
              "  - OFT critically depends on language tokens\n"
              "  - This contradicts Theorem 3's M3-based prediction\n"
              "  - Reframe: 'OFT and OTP-Soft differ architecturally;\n"
              "    mechanism remains open'\n"
              "  Sell point: WEAKEST. Theorem 3 either has hidden assumption\n"
              "  violation or inductive bias provides M3-circumventing path."),
        "none": ("No conditional rule fires — outcome doesn't fit V7 §III.F.\n"
                 "Manual analysis required. Likely data quality issue or\n"
                 "unexpected outcome combination."),
    }
    return framings.get(rule_triggered, framings["none"])


def main():
    print("=" * 70)
    print("Stage 3 Post-hoc Analysis (V7 §III.E + §III.F)")
    print("=" * 70)

    if not INPUT_PATH.exists():
        print(f"\nERROR: Stage 3 results not found at {INPUT_PATH}")
        print("Stage 3 must complete first.")
        sys.exit(1)

    with open(INPUT_PATH) as f:
        raw = json.load(f)

    # Parse cell data: keys are "task{T}_{phrasing_id}"
    per_cell_results = {}  # (task_id, phrasing_id) -> list of bool
    for key, val in raw.items():
        if not key.startswith("task"):
            continue
        if not isinstance(val, list):
            continue
        # e.g., "task3_P1_word_order" → ("task3", "P1_word_order")
        parts = key.split("_", 1)
        if len(parts) != 2:
            continue
        task_part = parts[0]
        phrase_id = parts[1]
        try:
            task_id = int(task_part.replace("task", ""))
        except ValueError:
            continue
        per_cell_results[(task_id, phrase_id)] = val

    # Per-cell SR
    per_cell_sr = {k: float(sum(v) / len(v)) for k, v in per_cell_results.items()}

    print(f"\nCells loaded: {len(per_cell_sr)} of expected 60 (10 task × 6 phrasing)")
    if "global_sr" in raw:
        print(f"Global SR: {raw['global_sr']:.1%}")

    # === Per-cell SR table ===
    print("\n" + "=" * 70)
    print("Per-cell SR table (rows: tasks, columns: phrasings)")
    print("=" * 70)
    print(f"{'task':<6}", end="")
    for pid in PHRASE_IDS:
        print(f"{pid[:15]:<16}", end="")
    print()
    for t in range(10):
        print(f"{t:<6}", end="")
        for pid in PHRASE_IDS:
            sr = per_cell_sr.get((t, pid))
            print(f"{sr:.0%}{'':>11}" if sr is not None else f"{'N/A':<16}", end="")
        print()

    # === Per-cell McNemar tests ===
    print("\n" + "=" * 70)
    print("McNemar tests (per task × paraphrasing vs identity)")
    print("=" * 70)
    mcnemar_results = []
    print(f"{'task':<6}{'phrasing':<20}{'identity':<12}{'paraphrased':<14}{'n_01':<6}{'n_10':<6}{'chi2':<10}{'p_value':<10}")
    for t in range(10):
        identity = per_cell_results.get((t, "P0_identity"))
        if identity is None:
            continue
        for pid in PARAPHRASED_IDS:
            paraphrased = per_cell_results.get((t, pid))
            if paraphrased is None:
                continue
            chi2_stat, p_val, n_01, n_10 = mcnemar_pvalue(identity, paraphrased)
            mcnemar_results.append({
                "task": t,
                "phrasing": pid,
                "identity_sr": sum(identity)/len(identity),
                "paraphrased_sr": sum(paraphrased)/len(paraphrased),
                "n_01": n_01,
                "n_10": n_10,
                "chi2": chi2_stat,
                "p_value": p_val,
            })
            print(f"{t:<6}{pid:<20}{sum(identity)/len(identity):<12.0%}"
                  f"{sum(paraphrased)/len(paraphrased):<14.0%}{n_01:<6}{n_10:<6}"
                  f"{chi2_stat:<10.3f}{p_val:<10.4f}")

    # === V7 §III.F rule application ===
    print("\n" + "=" * 70)
    print("V7 §III.F Rule Application")
    print("=" * 70)
    rule_results = apply_v7_rules(per_cell_sr, mcnemar_results, None)

    print(f"\nIdentity mean SR (across 10 tasks): {rule_results['identity_mean_sr']:.1%}")
    print(f"Mean paraphrased SR:                 {rule_results['mean_paraphrased_sr']:.1%}")
    print(f"\nPer-phrasing mean SR:")
    for pid in PARAPHRASED_IDS:
        sr = rule_results["per_phrase_mean_sr"].get(pid)
        if sr is not None:
            print(f"  {pid:<20}  {sr:.1%}")

    print(f"\nRule A (visual-grounded):")
    print(f"  paraphrased SR ≥ identity − 2pp:  {rule_results['rule_a_sr_cond']}")
    print(f"  McNemar p > 0.1 in all cells:     {rule_results['rule_a_mcnemar_cond']}")
    print(f"  → Rule A fires: {rule_results['rule_a_fires']}")

    print(f"\nRule B (language-conditioned brittle):")
    print(f"  paraphrased ≤ identity − 15pp:    {rule_results['rule_b_sr_cond']}")
    print(f"  ≥ 3 phrasings with mean p < 0.01: {rule_results['rule_b_mcnemar_cond']} "
          f"(actual {rule_results['rule_b_mcnemar_phrase_count']})")
    print(f"  → Rule B fires: {rule_results['rule_b_fires']}")

    print(f"\nRule C (mixed heterogeneous):")
    print(f"  per-task drop variance: {rule_results['rule_c_drop_var_pp2']:.1f} pp²"
          f" (threshold 20.0)")
    print(f"  → Rule C fires: {rule_results['rule_c_fires']}")

    print(f"\nRule D (thesis falsification):")
    print(f"  ≥ 8 tasks with mean drop ≥ 50pp:  {rule_results['rule_d_severe_cond']} "
          f"(actual {rule_results['rule_d_severe_tasks']})")
    print(f"  identity SR ≥ 95%:                {rule_results['rule_d_identity_cond']}")
    print(f"  → Rule D fires: {rule_results['rule_d_fires']}")

    triggered = rule_results["triggered_rule"]
    print(f"\n{'=' * 70}")
    print(f"TRIGGERED RULE: {triggered}")
    print(f"{'=' * 70}")
    print(f"\n{framing_implication(triggered)}")

    # Save
    out = {
        "version": "stage3_mcnemar_analysis_v1",
        "input_file": str(INPUT_PATH),
        "n_cells": len(per_cell_sr),
        "per_cell_sr": {f"task{t}_{p}": sr for (t, p), sr in per_cell_sr.items()},
        "mcnemar_results": mcnemar_results,
        "rule_results": rule_results,
        "framing_implication": framing_implication(triggered),
    }
    with open(OUTPUT_PATH, "w") as f:
        json.dump(out, f, indent=2)
    print(f"\nWritten: {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
