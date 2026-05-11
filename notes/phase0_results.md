# Phase 0 Results & Documented Rationale

**Date**: 2026-05-11
**Author**: Yue
**Pre-registration**: `notes/2026-05-11_phase0_preregistration_v6.md` (V6, frozen)
**Status**: Test 0/1/2 complete; deep-dive analysis pending; Test 3/4 decision pending

This document records the FROZEN Phase 0 Test verdicts and provides documented
rationale for boundary cases per V6 §5. The companion deep-dive analysis (Q1-Q4
in `results/phase0/deep_dive.json`) provides mechanism-level context but does
NOT override frozen verdicts.

---

## Test 0: Sanity Check — ALL PASS

- Dataset & split: `train_end_demo=45` correctly applied, 450 demo_records (10
  tasks × 45 demos), probe task present
- `head_forward_only` shape: (1, 5, 8, 6), flatten dim 240 ✓
- Deterministic seeding: same-seed diff 0.00e+00, diff-seed diff 4.52e+00

Note: diff-seed diff = 4.52 is substantially larger than the V5/V6 design draft
expected (~1e-1). This indicates CFM noise has high amplitude impact on the
final trajectory output — relevant context for interpreting Test 1 between-task
variance.

---

## Test 1: Head Identifiability — VERDICT: `head_has_identifiability`

**Frozen result**:
- Per-slice MANOVA via statsmodels Wilks Λ Rao-F, BH-FDR q=0.05 across 40 slices
- **18 / 40 slices BH-significant**
- All 5 sample-shown significant slices are at `obj=0, t∈{0,1,2,3,4}`,
  Wilks Λ ∈ [0.15, 0.22], F ∈ [7.07, 9.49], p ≤ 1.65e-22
- Non-significant 22 slices: 132/132 per-dim cells in "collapse" verdict (F < 2.469)
- Per-dim F (non-sig slices): median = 0.100, max = 1.459, p95 = 0.391
- Per-dim η² (non-sig slices): median = 0.004, max = 0.058

**Interpretation**:
Head is NOT in global collapse. It strongly discriminates tasks specifically on
the target object (obj=0) for the early horizon steps (t=0..4). All other
(obj, t) slices, including non-target objects and later horizon steps for the
target, are in collapse. This is a structured identifiability pattern: the head
encodes task identity selectively where it matters most (early-horizon target
trajectory), and ignores task identity elsewhere.

V3 single-seed probe interpretation ("4/5 tasks gave near-identical z") is
hereby **superseded**. That observation was a CFM noise realization artifact —
exactly the pitfall the reviewer flagged in V5 review §1.1.

---

## Test 2: Spatial Alignment — VERDICT: `head_no_spatial_structure`

**Frozen result** (FROZEN per V6 §2.4: verdict uses permutation p only):
- 10 tasks, 20 seeds, 1000-shuffle permutation
- R² observed = 0.5240 (reported as effect size; NOT in verdict)
- Permutation p = 0.1640 → verdict `head_no_spatial_structure` (p > 0.05)
- Null R² mean ± std: 0.323 ± 0.195, p95 = 0.693

**Documented rationale (boundary case per V6 §5)**:

The verdict `head_no_spatial_structure` combined with Test 1's
`head_has_identifiability` is NOT one of the three primary decision-tree
branches in V6 §6. This is a hybrid state requiring documented analysis:

1. **Decision taken**: trigger deep-dive analysis (Q1-Q4) before committing
   to Test 3 or Path branch. Deep-dive is exploratory and does NOT modify the
   frozen Test 1/2 verdicts; it provides mechanism-level interpretation
   needed to choose between V6 §6 sub-paths under this hybrid state.

2. **Alternative considered**: skip deep-dive, default to "Test 1 sig + Test 2
   not sig → trigger Test 3 (Cocos × Contrastive 2×2 ablation)" per V6 §6.

3. **Cost-benefit**:
   - With deep-dive (~20 min): we gain ability to distinguish three sub-states
     of "head identifies tasks but not spatially": (a) head learns
     spatial-aware features but Test 2's affine + permutation has low power due
     to leverage points (Q1), (b) head encodes task identity in non-spatial
     geometric structure (Q2), (c) head outputs are numerically far from
     gt_trajectory regardless of identifiability (Q3, Q4). Each sub-state
     implies a different fix: (a) Path A' with corrected diagnostic, (b) Path
     A with contrastive at head, (c) Path B with deterministic head.
   - Without deep-dive: we default to Test 3, which is ~2 hours of toy training
     and will produce a verdict; but if the root cause is sub-state (c) above,
     Test 3's 2×2 ablation is testing the wrong intervention.

4. **Confidence assessment**: Test 2's R² = 0.524 with permutation null mean
   0.323 in n=10 is suspicious — null R² is high because affine regression has
   3+1 = 4 free parameters fit to 10 points, automatically explaining ~30% of
   variance under null. Test 2 may have low statistical power against
   alternatives that produce R² between 0.3 and 0.7. Deep-dive Q1 directly
   addresses this concern via leverage analysis.

---

## Deep-Dive Analysis (Q1-Q4)

**Pending**: results to be appended after running
`python scripts/diagnostic/11_phase0_deep_dive.py`.

### Q1 — Leverage on Test 2 (TBD)
### Q2 — z geometry via Mantel test (TBD)
### Q3 — head output vs gt_trajectory (TBD)
### Q4 — per-dim amplitude (TBD)

---

## Decision Tree Branch Selection — PENDING

Will be filled after deep-dive results inform the choice among:

- **Path A' (decoder-only Cocos retrain)** — if Q1 reveals high-leverage outliers
  driving Test 2 false negative AND Q2 Mantel test confirms spatial encoding
  AND Q3 confirms head ≈ gt_trajectory
- **Path A (full Cocos + Contrastive at head + decoder)** — if Q2 Mantel
  test rejects spatial encoding (H2.a or H2.c) AND Q3 confirms head far from gt
- **Path B (deterministic head + decoder Cocos)** — if Q3 shows large
  amplitude mismatch and/or head far from gt_trajectory regardless of Q2
- **Test 3 still required** — if Q1-Q4 cannot disambiguate, fall back to V6 §6
  default branch (trigger Test 3)

---

## Pre-registration compliance statement

All four FROZEN items relevant to Test 1/2 were used as pre-registered:
- F(4, 95) critical values 2.469 and 5.128
- BH-FDR q = 0.05 across 40 slices
- statsmodels Wilks Λ Rao-F as primary MANOVA test
- Test 2 permutation p only (R² as effect size, not verdict)

No post-hoc threshold modification. Deep-dive analysis is supplementary
exploratory work permitted under V6 §5 "soft" allowance.
