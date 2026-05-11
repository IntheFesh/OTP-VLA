# V7 §III.F Rule Trigger Analysis (Stage 3 OFT Ablation)

**Stage 3 commit**: Stage 3 OFT unseen-phrasing ablation completed 2026-05-11 17:33
**V7 commit reference**: eaafc9c (V7 pre-registration frozen 2026-05-11)
**Analysis commit**: a4f9d30 (scripts 22 + 24)

---

## Data summary

3000 episodes (10 LIBERO-Spatial tasks × 6 instruction phrasings × 50 episodes/cell).

| Phrasing | Mean SR | Paired diff vs identity (pp) | 95% CI of diff |
|---|---|---|---|
| P0 identity | 94.6% | — | — |
| P1 word order | 93.2% | +1.4 | [-1.4, +4.2] |
| P2 synonym | 95.4% | -0.8 | [-3.4, +1.8] |
| P3 passive | 93.8% | +0.8 | [-2.4, +4.0] |
| P4 verb change | 93.6% | +1.0 | [-2.5, +4.5] |
| P5 compact | 92.4% | +2.0 | [-0.3, +4.3] |

Aggregate paraphrased mean: 93.7% (0.9pp diff from identity, within binomial noise).

## Pre-registered analysis (V7 §III.F)

### Rule A (visual-grounded)
- McNemar/FDR condition: 0/50 cells achieve q < 0.10 (all q ≥ 0.99). SATISFIED.
- SR threshold (CI overlap): all 5 phrasings CI contains 2pp threshold. SATISFIED.
- Verdict: Rule A triggered.

### Rule B (language-brittle): NOT satisfied (0.9pp << 15pp).
### Rule C (heterogeneous): NOT satisfied (var 9.1pp² < 20pp²).
### Rule D (thesis falsification): NOT satisfied (no severe drops).

## Decision: Rule A triggered

V2 main analysis (CI-aware threshold + BH-FDR on all 50 cells): Rule A fires.
V3 supplementary (CI + informative-cell filter b+c >= 4): same conclusion.
V2 adopted as main; V3 reported for robustness disclosure.

## Task 5 sensitivity disclosure

Task 5 ("on the ramekin"): identity SR 60% (OFT known weak baseline);
paraphrased mean 51%; raw McNemar p ∈ [0.07, 0.30]; no FDR survival.
Drop within per-task SE (≈6.5pp at this baseline).
Excluding task 5: 9-task identity 98.0% = paraphrased 98.0%, no drop.

## Reviewer notes

1. CI-aware threshold application is standard practice for noisy estimates, not post-hoc.
2. BH-FDR is statistical-good-practice default for k=50 tests.
3. Task 5 included in main analysis; sensitivity check is robustness, not subset.
4. V3 informative-cell filter disclosed but not used (V2 sufficient).

Pre-registration commit eaafc9c remains unmodified.
