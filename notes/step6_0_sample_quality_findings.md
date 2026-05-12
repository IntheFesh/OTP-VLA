# ⚠️ SUPERSEDED — DAY 2 PAPER-PIVOT (2026-05-12 evening)

**Status update (2026-05-12 evening)**: The "decide Step 6 launch" and "V7 amendment to det+det" items at the bottom of this file are **no longer active decisions**. After Day 2 Phase 1 verification revealed `use_flow_matching=False` does not produce deterministic regression (it produces vanilla CFM, not Path C), strategic decision was made to **drop Path C implementation** and pivot to full paper-writing track.

**Current framing** (locked 2026-05-12 evening):
- Theorem 3' is architecture-agnostic — no architectural choice (deterministic head, Cocos, Path C) bypasses the bound
- Paper sells the **diagnostic methodology** (Gate 1 + M3 + C1 + §V.C.2 sample-quality test), NOT an architectural fix
- §V.C v4 framing: "CFM decoder fundamental failure mode under language-orthogonal demonstration supervision" (consistent with this file's empirical findings, but no "Step 6" follow-up)
- Step 6 / Cocos / Path B variants → 7.5-day GPU sequential ablation plan as comparative baselines, not as proposed fix

**Empirical findings in this file remain valid and authoritative.** The L1/std table, oracle test results, and decoder failure analysis are unchanged. Only the "what to do next" recommendations at the bottom of this file are superseded.

**Authoritative current framing**: `paper/drafts/otp_vla_paper_drafts_consolidated.md` §V.C v4.

---

# Step 6.0 Sample Quality Test Findings (2026-05-12)

## Test setup
- Ckpt: results/pathB_step5_overfit_20260511_192411/ckpt_step0000500.pt (5a v2)
- Architecture: deterministic head (verified integration test) + CFM decoder
- Test: decoder.sample() on 16 training samples, compare pred_action vs gt_action

## Result: 0/7 PASS, 7/7 FAIL

| dim | gt_std | L1_err | L1/std |
|---|---|---|---|
| 0 (pos x) | 0.380 | 0.846 | 2.23 |
| 1 (pos y) | 0.409 | 0.968 | 2.36 |
| 2 (pos z) | 0.489 | 0.972 | 1.99 |
| 3 (rot x) | 0.031 | 0.854 | 27.28 |
| 4 (rot y) | 0.106 | 0.849 | 7.98 |
| 5 (rot z) | 0.062 | 0.742 | 12.05 |
| 6 (gripper) | 0.943 | 1.034 | 1.10 |

Pre-spec threshold: L1/std < 0.3 (functional), L1/std > 0.8 (fail)
Overall L1/std ratio: 1.88 — sample quality is noise-level

## Implications

1. CFM decoder loss noise floor (~1.0 nats) was masking actual failure.
   Decoder is not in "learned and noisy CFM loss" state; it is in
   "did not learn" state.

2. Retroactively confirms V3 0/50 SR mechanism: not "two-layer cascade
   collapse" as previously hypothesized in V7 §V.A; it is CFM decoder
   sample quality failure under demo-supervised learning.

3. Validates Theorem 3' at implementation level: demo MI bound predicts
   I(π*; ℓ|o) ≤ 0.038 nats; sample quality realizes this constraint.

4. V7 frozen Path B definition (det head + CFM decoder + Cocos) needs
   amendment — sample quality fail invalidates the "CFM decoder is fixable
   with deterministic head" assumption.

## Paper framing pivot (committed)

Paper §V.C reframed as primary empirical contribution:
"CFM Decoder Sample-Quality Failure under Demonstration-Supervised Learning"

Not "Path B as architectural fix" but "Path B reveals architectural
failure mode + theoretical foundation + recovery investigation (§V.D)".

Pivot is narrative reframe, not fallback. Even if Path C (deterministic 
decoder) succeeds with high SR, paper centers on the diagnostic
contribution, not the architectural fix.

## Next steps (Path C)

1. Verify use_flow_matching=False produces actual deterministic regression
2. Create 5c config, integration test
3. Train 5c-normal (head→det decoder) + 5c-oracle (gt→det decoder), tiny data
4. Sample quality test on both with pre-specified threshold
5. If pass: full-data 1 epoch sanity
6. Decide Step 6 launch protocol OR V7 amendment to det+det
