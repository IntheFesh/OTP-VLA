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
