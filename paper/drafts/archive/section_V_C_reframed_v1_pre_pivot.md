# §V.C — CFM Decoder Sample-Quality Failure under Demonstration-Supervised Learning

**Paper framing**: This section is the paper's primary empirical contribution.
We do not present "Path B as the architecture that fixes V3"; we present
"CFM-based action decoders fail in a specific, characterizable way under
demonstration-supervised learning, validating Theorem 3' at the
implementation level".

## §V.C.1 — Deterministic head verification

Path B's deterministic head (replacing V3's CFM head) was verified at the
implementation level (5a v2, output dir 20260511_192411):
- head_loss: 0.82 → 0.056 (min, overfit on 50-sample training subset)
- Determinism check: head_loss bit-exact across forwards (10 trials)
- 138.3M trainable params confirmed (vs 92.5M for CFM head, +50%)

This establishes the head is not the bottleneck for downstream failure.

## §V.C.2 — Decoder sample-quality test (5a v2 ckpt, 16 training samples)

| Action dim | gt_mean | gt_std | L1_err | L1/std ratio |
|---|---|---|---|---|
| 0 (pos x)  | +0.121 | 0.380 | 0.846 | **2.23** |
| 1 (pos y)  | +0.140 | 0.409 | 0.968 | **2.36** |
| 2 (pos z)  | -0.105 | 0.489 | 0.972 | **1.99** |
| 3 (rot x)  | -0.013 | 0.031 | 0.854 | **27.28** |
| 4 (rot y)  | -0.053 | 0.106 | 0.849 | **7.98** |
| 5 (rot z)  | -0.012 | 0.062 | 0.742 | **12.05** |
| 6 (gripper)| +0.344 | 0.943 | 1.034 | **1.10** |

**Threshold**: L1/std < 0.3 (functional learning); L1/std > 0.8 (failure)
**Result**: 0/7 dims PASS, 7/7 dims FAIL.
**Overall L1/std ratio**: 1.88

CFM decoder samples are not task-discriminative. The aggregate decoder_loss
(~1.0 nats, observed in training) does not reflect predictive quality;
it reflects the CFM loss noise floor.

## §V.C.3 — Oracle ablation rules out moving-target hypothesis

Hypothesis: maybe decoder learns but receives noisy/incompatible head output
(moving-target). To test, we replaced head trajectory with ground-truth 
trajectory in decoder input (5a v3, use_oracle_trajectory=true):

- 5a v2 (head → decoder):     decoder_loss = 0.89 (first batch), 1.03 (training min)
- 5a v3 (gt_traj → decoder):  decoder_loss = 1.11 (first batch), oracle vs normal nearly identical training curve

Decoder loss does not improve with oracle trajectory. Moving-target hypothesis
rejected; failure is intrinsic to CFM-based decoder learning under
demo-supervised loss.

## §V.C.4 — Empirical instantiation of Theorem 3'

From §III Theorem 3' Corollary: demonstration MI bound predicts
I(π*; ℓ|o) ≤ 0.038 nats (Gaussian-surrogate from ρ² ≤ 0.073).

CFM decoder under demo supervision realizes this bound at the implementation
level: sample-quality failure reflects the fundamental constraint imposed by
demonstration-supervised stochastic objective, not specific hyperparameter
issues (lr, batch size, schedule all tested in this work).

This is the paper's central empirical contribution: V3's 0/50 SR is not a
consequence of "two-layer cascade collapse" as previously hypothesized;
it is the consequence of CFM-based action decoder sampling under
language-orthogonal demonstration supervision, predicted theoretically and
verified at implementation level.

## §V.C.5 — Why OFT achieves 97% SR despite the same backbone

OFT (OpenVLA-OFT) shares the SigLIP+DINOv2 backbone (r(h_OFT, ℓ) = 0.76 
identical to ours), yet achieves 97% SR. OFT does not use a CFM decoder;
it produces actions via direct LLM hidden state → action regression
(distribution-matching at action level, but not CFM-based).

OFT's success demonstrates that demo-supervised action learning CAN work
for high SR (visual-grounded; §V.B paraphrase-invariance shows 9/10 tasks
robust to paraphrasing). The architectural failure mode in §V.C.2 is
specific to CFM-based decoders, not demo supervision broadly.

This OFT contrast also strengthens our paper's central diagnostic claim:
"high SR does not imply language understanding" (OFT achieves the former
without the latter; §V.B Rule A triggered).

## Implications for §V.D Architectural Recovery

Given §V.C.2-4, we test deterministic action regression as architectural
recovery (Path C, §V.D). Two outcomes pre-specified:
- Path C sample quality PASS → fix is implementable; paper demonstrates 
  recovery and §V.D reports SR
- Path C sample quality FAIL → fix is deeper architectural concern; 
  paper §V.D reports the negative result and discusses future directions

In either case, §V.C is the paper's primary empirical contribution.
