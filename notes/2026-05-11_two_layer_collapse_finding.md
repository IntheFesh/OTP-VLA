# OTP-VLA Two-Layer Conditional Collapse — 2026-05-11

## Status
30-epoch retrain converged but sim eval 0/50 SR.
Diagnosed as two-layer conditional-to-marginal collapse (head + decoder).

## Loss convergence
- Step 50:    head=1.93, decoder=1.26
- Step 16000: head=0.88, decoder=0.39
- Step 99000: head=0.27, decoder=0.14
- 30 epochs, 9h17min wall, paper_ready_ckpt at step 99000

## Sim eval results
- 10 task × 5 ep, num_samples=1, reduction=none: **0/50**
- task 0 × 5 ep, num_samples=20, reduction=mean: **0/5**
- All episodes terminate at max_steps=600

## Decisive test (decoder layer): 4 of 7 action dims = training marginal mean
| dim | label | GT marginal mean | model output mean | diff |
|---|---|---|---|---|
| 0 | Δpos_x | +0.150 | +0.363 | 0.21 ✓ |
| 1 | Δpos_y | +0.134 | +0.259 | 0.13 ✓ |
| 2 | Δpos_z | −0.154 | **−0.174** | **0.02** ✗ |
| 3 | Δrot_x | −0.005 | +0.003 | 0.008 ✗ |
| 4 | Δrot_y | −0.011 | −0.046 | 0.03 ✗ |
| 5 | Δrot_z | −0.020 | +0.002 | 0.02 ✗ |
| 6 | Gripper | +0.092 | −1.125 | 1.22 ✓ |

4 dims (2,3,4,5) match marginal within 0.03 → CFM ignoring conditioning on these dims.

## Head layer probe (today): OTPHead z also collapsed
Tested 5 different LIBERO-Spatial tasks, same seed → measure cross-task variation in z[obj0, t=0, :].

| z dim | cross-seed std (same task) | cross-task std (different tasks) | ratio task/seed |
|---|---|---|---|
| 0 | 0.114 | **0.033** | 0.29× |
| 1 | 0.447 | 0.807 | 1.81× |
| 2 | 0.468 | 0.825 | 1.76× |
| 3 | 0.113 | **0.060** | 0.53× |
| 4 | 0.113 | 0.092 | 0.82× |
| 5 | 0.132 | **0.046** | 0.35× |

4 of 6 z dims have cross-task std lower than cross-seed CFM noise → head producing nearly task-invariant z.

Per-task z[obj0, t=0, :3]:
- "wooden cabinet":   [+0.16, −1.53, −1.68]
- "next to ramekin":  [−0.03, −1.52, −1.78]
- "on cookie box":    [+0.00, −1.53, −1.97]
- "next to plate":    [−0.04, −1.60, −1.76]
- "on ramekin":       [+0.20, +2.86, +2.53] (outlier)
4 of 5 tasks produce nearly identical z.

## Schema verified clean
- predictor._build_batch vs assemble_batch identical (6 keys, shape+dtype)
- Quat convention: sim wxyz == so3_to_quat output wxyz
- Train data consistency: Pearson corr(Δee_z, action_z) = 0.974
- Inference: num_sample_steps head=16, decoder=8 (not 1-step bias)
- Action range OK, all finite

## Diagnosis
**Two-layer collapse**: image+lang → [head CFM collapse → task-invariant z] → [decoder CFM collapse → marginal-mean action] → 0% SR

Per-dim training data std:
- Δpos_x/y/z: 0.41/0.35/0.51
- Δrot_x/y/z: 0.04/0.07/0.06 (10× smaller)
- Gripper: 0.99
→ rotation dims have effectively zero gradient signal, collapse first

Dong et al. 2025 (Conditioning Matters, arXiv:2505.11123) Theorem 1 documents this for single-stage CFM. Two-layer case appears unreported in literature.

## Pending decision
Path B (recommended): head → L1 regression deterministic; decoder keeps CFM + add Cocos source + cross-attention
Path C (safety net): both head and decoder → L1 regression (matches OpenVLA-OFT recipe)
Path α: pivot to workshop, write current finding as negative result paper

## Key paths
- ckpt: results/otp_soft_30ep_split45_20260510_183531/ckpt_step0099000.pt
- config: configs/otp_soft_30e.yaml
- predictor: otp/eval/predictor.py (653 lines)
- eval loop: scripts/eval/eval_standard.py
- CFM diagnostic: scripts/diagnostic/05c_action_distribution_histogram.py
- flow matching: otp/utils/flow_matching.py
