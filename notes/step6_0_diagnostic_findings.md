# Step 6.0 Diagnostic Findings (2026-05-11 ~18:30)

## Data source
- 5a ckpt: results/pathB_step5_overfit_20260511_173519/ckpt_step0000500.pt
- 5a train_log.csv: 500 steps, 50 samples × 5 demos × 10 tasks subset

## Critical finding: Decoder is not learning

| Metric | Step 10 | Step 500 | Change |
|---|---|---|---|
| Total loss | 3.80 | 2.43 | -36% |
| head_loss | 2.66 | 1.23 | -53% |
| decoder_loss | 1.14 | 1.19 | +5% (no learning) |

OTP head (31.59M params, deterministic) learns: head_loss drops from 2.66
to 1.23. But decoder (~92M total params via shared backbone) stuck:
decoder_loss varies between 1.0-1.4 throughout 500 steps with no
downward trend.

## LR schedule audit

| Step | LR |
|---|---|
| 10 (warmup) | 2e-5 |
| 50 (peak) | 1e-4 |
| 200 (mid-decay) | 7.5e-5 |
| 400 | 1.2e-5 |
| 480 | 4.9e-7 |
| 500 | 0.0 |

Cosine decay reaches 0 at step 500. Last 50 steps effective LR < 1e-6.
This is suboptimal but NOT primary cause — even mid-decay (step 200 at
lr 7.5e-5), decoder_loss already stuck.

## Hypothesis (to verify tomorrow)

H1: ShortcutFlowMatching noise floor
  - CFM loss = E_t[||v_pred - v_target||²], inherent t-sampling variance
  - For 56-dim target with std ~1.0, theoretical floor ~1.0
  - 1.19 may BE the floor

H2: Moving target (head output)
  - decoder receives head(...) which is still learning
  - decoder chases moving distribution

H3: Architectural coupling
  - head→decoder pipeline may have design issue

## Action plan for tomorrow

1. Inference script: feed oracle trajectory (gt) to decoder, check loss
   - If decoder_loss drops with oracle traj → H2 confirmed
2. Switch decoder to deterministic L1 (use_flow_matching=False), retest
   - If loss drops → H1 confirmed
3. Based on hypothesis, design fix
4. Re-run Step 5a v2 with fix, verify decoder learns
5. Then launch Step 6 (with --cocos AFTER cocos init fix)

## Step 6 launch decision

**DEFERRED until decoder learning issue resolved.**

Reviewer was right to push back against blind launch. 30 min diagnostic
revealed decoder bottleneck. Launching Step 6 tonight would waste
12.5+ hr GPU on architecturally-stuck decoder.

LR schedule extension may also be needed (cosine reaching 0 at training
end is too aggressive for 50-epoch full-data run).
