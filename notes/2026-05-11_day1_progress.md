# OTP-VLA Day 1 Implementation Progress (2026-05-11)

**Author**: Yue + Claude session.
**Scope**: First implementation day after V7 pre-registration commit. Phase 0 → V7 → Path B core + Step 6 prep + Stage 3 ablation in-flight.

---

## §I. State at Day 1 Start (morning, 2026-05-11)

Inherited from prior sessions:
- Phase 0 diagnostic battery complete (Test 0/1/2/4 + Q1-Q4 + C1/C2 + Gate 1/2 + SC1-3 + M3 + M3 robust)
- Three-layer empirical chain established:
  - Backbone faithful: $r(h_{\text{OFT}}, \ell) = +0.76$
  - Demonstrations language-orthogonal: $r(\text{traj}, \ell) \in [-0.27, -0.20]$
  - V3 policy unfaithful: $r(z_{\text{OTP}}, \ell) = -0.19$
- Capacity ruled out: $\bar F^a = 11.30$
- V6 pre-registration frozen.

Outstanding decisions at start of day:
- V7 not yet committed (was held pending OFT ablation outcome — incorrect approach, see §II)
- OFT predictor not implemented (was placeholder NotImplementedError in `OTPSoftPredictor.predict_chunk`)
- Path B implementation plan not started (V7 §IV).

---

## §II. V7 Pre-Registration Decision (mid-morning)

Reviewer-Yue exchange clarified that V7 commit should not block on OFT ablation outcome. Pre-registration standard practice is to commit **conditional decision rules** ("if X observed, run Y"), not concrete conclusions. V7 frozen with:

**Theoretical framework (§II)**:
- Theorem 2 reformulated: $C2: I(z; \tau \mid o) > 0$ (replacing weaker $H(z|o) > 0$)
- Lemma 1 (C3-preservation under contrastive training) — frozen since V5
- **Theorem 3 (Supervision-induced faithfulness collapse)** — main theoretical contribution:
  $$I(a; \tau \mid o) = 0 \Rightarrow I(\pi^*; \tau \mid o) = 0$$
  for per-sample distribution-matching objectives, regardless of backbone capacity.
- Corollary (necessity of language-aware supervision)
- Remark (compositional faithfulness, weakened DPI)

**OFT ablation protocol (§III) frozen**:
- 10 LIBERO-Spatial tasks × 6 phrasings (1 identity + 5 paraphrased semantic-preserving)
- 50 episodes per (task, phrasing) cell = 3000 episodes
- Identity SR pass gate ≥ 92% (binomial SE ≈ 0.78pp for n=500)
- Per-task McNemar test on paired outcomes
- 4 conditional rules (mutually exclusive): A (visual-grounded), B (language-brittle), C (mixed), **D (thesis falsification fallback)**
- **Plan B**: V3 self-ablation substitute if 16-hr debug cap exhausted

**Path B implementation plan (§IV)** — 6 steps with z-distribution test as first-class:
1. ckpt save bug fix
2. deterministic head
3. z-distribution compatibility test
4. decoder Cocos source distribution
5. single-task overfit sanity
6. multi-seed retrain (3 seeds × 50 epoch)

**§V framing rules**: per-outcome paper §V.B template (Rule A/B/C/D/Plan B) frozen.

Commit: `eaafc9c` (V7 frozen 2026-05-11).

---

## §III. OFT Predictor Implementation (afternoon, 12:00-13:30)

Three rounds of bug isolation before Stage 1 PASS:

| Bug | Fix | Commit |
|---|---|---|
| `OpenVLAOFTPredictor.__init__` was stub (NotImplementedError) | Adopted `experiments/robot/libero/run_libero_eval.py` reference impl directly via sys.path insert | `22d93a1` |
| `run_libero_eval.py:299` called `env.get_observation()` (broken in current LIBERO version) | Always pass `initial_state` from `task_suite.get_task_init_states(task_id)` to bypass else branch | `c96c5e8` |
| PyTorch 2.6+ default `weights_only=True` broke LIBERO `init_states` load (numpy reconstructor) | Monkey-patch `torch.load` at script top before LIBERO imports | `037700f` |

Stage 1 (determinism, 1 task × 5 ep × 2 runs): **5/5 PASS, bit-exact match**.

Stage 2 first attempt:
- Identity SR 31/50 = 62%, gate FAIL.
- Diagnosis: `SPATIAL_PHRASES` list was **alphabetical**, but LIBERO `task_suite.get_task(i)` uses an **arbitrary task-index ordering**. Only tasks 0 and 9 happened to align; rest received instructions referring to wrong spatial phrase.
- Fix: corrected ordering via `task_suite.get_task(i)` + `get_libero_env(task)` default_desc extraction.

Commits:
- `4c524ba` — SPATIAL_PHRASES corrected to true LIBERO task ordering.
- `547ad52` — V7 amendment commit recording the typo-level protocol correction.

Stage 2 re-run: **50/50 = 100%, V7 §III.D gate PASS**.

Stage 3 (3000-episode full ablation) launched: `nohup python -u ...` in background.

---

## §IV. Path B Implementation Steps 2/3/4 (afternoon, 14:00-15:30)

### Step 1: NO-OP

V6 memory claimed "16GB per ckpt requires `requires_grad=True` filter fix". **Investigation showed `train_otp_soft.py:252-258` already implemented this filter**. Actual ckpt size = 1.1GB = trainable_state (~410MB) + AdamW optimizer state (~820MB). V6 memory was stale.

### Step 2: Deterministic head — commit `60f610c`

Added `OTPHead(deterministic=True)` mode bypassing ShortcutFlowMatching:
- Cross-attention output (B, N_obj × hidden_dim) → 3-layer MLP (GELU) → trajectory (B, N_obj × H × 6)
- Training: L1 loss vs gt_trajectory (replaces CFM consistency loss)
- Inference: 1-shot deterministic forward (no sampling, no noise)
- 31.59M trainable params (deterministic mode)
- `velocity_net = None`, `flow_matcher = None` when `deterministic=True`

Smoke tests passed: bit-exact sample reproducibility, finite L1 loss, finite backward grad.

### Step 3: z-distribution compatibility — commit `6b07a31`

Loaded `results/phase0/test1_Z.npy` (V3 CFM head's 100 trajectory samples, shape (5, 20, 5, 8, 6)). Sampled 100 trajectories from random-init Path B deterministic head.

Results:
- V3 per-sample flat norm: 21.04, Path B: 0.44 (48× mismatch)
- Per-dim std median ratio: 7.52 (outside [0.5, 2.0] threshold)
- Mean-vector cosine sim: 0.036 (near-orthogonal)
- Top eigenvalues: V3 [14.4, 6.6, 2.8] vs Path B [0.006, 0.005, ...]

**Decision**: V3 decoder weights **NOT inheritable**. Path B decoder retrains from scratch. (This is the expected outcome per V7 design — Step 3 formally records the rationale.)

### Step 4: Decoder Cocos source distribution — commits `a9eba33` + `97ee71a`

Two-part implementation:

**4a (`a9eba33`)**: Added `x_0_source: Optional[Tensor] = None` parameter to `FlowMatching.forward` and `ShortcutFlowMatching.forward`. When None, default `torch.randn_like(x_1)` (preserved). When provided, replaces with caller-supplied source. `sample()` method already supported `x_0` parameter.

**4b (`97ee71a`)**: Added `CocosSourceMLP` class to `LanguageAgnosticDecoder`:
- 3-layer MLP with GELU: condition_dim (=512) → target_dim (=H × action_dim = 56)
- New decoder params: `use_cocos_source: bool = False`, `cocos_alpha: float = 1.0`, `cocos_beta: float = 1.0`
- Training: $x_0\_src = \alpha \cdot F_\phi(c) + \beta \cdot \text{randn}$ passed to flow_matcher
- Inference: same shifted source passed to `flow_matcher.sample(x_0=...)`
- C3 invariants preserved (no language fragment in submodule names)
- 0.21M extra Cocos params

Smoke tests passed: training loss finite (1.24), inference shape (B, H, 7) preserved, backward grad finite (6.80).

---

## §V. Step 5 + Step 6 Preparation (15:30-16:30)

### Step 5 configs (commit `f4c61bd`)

Two variants for tonight's overfit sanity:
- `configs/otp_soft_pathB_overfit.yaml` (5a, base Path B, no Cocos)
- `configs/otp_soft_pathB_overfit_cocos.yaml` (5b, Cocos enabled)

Both: `deterministic=true`, `lr=1e-4`, `num_steps=500`, `batch_size=8`, `train_end_demo=5`. Pass criterion: L1 loss → 0.01 within 500 steps.

### Step 6 prep (commit `ce8cc06`)

3 seed configs (`otp_soft_pathB_seed{42,137,2026}.yaml`):
- Inherits `otp_soft_frozen`
- `deterministic=true`, `use_cocos_source=false` default
- `lr=1e-4`, `num_epochs=50`, `batch_size=16`, `ckpt_every=5000`

Launch script `scripts/launch/launch_pathB_step6.sh`:
- Sequential 3-seed launcher with optional `--cocos` flag
- Pre-flight GPU memory check (warn if <60GB free)
- nohup-friendly

Aggregation script `scripts/diagnostic/20_pathB_multiseed_aggregate.py`:
- Loads each seed's final ckpt + runs LIBERO sim eval (10 tasks × 50 ep)
- Reports SR mean ± std per-task and overall

### Step 5 launcher (commit `e645217`)

`scripts/launch/launch_pathB_step5.sh`:
- Auto-detects Stage 3 completion via `ps aux` polling (120s interval)
- After Stage 3 done + 60s GPU cleanup, runs Step 5a → 5b sequentially
- Failure-tolerant (5b runs even if 5a crashes)
- Launched at 14:29 with `nohup` (PID 111263)

### Stage 3 McNemar analysis (commit `a1e59cc`)

`scripts/diagnostic/21_stage3_mcnemar_analysis.py`:
- Reads `results/phase0/oft_unseen_phrasing_full.json`
- Per-cell SR table + per-cell McNemar tests
- Applies V7 §III.F rules A/B/C/D with explicit thresholds
- Outputs triggered rule + paper §V.B framing implication

---

## §VI. Stage 3 Status (15:30+, in-flight)

OFT 3000-episode full ablation running in background (PID 36197, launched 13:32).

**Progress at 15:30**: ~15/60 cells done (Global SR ≈ 95%). Currently in task 5 × P1_word_order.

**Notable data**:
- All P0_identity cells: ≈95-100% SR (gate replicated)
- P1_word_order on tasks 0/1/3: 100% SR
- **First paraphrased SR drop seen**: task 5 × P1_word_order = ~48% (50% drop). Possibly task-specific (P1 word-order reshuffling combined with "on the ramekin" spatial phrase).
- ETA: 168 min ≈ 2.8 hr → completion expected ~18:00.

---

## §VII. Tomorrow's Workflow

When user wakes up (post Stage 3 + Step 5 completion):

1. **Verify completion**:
   ```bash
   ls -la /root/autodl-tmp/OTP-VLA/results/phase0/oft_unseen_phrasing_full.json
   tail -50 /root/autodl-tmp/OTP-VLA/logs/launcher_step5_*.log
   ```

2. **Run Stage 3 analysis** (gives V7 §III.F triggered rule):
   ```bash
   cd /root/autodl-tmp/OTP-VLA
   python scripts/diagnostic/21_stage3_mcnemar_analysis.py
   ```
   This decides paper §V.B framing direction.

3. **Review Step 5 results** (decides Cocos toggle for Step 6):
   - Inspect `logs/pathB_step5a_*.log` for base Path B loss curve
   - Inspect `logs/pathB_step5b_*.log` for Cocos variant
   - Pass criterion: final 50 steps mean L1 < 0.01
   - If both pass: choose variant with lower final loss for Step 6
   - If only 5a passes: Step 6 without Cocos
   - If only 5b passes: Step 6 with Cocos (suggests Cocos is essential)
   - If both fail: deterministic head architectural issue → redesign needed

4. **Launch Step 6 multi-seed retrain**:
   ```bash
   # With Cocos:
   nohup bash scripts/launch/launch_pathB_step6.sh 42 137 2026 --cocos \
     > logs/launcher_step6_$(date +%Y%m%d_%H%M).log 2>&1 &
   
   # Or without Cocos:
   nohup bash scripts/launch/launch_pathB_step6.sh 42 137 2026 \
     > logs/launcher_step6_$(date +%Y%m%d_%H%M).log 2>&1 &
   ```
   Wall clock: ~37.5 hr sequential (~1.5 days). Survives SSH disconnect.

5. **Start paper §V.B drafting** based on Stage 3 verdict:
   - V7 §V has per-outcome framing templates (Rule A/B/C/D/Plan B)
   - Stage 3 analysis already prints which framing applies
   - Fill in data tables (per-cell SR, McNemar p-values, per-task drops)
   - Theorem 3 proof appendix (V7 §VII.1 deferred)

---

## §VIII. Commits Summary (15 commits today)

| Commit | Description |
|---|---|
| `1233093` | Phase 0 incremental fixes (predictor + diagnostic 06-08) |
| `e95fdc2` | gitignore Jupyter checkpoints |
| `c2263a6` | OFT backbone phrasing-robustness script (script 17) |
| (record) | `notes/phase0_results.md` finalized |
| **`eaafc9c`** | **V7 pre-registration frozen** (Theorem 3, Rules A/B/C/D, Plan B) |
| `22d93a1` | OFT unseen-phrasing V1 (script 18) |
| `037700f` | `weights_only=False` torch.load monkey-patch |
| `c96c5e8` | `initial_state` always-passed fix (LIBERO `get_observation` bug) |
| `4c524ba` | SPATIAL_PHRASES alphabetical → LIBERO task-index ordering |
| `547ad52` | V7 amendment (SPATIAL_PHRASES typo correction record) |
| **`60f610c`** | **Path B Step 2**: deterministic head |
| **`6b07a31`** | **Path B Step 3**: z-distribution compatibility test |
| `a9eba33` | Path B Step 4a: flow_matching `x_0_source` parameter |
| **`97ee71a`** | **Path B Step 4**: Cocos decoder source distribution |
| `f4c61bd` | Step 5 configs (5a base + 5b Cocos) |
| `ce8cc06` | Step 6 prep: 3 seed configs + launcher + aggregation |
| `e645217` | Step 5 launcher (auto-detect Stage 3 completion) |
| `a1e59cc` | Stage 3 McNemar analysis script (V7 §III.F application) |

All commits pushed to `IntheFesh/OTP-VLA` main.

---

## §IX. Open Risks for Tomorrow

1. **Step 5 may fail (deterministic head can't overfit)** — would indicate base architectural issue. Mitigation: Step 5 launcher runs both 5a and 5b; if both fail, redesign needed before Step 6.

2. **Stage 3 Rule D might fire** — paper main thesis (Theorem 3 prediction) falsified. V7 §V.D Plan B fallback framing applies; paper sell point weakens to architectural comparison.

3. **Step 6 multi-seed sequential takes 37.5 hr** — if any seed crashes mid-training, launcher aborts remaining. Manual restart from crash point may be needed.

4. **`results/` dir is gitignored** — Stage 3 raw data, ckpts, JSON outputs not tracked in git. Backup to external storage if data is critical. AutoDL container persistence policy unknown.

5. **train_otp_soft.py has stashed WIP** — `git stash@{0}` contains "WIP train_otp_soft trainable_state key change". Stash content is cosmetic refactor of save block (style only, no behavioral change). Decide whether to apply or drop after Step 5 finishes.

---

*Record frozen 2026-05-11 ~16:30. Next session resumes from §VII workflow.*
