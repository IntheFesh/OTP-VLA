# §V.D Ablation Pre-Registration + Mini E2E Test Protocol

**Date**: 2026-05-13 Day 5 (locked before GPU launch)  
**Status**: Pre-registered before any data collection. Commits before GPU launch serve as timestamped pre-registration.  
**Authority**: Yue (strategic), Claude (implementation), Reviewer (technical audit).

---

## Section 1 — Ablation cell assignment

### 1.1 Architecture variants (4 cells × 3 seeds = 12 retrains, conditional on variant (c) gate)

| Cell ID | Variant | Head | Decoder | Source distribution | Seeds |
|---|---|---|---|---|---|
| A | Path B base | Deterministic | CFM (Shortcut) | N(0, I) | 42, 137, 2026 |
| B | Path B + Cocos | Deterministic | CFM (Shortcut) | Cocos N(αF_φ, β²I) | 42, 137, 2026 |
| C | Variant (c) reverse | CFM (Shortcut) | Deterministic regression (L1) | N/A | 42, 137, 2026 |
| D | V3 baseline | CFM (Shortcut) | CFM (Shortcut) | N(0, I) | 42, 137, 2026 |

**Conditional**: Cell C inclusion gated on mini E2E test pass (§3 below).

### 1.2 Conditioning input drop ablations (3 cells × 3 seeds = 9 retrains)

| Cell ID | Variant | Drop flag | Architecture base | Seeds |
|---|---|---|---|---|
| E | Drop mesh | drop_mesh=True | V3 baseline | 42, 137, 2026 |
| F | Drop grasp | drop_grasp=True | V3 baseline | 42, 137, 2026 |
| G | Drop proprio | drop_proprio=True | V3 baseline | 42, 137, 2026 |

### 1.3 Total retrains

- **If mini E2E pass**: 12 + 9 = 21 retrains × 12.5 hr = 262.5 hr = 11 days, ~$1500
- **If mini E2E fail**: 9 + 9 = 18 retrains × 12.5 hr = 225 hr = 9.4 days, ~$1290 (drop Cell C)

Plus LIBERO-Goal M3 cross-bench measurement: +1 hr GPU = $0.25.

### 1.4 Training hyperparameters (all cells)

- Learning rate: 1.0e-4 (per userMemories #1, NOT 1e-3)
- Epochs: 30
- Batch size: 16
- DataLoader workers: 4 (verified Day 5 morning)
- Optimizer: AdamW (existing)
- Backbone: OpenVLA-OFT, frozen (`backbone_mode: frozen`)
- Data: LIBERO-Spatial full training split

---

## Section 2 — Evaluation protocol

### 2.1 Per-task SR measurement

- Each cell × seed evaluated on 10 LIBERO-Spatial tasks
- 50 rollout episodes per task
- Total per cell × seed: 500 episodes
- Total across 21 cells × 3 seeds (max config): 31,500 episodes

### 2.2 Statistical analysis (pre-registered)

**Primary outcome**: Per-task mean SR averaged across 3 seeds per cell.

**Pairwise comparisons** (architecture cells):
- A vs D (Path B vs V3): does head choice matter?
- B vs A (Cocos vs N(0,I)): does source distribution matter?
- C vs A and C vs D (variant c vs Path B and V3): does decoder family matter? (if C exists)

**Drop ablation comparisons**:
- E vs D (drop mesh vs V3)
- F vs D (drop grasp vs V3)
- G vs D (drop proprio vs V3)

### 2.3 Statistical tests

- **McNemar paired test** per cell pair, applied per task (10 tasks per pair)
- **Benjamini-Hochberg FDR adjustment** across all pairs × tasks
  - 6 architecture pairs × 10 tasks = 60 comparisons (if all 4 cells run)
  - 3 drop pairs × 10 tasks = 30 comparisons
  - Total: 90 comparisons, BH-FDR target q < 0.05
- **Wilson 95% CI** per cell × task (n=150 episodes per cell-task = 3 seeds × 50 episodes)
- **Welch's t-test** for paired-aggregate seed-level SR comparison

### 2.4 Practical-equivalence threshold

±2 percentage points absolute SR difference.

### 2.5 Decision rules

- **Strong claim of decoder family axis (cell C exists)**: significantly different from D at q<0.05 AND/OR CI excludes ±2pp → §V.D reports "decoder family is significant determinant"
- **Decoder family axis untested (cell C fails mini E2E)**: §V.D reports "comprehensive decoder family ablation deferred to future work; §V.C.5 OFT contrast and Theorem 3' Remark 2 provide indirect evidence"
- **Conditioning input drops show small effect (all CI within ±2pp)**: §V.D reports "conditioning input richness is not the dominant SR determinant, consistent with Theorem 3' decoder-family interpretation"
- **Conditioning input drop shows large effect**: §V.D reports the specific drop and discusses its role

---

## Section 3 — Mini E2E test protocol (variant (c) gate)

### 3.1 Purpose

Detect propagation/architecture bugs in variant (c) implementation BEFORE 3 seed × 12.5 hr = 37.5 hr GPU commitment. Per Memory #12 lesson: isolated-module smoke tests miss config-propagation bugs; only end-to-end pipeline runs catch them.

### 3.2 Mini E2E configuration

- **Full LIBERO-Spatial (10 tasks)** — task subset filtering not supported in current LIBEROOTPDataset
- 2 training epochs
- Full pipeline: config YAML → OTPSoftModel construction → DataLoader → training loop → checkpoint save → eval
- **~30-40 min GPU per run** (revised from earlier 2.5 hr estimate based on Path B convention of 50 epoch = 12.5 hr)

**Rationale for full-data vs 1-task**: Adding task subset support requires modifying production DataLoader before GPU launch, which introduces bug risk. Full-data 2-epoch mini-run is **more representative** of production ablation conditions and only ~30 min cost. Total mini E2E GPU: ~60-80 min sequential = ~$8.

### 3.3 Two runs in sequence

**Run 1**: Variant (c) candidate (CFM head + deterministic decoder, N(0,I) source) — ~30-40 min  
**Run 2**: Path B baseline (deterministic head + CFM decoder, N(0,I) source) — ~30-40 min  

Sequential execution avoids parallel-OOM risk on single GPU. Total mini E2E budget: **~60-80 min GPU × $6/hr ≈ $6-8**.

### 3.4 Pass criteria (all 7 must pass for variant (c) to be included)

#### Criterion 1 — Config propagation via loss magnitude signature

**Pre-registered expected ranges**:
- Variant (c) step-1 loss expected: [0.5, 1.5]  
  Justification: L1 loss on standardized actions, expected near E[|N(0,1)|] · scale = ~0.6-1.0
- Path B baseline step-1 loss expected: [1.0, 3.0]  
  Justification: CFM velocity loss on normalized actions typical magnitude

**Pass condition**:
1. Variant (c) step-1 loss is in [0.5, 1.5]
2. Path B step-1 loss is in [1.0, 3.0]
3. Variant (c) step-1 loss is outside [1.0, 3.0] (verifies different from Path B baseline)

#### Criterion 2 — Loss descent

- Variant (c) loss decreases ≥ 50% over first 100 training steps
- Loss monotonically decreasing over last 30 steps of 100-step window (within 10% noise tolerance)

#### Criterion 3 — Path B baseline parallel comparison

Both runs (Run 1 + Run 2) must complete training successfully (no crash, finite loss throughout). Used as anchor for Criterion 5.

#### Criterion 4 — Checkpoint save validity

- At least 1 checkpoint saved during 2-epoch run
- Checkpoint file size < 500 MB (verifies trainable-only filter, NOT full 7.6B backbone dump)

#### Criterion 5 — Sample quality relative comparison

At end of 2-epoch training:
- Compute per-dim L1/std ratio on training task subset (~50 demonstrations × 8 timesteps)
- Variant (c) per-dim L1/std must be ≥ 20% better (lower) than Path B parallel baseline on ≥ 4/7 action dimensions

#### Criterion 6 — Numerical stability

- All gradient norms finite throughout training (no Inf)
- All loss values finite throughout (no NaN)

#### Criterion 7 — Determinism

- Two forwards of variant (c) model with same input + same seed produce bit-exact output
- Path B baseline forward also bit-exact under same conditions

### 3.5 Decision gate

**ALL 7 criteria pass** → variant (c) included → 21 retrains full ablation launch  
**ANY criterion fails** → variant (c) dropped → 18 retrains fallback launch (cells A, B, D, E, F, G only)  
**No debug-retry loop** (strict-mode rejection of sunk-cost recovery)

### 3.6 If variant (c) dropped, paper handling

§VI Future Work adds:
> "Direct ablation of decoder family choice (CFM head paired with deterministic regression decoder) was attempted in this work via the §IV.B architectural axis. Pre-flight end-to-end testing on a single-task subset revealed [specific failure mode observed]; formal multi-seed evaluation of this variant is deferred to future work. The §V.C.5 OFT contrast and Theorem 3' Remark 2 provide indirect evidence for the decoder family axis hypothesis."

---

## Section 4 — Implementation pre-commit checklist

Before mini E2E GPU launch:

- [ ] Variant (c) code implementation complete in `language_agnostic_decoder.py`
- [ ] Config flag `use_deterministic_decoder` threaded through `otp_soft_model.py`
- [ ] All 7 mini E2E pass criteria explicitly tested-for in test script
- [ ] Pre-registration doc (this file) committed to repo with timestamp
- [ ] Backup files exist (`.bak_*`) for both modified source files
- [ ] All 21 config files generated and inspected

Before full ablation GPU launch:

- [ ] Mini E2E decision gate evaluated, outcome recorded
- [ ] Full ablation cells locked (21 or 18)
- [ ] Disk space verified for 21 × 200 MB checkpoints = ~4.2 GB
- [ ] HF cache verified persistent at /root/autodl-tmp/hf_cache
- [ ] §V.D analysis script verified runs on a 1-cell test data subset
- [ ] All config files dry-run loadable (no YAML syntax errors)

---

## Section 5 — Acceptance-rate-protective decisions documented

This pre-reg locks the following acceptance-rate-protective decisions:

1. **Max config 7 cells × 3 seeds** (not minimal 5-cell) — per "中稿率 > 经费" priority
2. **Variant (c) included by default** — addresses §IV.A "decoder family axis" coherence with §V.C.5
3. **Mini E2E test (not isolated smoke)** — per user lesson: "smoke test 通过, 训练失败浪费一天" pattern
4. **Sequential mini E2E (not parallel)** — avoids OOM risk on single GPU
5. **Pre-registered loss magnitude ranges** — removes post-hoc magnitude eyeballing
6. **Path B baseline parallel comparison** — anchors sample quality relative criterion
7. **No debug-retry on mini E2E failure** — strict-mode sunk-cost discipline
8. **LIBERO-Goal M3 cross-benchmark add** — strengthens §VI.5 single-benchmark scope caveat

---

## Section 6 — Authority signatures (pre-commit timestamp)

- Strategic confirmation: Yue ("一切按照推荐的来", "中稿率 > 一切")
- Implementation by: Claude
- Audited by: Reviewer (multi-round, including the Section 3 mini E2E protocol catch)

Committed via: `git commit` with this file content frozen.
