# Phase 0 Diagnostic Results — Historical Record

**Status**: Frozen 2026-05-11. Diagnostic data immutable. Forward-looking decisions (paper framing, Theorem 3 formulation, Path B implementation choices) documented in V7 pre-registration (pending OFT unseen-phrasing ablation).

**Project**: OTP-VLA, IEEE Robotics and Automation Letters (rolling submission).

**Author**: Yue (independent researcher).

---

## 1. Diagnostic Data

All raw outputs in `results/phase0/*.json`. Scripts in `scripts/diagnostic/`.

### 1.1 Test 0: Sanity Check (committed 2026-05-10)

- **Protocol**: Verify dataset split (`train_end_demo=45`), shape consistency, seeding determinism.
- **Result**: PASS. Diff-seed diff = 4.52 on head z output, indicating CFM noise amplitude is non-trivial.

### 1.2 Test 1: Head Identifiability (2026-05-10)

- **Protocol**: 5 LIBERO-Spatial tasks (id 0-4) × 20 seeds × 5 objects × 8 horizon × 6 dims. Per-slice MANOVA (Wilks' Λ Rao-F approximation via statsmodels). BH-FDR correction at q = 0.05 across 40 slices.
- **Raw**: 18/40 BH-significant slices. Top 5 by p-value:
  - (obj=0, t=0): p = 1.7 × 10⁻²³
  - (obj=0, t=1): p = 4.2 × 10⁻²³
  - (obj=0, t=2): p = 8.1 × 10⁻²²
  - (obj=0, t=3): p = 2.1 × 10⁻²²
  - (obj=0, t=4): p = 5.6 × 10⁻¹⁸
- **Non-significant slices**: 132/132 cells in collapse mode (η² median 0.004).
- **Output**: `results/phase0/test1_head_identifiability.json`, `results/phase0/test1_Z.npy`.

### 1.3 Test 2: Spatial Alignment (2026-05-10)

- **Protocol**: Multivariate linear regression of head μ[obj=0, t=0..4].mean (3-dim slice) onto target object world position P (3-dim) across K=10 tasks. R², permutation p over 1000 shuffles.
- **Raw**:
  - R² = 0.524, permutation p (one-sided) = 0.164.
  - Per-dim Pearson: x r = −0.458 p = 0.183, y r = +0.063 p = 0.863, z r = −0.045 p = 0.902.
- **Verdict**: `head_no_spatial_structure` (no significant spatial encoding).
- **Output**: `results/phase0/test2_spatial_alignment.json`.

### 1.4 Test 4: Capacity Test (2026-05-11)

- **Protocol**: V6.1 action-supervised regression. Width-doubled deterministic CapacityHead (`hidden_dim = 1024 × 2 = 2048`, 260.3M params) bypassing CFM. Frozen backbone, 1000-step L1 training on 2 tasks (task 0, task 5) × 45 demos. Per-action-dim F-test across K=2, N=20.
- **Raw**:
  - F̄ᵃ_mean = 11.30, F̄ᵃ_max = 26.90.
  - Per-dim F: [17.34, 6.33, 4.95, 22.34, 26.90, 0.25, 1.02].
  - Per-dim p: [1.7e-4, 1.6e-2, 3.2e-2, 3.1e-5, 7.4e-6, 6.2e-1, 3.2e-1].
  - Per-dim η²: [0.31, 0.14, 0.12, 0.37, 0.41, 0.006, 0.026].
  - Final training L1: 0.0274 (10× decrease from 0.27).
- **Critical values**: F_{0.05}(1, 38) = 4.10, F_{0.001}(1, 38) = 12.61.
- **Verdict**: `backbone_sufficient_path_B_correct` (F̄ᵃ ≥ both critical values).
- **Output**: `results/phase0/test4_capacity.json`.

### 1.5 Deep-Dive Q1-Q4 (2026-05-11)

- **Q1 — Leverage on Test 2**:
  - Per-dim Pearson r non-significant across all 3 dims (max |r| = 0.46).
  - Leave-one-out: drop task 7 → R² rises by 0.21 (task 7 is a leverage outlier); drop task 5 → +0.17; drop task 8 → −0.15.
- **Q2 — Geometry of head z (Mantel test)**:
  - PCA on 10 task-mean z slices: PC1 explains 92.1%, PC1 separates task 7 (+12.10) from others (-1.5 to -1.9). PC2 explains 6.7%, separates task 8 (+3.33).
  - 8/10 tasks (0-6, 9) cluster at PC1 ∈ [-1.79, -1.51], PC2 ∈ [-0.40, -0.16].
  - Mantel r(z, target world position) = +0.058, p = 0.376.
- **Q3 — Amplitude match**:
  - Head μ mean abs amplitude = 0.932; GT mean abs amplitude = 0.936. Ratio 0.995.
  - Per-task obj=0 L2 distance from gt: task 0-6, 9 mean ≈ 0.16-0.28; task 7 mean = 0.37, max = 3.80; task 8 mean = 0.43, max = 5.43.
- **Q4 — Per-dim amplitude**:
  - All 6 dims std_head / std_gt ∈ [0.5, 3.0]. No amplitude saturation.
- **Output**: `results/phase0/deep_dive.json`.

### 1.6 Condition Check (C1 + C2) (2026-05-11)

- **C1 — Visual pass-through gate**:
  - Mantel r(z, visual_CLS mean-pool) = +0.025, p = 0.39.
  - Mantel r(z, language_CLS mean-pool) = −0.19, p = 0.91.
  - language/visual r ratio = −7.6.
  - Verdict: `visual_pass_through_no_task_conditioning` (head not driven by either modality).
- **C2 — η² for 18 significant slices from Test 1**:
  - Multivariate η² (trace-based) median = 0.260, range [0.05, 0.96].
  - Top 5 slices: (obj=1, t=0) η² = 0.96, (obj=1, t=1) 0.74, (obj=4, t=0) 0.52, (obj=1, t=2) 0.41, (obj=4, t=1) 0.41.
  - Verdict: `noise_dominated_clustered_structure_unreliable` (median < 0.30; significant slices concentrate on non-target objects).
- **Output**: `results/phase0/condition_check.json`.

### 1.7 Gate 2: Robust Encoding Check (2026-05-11)

- **Protocol**: D_language vs D_visual under 4 encoding methods (mean-pool, CLS, last-token, full-seq) × 2 normalizations (raw, L2-normalized). 7 valid combinations (CLS/last raw and L2 give D = 0 due to constant BOS/EOS tokens).
- **Raw**:
  - Mean-pool raw: D_lang = 0.097, D_vis = 3.75, ratio 0.026.
  - **Mean-pool L2-normalized: D_lang = 0.42, D_vis = 0.26, ratio 1.63**.
  - CLS raw/L2 and last raw/L2: D_lang = 0 (trivially, BOS/EOS constant); ratios undefined/zero.
  - Full-seq raw: D_lang = 1.41, D_vis = 35.9, ratio 0.039.
- **Token-level L2 norms**: visual = 29.0, text = 1.03 (28× imbalance).
- **Verdict**: Original "D_lang << D_vis" framing is a norm artifact. Under L2-normalization, language is more dispersed than visual on the unit sphere (ratio 1.63). The original observation reflects modality magnitude imbalance, not language signal absence.
- **Output**: `results/phase0/gate2_robust_encoding.json`.

### 1.8 Gate 1: OFT Backbone Diagnostic (2026-05-11)

- **Protocol**: Extract h_OFT = backbone `hidden_states[:, -1, :]` (last text token after LLM forward over visual + text sequence). 10 tasks × 5 demos, mean-pooled per task. Mantel test against visual_CLS, language_CLS, proprioception. 1000-shuffle null.
- **Raw**:
  - h_OFT L2 norm mean: 59.4 (compared to visual_CLS 14.3, language_CLS 0.23, proprio 1.27).
  - **Mantel r(h_OFT, language) = +0.7423, p < 0.001.**
  - Mantel r(h_OFT, visual) = +0.3674, p = 0.032.
  - Mantel r(h_OFT, proprio) = −0.1664, p = 0.253.
- **Verdict**: `language_driven_OFT_genuinely_conditions_on_instruction`.
- **Output**: `results/phase0/gate1_oft_diagnostic.json`.

### 1.9 SC1-SC3: Sanity Checks on Gate 1 (2026-05-11)

- **SC1 — Token position**:
  - Position [-1] index = 289 (T_tok = 290, N_text = 34, N_visual = 256).
  - Last text token id = 29901 (`":"` — template-end colon "Out:").
  - This token has attended over entire (visual + text) sequence via causal LLM.
  - Verdict: PASS, valid representation.
- **SC2 — Embedding consistency**:
  - language_CLS run-to-run max element diff = 0.00 (byte-identical).
  - C1 D_language mean = 0.167; Gate 1 D_language mean = 0.097. Difference is sampling effect (C1 used different sample indexing).
  - Verdict: PASS, C1 and Gate 1 use byte-identical computation path.
- **SC3 — Replicate Gate 1 with K=10, n_demos=10**:
  - r(h_OFT, visual) = +0.3874 (vs 0.3674), Δ = +0.020.
  - r(h_OFT, language) = **+0.7575** (vs 0.7423), Δ = +0.015.
  - Verdict: PASS, stable in 0.6-0.8 range, finding replicates.
- **Output**: `results/phase0/gate1_sanity_and_m3.json`.

### 1.10 M3: Demonstration-Trajectory ⊥ Language (2026-05-11)

- **Protocol**: For each of 10 tasks × 10 demos, extract gt_trajectory ∈ ℝ^(5 × 8 × 6). Flatten to 240-dim. Mantel test against language_CLS. 1000-shuffle null.
- **Raw**:
  - Per-task trajectory mean L2 ∈ [20.38, 20.87] (CV < 1%, near-uniform across tasks).
  - **Mantel r(demo_traj, language) = −0.270, p (one-sided positive) = 0.964**.
  - Null distribution: mean = +0.003, std = 0.191.
- **Verdict**: `M3_CONFIRMED` (demos show negligible-to-anticorrelated language conditioning).
- **Output**: `results/phase0/gate1_sanity_and_m3.json`.

### 1.11 M3 Robust Check across Representations (2026-05-11)

- **Protocol**: Repeat M3 Mantel test under 3 trajectory representations:
  - Flat (240-dim, original)
  - Endpoint (30-dim, last horizon step only, 5 objects × 6 dims)
  - Waypoint subsample (90-dim, t ∈ {0, 3, 7}, 5 objects × 3 waypoints × 6 dims)
- **Raw**:
  - **Flat**: r = −0.270, p_two = 0.156, null mean = +0.003, null std = 0.191. Δr from r_h = 0.76 = **+1.027**.
  - **Endpoint**: r = −0.196, p_two = 0.190, null mean = +0.009, null std = 0.212. Δr = **+0.953**.
  - **Waypoint**: r = −0.269, p_two = 0.155, null mean = +0.004, null std = 0.191. Δr = **+1.026**.
- **Verdict**: `M3_ROBUST` (all 3 representations: r ∈ [−0.27, −0.20], all gaps from r_h ≥ 0.95, no representation gives r >> 0).
- **Cross-representation consistency** is stronger evidence than individual Mantel significance at K = 10 (low power on individual tests).
- **Output**: `results/phase0/m3_robust_check.json`.

---

## 2. Decision Log

Chronological record of framing decisions during Phase 0. Each entry: date, event, decision, rationale.

| Date | Event | Decision |
|---|---|---|
| 2026-05-09 22:50 | V3 retrain at `ckpt_step0099000.pt`. Loss converged cleanly (head 1.93→0.27, decoder 0.39→0.14) but sim eval = 0/50 SR on LIBERO-Spatial. | Initiate Phase 0 diagnostic battery to localize failure mode. |
| 2026-05-10 morning | V6 pre-registration committed (`notes/2026-05-11_phase0_preregistration_v6.md`). Frozen items: critical F values, Mantel protocol, BH-FDR settings, 4-condition Test 3 design (Cocos × Contrastive ablation). | Run Tests 0/1/2/4 before any framing. |
| 2026-05-10 evening | Tests 0/1/2/4 complete. T1 18/40 sig, T2 R²=0.52 p=0.16, T4 F̄ᵃ=11.30. | Hypothesis: Tier 1 narrative — head identifiable but spatially uncalibrated. |
| 2026-05-11 morning | Deep-dive Q1-Q4. PCA reveals "8-into-1 + 2 outliers" structure. Mantel r(z, target_P) = 0.058. | Hypothesis: "selective clustered collapse on visually-similar tasks". Plan to make this the paper's main finding. |
| 2026-05-11 noon | C1+C2 condition check. r(z, language) = −0.19, r(z, visual) = +0.025. η² median 0.26, top η² on non-target objects. | Clustered-collapse hypothesis falsified: head not condition on either modality, signal concentrates on wrong object slots. |
| 2026-05-11 afternoon | Literature search. Contrastive Flow Matching (ICCV 2025) covers "condition separation failure in conditional FM"; Conditioning Matters (NeurIPS 2025) covers marginal-action collapse. Narrowed framing to "stacked CFM in VLA" specifically. | Tier 1 upgrade framing weakened. Considered fallback to Tier 3. |
| 2026-05-11 late afternoon | Gate 2 robust encoding check. raw D_lang = 0.097 vs D_vis = 3.75 ratio 0.026; L2-normalized D_lang = 0.42 vs D_vis = 0.26 ratio 1.63 (reversal). 28× modality magnitude imbalance. | Original "language signal absent" framing rejected. Reframed as modality-norm-imbalance not signal absence. |
| 2026-05-11 evening | Gate 1 OFT backbone diagnostic. **r(h_OFT, language) = +0.74, p < 0.001**. r(h_OFT, visual) = +0.37, r(h_OFT, proprio) = −0.17. | Backbone is language-faithful. Failure happens between backbone and OTP-Soft head. Framing pivoted to compositional-faithfulness. |
| 2026-05-11 evening | SC1-3 sanity check on Gate 1. All pass. SC3 replicates r(h_OFT, language) = +0.76 at n_demos=10. | Gate 1 finding is not measurement artifact. |
| 2026-05-11 evening | M3 demo trajectory vs language. Mantel r = −0.270, p (one-sided +) = 0.96. | Demonstration-level conditioning failure hypothesis surfaced. |
| 2026-05-11 night | M3 robust check across 3 trajectory representations. All 3 give r ∈ [−0.27, −0.20], Δr from r_h ≥ 0.95. | M3 robust. Theorem 3 (supervision-induced faithfulness collapse) candidate identified as paper sharpest theoretical contribution. |
| **PENDING 2026-05-12** | OFT unseen-phrasing ablation. | Distinguishes explanation A (OFT visual-grounded) vs B (architectural language path) vs C (language-as-scene-query). Resolves §V framing direction. |
| **POST-ABLATION** | V7 pre-registration commit. | Theorem 2 reformulation + Theorem 3 + Corollary + Lemma 1 + framing direction. |

---

## 3. Outstanding Questions

### 3.1 OFT Unseen-Phrasing Ablation (Pre-V7-commit)

**Test design**: 5 paraphrased instruction sets (semantic-preserving) + 1 identity control set. 10 LIBERO-Spatial tasks. 50 episodes per (task, phrasing) cell. Per-task SR + 3-stage SR (reaching / grasping / placement).

**Resolution outcomes**:
- **Explanation A** (OFT SR保持 in unseen phrasing): OFT relies on visual/scene grounding. M3 thesis strengthened to "LIBERO-Spatial benchmark inadequate for testing language conditioning". Paper §V.B writes contrastive (r_h, r_demo, r_z) chain + Theorem 3 instantiation. Target title direction: "Language Signal Loss in Pretraining-to-Demonstration Distillation".
- **Explanation B** (OFT SR drops uniformly): OFT has architectural language path independent of trajectory supervision. M3 framing weakened to "head-architecture failure under language-orthogonal supervision". Paper §V.B narrower scope. Title direction: "Stacked CFM Heads Fail to Preserve Backbone Language Signal".
- **Explanation C** (OFT SR partial drop, task-dependent): Language-as-scene-query mechanism. M3 + Theorem 3 still hold, but §V.B requires nuance — language affects via visual retrieval pathway. Title direction: "Quantifying Modality Contributions in VLA Policies".

**This question must resolve before V7 commit and Path B retrain framing.**

### 3.2 Theorem 3 Loss-Specification (V7 Writing-Phase)

**Question**: Theorem 3 statement covers "per-sample distribution-matching loss" generally. Different loss types (L2, L1, CFM, ShortcutFM) require either unified or specialized proof statements. V7 should commit to one of:
- Single general theorem covering all loss types (proof becomes more abstract)
- Multiple specialized variants (paper appendix carries 3-4 proofs)

### 3.3 Compositional Faithfulness Remark Sharpening (V7 Writing-Phase)

**Question**: Whether to retain the Remark on compositional faithfulness as a DPI corollary. Reviewer feedback: trivial, weakens main thesis. Two options:
- Remove entirely, replace with single sentence in §III intro
- Retain as Observation labeling the empirical instantiation as separate from Theorem 3

---

## 4. References

- **V6 pre-registration** (frozen 2026-05-10): `notes/2026-05-11_phase0_preregistration_v6.md`
- **V7 pre-registration** (pending OFT ablation, target 2026-05-12): `notes/2026-05-12_phase0_preregistration_v7.md`
- **Phase 0 raw outputs**: `results/phase0/*.json`, `results/phase0/test1_Z.npy`, `results/phase0/gate1_*.npy`
- **Phase 0 logs**: `logs/test*_*.log`, `logs/gate*_*.log`, `logs/sanity_m3_*.log`, `logs/m3_robust_*.log`
- **Diagnostic scripts**: `scripts/diagnostic/06_test0_sanity.py` through `scripts/diagnostic/16_m3_robust_check.py`

---

*Frozen 2026-05-11. Subsequent updates only via amendment with timestamp.*
