# Phase 0 Pre-Registration (V6)

**Date frozen**: 2026-05-11
**Author**: Yue (independent researcher)
**Status**: Frozen pre-Phase-0. Hard thresholds locked; branch decisions soft (boundary cases documented).
**Supersedes**: V5 draft (uncommitted). All V5 changes absorbed into V6 with 9 revisions per second-round review.

This document is a pre-registration of the Phase 0 diagnostic battery for OTP-VLA's V6 design.
Statistical thresholds and pass/fail criteria are frozen here BEFORE running any Test. Any
modification post-test requires opening a V7 document with explicit revision rationale.

---

## Revision log (V5 → V6)

- **M1**: §2.3 MANOVA computed via `statsmodels` Rao-F approximation; hand-derived Bartlett χ² removed
- **M2**: §2.4 verdict uses permutation p only; R² reported as effect size but not in verdict logic
- **M3**: §2.6 capacity test measures action-level F̄ᵃ (not z-level F̄ᶻ), reflecting backbone task-discriminability rather than ill-defined z-representation goal
- **S1**: §2.5 Test 3 protocol details locked — task subset {0, 5}, warm-start from V3 ckpt, F summary as (median, max, 95th-percentile) triple
- **S2**: §2.5 contrastive loss uses batch-level negatives (~14 negatives at batch 16) instead of task-level prototypes
- **S3**: §4 timeline includes engineering overhead (ckpt-save bug fix, module implementation, single-task overfitting sanity)
- **A1**: §2.2 reports η² effect size alongside F and p
- **A2**: §2.3 second-layer per-dim test reported as exploratory diagnostic, no FDR (6 rotation dims share quaternion dependence; m=6 makes BH≈Bonferroni)
- **A3**: §2.7 adds Remark 1 on semantic invariance via bottleneck (C1+C3 give compositional property)

---

## §1 Problem statement

V3 completes 30-epoch training, sim eval on LIBERO-Spatial gives 0/50 SR. Loss converges
(head 1.93→0.27, decoder 0.39→0.14). Two-layer single-seed probes show decoder 4/7
action dims collapse to training marginal mean (diff < 0.03) and head 4/5 task have
near-identical z[obj₀, t=0, :3] (cross-task std < cross-seed std on dims 0, 3, 5). The
head probe cannot distinguish three structurally different failure modes (true collapse /
partial collapse / motion-prior learning), requiring Phase 0 diagnostic battery.

---

## §2 Theoretical support

### §2.1 Failure mode formalization

Conditional flow matching loss:

L_CFM(θ) = E_{t, x_0, x_1, c}[ ||v_θ(t, x_t, c) - (x_1 - x_0)||² ]

Dong et al. 2025 Theorem 1: when p(x₁|c) is highly overlapping across c, there exists a
family of solutions v_θ⋆ such that v_θ⋆(t, x, c₁) ≈ v_θ⋆(t, x, c₂), and this family is
an attractor basin in the loss landscape. Model learns marginal p(x₁) and drops c.

**Stacked CFM cascade collapse hypothesis**: head outputs z = v_θ^head(·, c^head),
decoder outputs a = v_θ^dec(·, c^dec) where z ∈ c^dec. Head Dong-collapse ⇒ z degenerates
to constant ⇒ decoder effective conditioning dimension drops ⇒ decoder enters its own
Dong-collapse. Two-layer independent collapse compounding is not yet documented in the
literature — the potential novelty of this work.

### §2.2 Head identifiability via standard ANOVA F-test

Per-dim one-way ANOVA, k=5 tasks, n=20 seeds, N=100:

F^(j) = MSB / MSW
      = [Σ_t n(z̄_t^(j) - z̄_··^(j))² / (k-1)] / [Σ_t Σ_s (z_{t,s}^(j) - z̄_t^(j))² / (N-k)]

Reference distribution F(4, 95):
- F_{0.05} = 2.469 (boundary for partial)
- F_{0.001} = 5.128 (boundary for strong identifiability)

**[A1] Additionally report η² as sample-size-invariant effect size**:

η²^(j) = SSB^(j) / (SSB^(j) + SSW^(j))

Cohen 1988 conventions: η²=0.01 small, η²=0.06 medium, η²=0.14 large. Collapse
hypothesis predicts η² ≈ 0.

**FROZEN per-dim verdict**:
- F^(j) < 2.469 (p > 0.05): true collapse
- 2.469 ≤ F^(j) < 5.128 (p ∈ (0.001, 0.05]): partial collapse
- F^(j) ≥ 5.128 (p ≤ 0.001): identifiable

### §2.3 Multi-dimensional comparison — two-layer scheme

**Layer 1 (global identifiability)**: Per-slice 6-dim MANOVA. z shape (N_obj=5, H=8, 6)
sliced into 40 6-dim slices.

**[M1] Computed via `statsmodels.multivariate.manova.MANOVA`**, reporting Wilks' Λ with
Rao-F approximation. All four MANOVA statistics (Pillai's trace, Wilks' Λ,
Hotelling-Lawley, Roy's largest root) preserved in supplementary; primary test is Wilks Λ.

40 slice p-values corrected by Benjamini-Hochberg FDR (q = 0.05).

**FROZEN global verdict**:
- ≥1 slice BH-significant: head has identifiability somewhere
- All slices non-significant: head global collapse

**Layer 2 (exploratory dim-level diagnostic)**: For slices NOT BH-significant, per-dim
univariate F-test localizes collapse-prone dimensions.

**[A2] No FDR at this layer** — framed as exploratory diagnostic. Rationale: (a) 6 dims
include rotation 3-tuple sharing quaternion dependence, violating FDR weak-dependence
assumption; (b) at m=6, BH ≈ Bonferroni. Raw F, raw p, η² reported per dim.

### §2.4 Spatial alignment test

Setup: K=10 LIBERO-Spatial tasks (doubled from V3's 5), 20 seeds per task, mean over
seeds gives μ_t^pos ∈ ℝ³. Target object world position p_t extracted from sim init.

Affine regression μ_t^pos = A p_t + b + ε. Report R² and 1000-shuffle permutation p
on R² statistic.

**[M2] FROZEN verdict uses permutation p ONLY**:

| permutation p | verdict |
|---|---|
| p < 0.01 | head spatial-aware (problem in decoder) |
| p ∈ [0.01, 0.05] | borderline — boundary case, joint judgement with §2.3 required, document rationale |
| p > 0.05 | head failed to learn spatial structure |

R² reported alongside as effect size, not part of verdict logic. Rationale for change:
permutation p measures "signal exists significantly above chance", R² measures "fraction
of variance explained" — joining them with "and" produces logically incoherent verdicts
in small-n regimes where permutation null R² already has wide distribution.

### §2.5 Cocos × Contrastive 2×2 ablation on head

Standard Cocos source: x_0 ~ N(α F_φ(c), β² I). Dong original validation is on low-dim
conditioning + geometrically interpretable target. Head target z ∈ ℝ²⁴⁰ is abstract
latent trajectory — Cocos behavior in this regime is the open question.

**FROZEN 2×2 ablation design**:

| Condition | Source | Loss |
|---|---|---|
| Baseline | N(0, I) | L_CFM only |
| +Cocos | N(α F_φ(c), β² I) | L_CFM only |
| +Contrastive | N(0, I) | L_CFM + λ L_contrast |
| Both | N(α F_φ(c), β² I) | L_CFM + λ L_contrast |

**[S2] Batch-level InfoNCE** (not task-prototype-level):

L_contrast = -Σ_i log [ exp(sim(z_i, e_{t(i)}) / τ) / Σ_j exp(sim(z_i, e_{t(j)}) / τ) ]

where i indexes batch samples, t(i) is sample i's task. At batch=16, ~14 negatives per
positive — 7× stronger signal than task-prototype InfoNCE at k=2.

Hyperparameters: α=1.0, β=1.0, τ=0.07, λ=1.0. F_φ = MLP(c_dim → 512 → 256 → latent_dim),
GELU activations. c = concat(backbone_hidden_CLS, language_token_CLS).

**[S1] Protocol details (FROZEN)**:
- 2-task subset: task_id ∈ {0, 5} (cross-group selection: tasks 0-4 vs 5-9 are two spatial
  layout groups in LIBERO-Spatial; cross-group pair maximizes spatial separation)
- Warm-start: load head state from V3 checkpoint (path in .paper_ready_ckpt), NOT random
  init. Rationale: we test "can Cocos+Contrastive recover an already-collapsed V3 head",
  not "what works from scratch" — the former is deployment-relevant
- 500 demos per task, 1000 train steps, batch=16, lr=1e-4
- Post-train F̄ measurement: collect z at 20 seeds × 2 tasks, compute univariate F per
  (obj, t, dim) cell (240 cells total)
- **F summary as triple**: (median, max, 95th-percentile) across 240 cells

**FROZEN Tier 1 trigger**: any non-baseline condition satisfies BOTH:
- median F̄ improvement over baseline ≥ 5×
- max F̄ ≥ 5.128 (i.e., at least one cell shows strong identifiability)

### §2.6 Capacity test (action-level)

**[M3] Target is action-level F̄ᵃ, not z-level F̄ᶻ**. Rationale: in z→a supervised
regression, a wide z→a projection can compensate for arbitrarily task-agnostic z, making
F̄ᶻ uninformative about backbone capacity. F̄ᵃ directly tests whether the backbone
hidden state carries task-discriminative signal that can reach the action space.

Setup:
- Width-doubled OTPHead (same depth, no flow matching wrapper, deterministic forward)
- Linear projection z → a ∈ ℝ⁷
- L1 loss on demo first-step actions, 1000 steps, 2-task subset, batch=16, lr=1e-4
- Post-train F̄ᵃ measurement: 20 demos per task as "samples" (deterministic, no seed noise)
- Per-dim F across 7 action dims; report (mean, max, η²-vector)

**FROZEN verdict**:
- F̄ᵃ ≥ 5.128: backbone carries sufficient task-discriminative signal, CFM head collapse
  is a loss-landscape problem → Path B (deterministic head + decoder Cocos) is correct
- F̄ᵃ ∈ [2.469, 5.128): marginal, document decision with rationale
- F̄ᵃ < 2.469: even in cleanest supervised setting, action is not task-discriminable from
  this backbone projection → backbone hidden state extraction needs reassessment
  (different layer, task-token attention pool, etc.); Path B alone insufficient

### §2.7 Theorem 2 preservation under V6 design

C1 (bottleneck necessity, z task-determined): Cocos + contrastive strengthen this.
C2 (non-visual-determinacy, H(z|o) > 0): head remains stochastic, automatic.
C3 (decoder language-agnosticism): formal Lemma 1.

**Lemma 1 (C3-preservation under contrastive head training).** Let φ_θ be a decoder with
computational graph satisfying ℓ ∉ inputs(φ_θ). Let p_θ(z|o,ℓ) be a head trained with
auxiliary loss L_aux(z, emb(ℓ)) acting only on head parameters. Then C3 holds for the
composite policy π_θ(a|o,ℓ) = ∫ φ_θ(z, w(o)) p_θ(z|o,ℓ) dz.

*Proof sketch.* C3 is defined architecturally as ℓ ∉ inputs(φ_θ). L_aux affects
∇_θ L_aux which updates head parameters via backprop; the forward computation of φ_θ
remains invariant under any choice of head training signal. Therefore φ_θ remains
language-agnostic at inference. ∎

**[A3] Remark 1 (Semantic invariance via bottleneck).** C3 combined with C1 yields a
stronger property: φ_θ's output depends on ℓ only through the task-determined latent z,
hence is invariant to lexical paraphrasing of ℓ that preserves task identity. Formally,
for paraphrase pairs (ℓ, ℓ') with the same underlying task τ, the data-generating
distribution gives z(o, ℓ) ≡ᵈ z(o, ℓ') (up to head stochasticity), and consequently
π_θ(a|o,ℓ) ≡ᵈ π_θ(a|o,ℓ'). This compositional generalization property distinguishes
OTP-Soft from OFT-style architectures that take ℓ as a direct decoder input, and is
empirically validated via the unseen-phrasing ablation in §VI.

This elevates Lemma 1 from "C3 is trivially preserved by architecture" to "C3+C1 give a
non-trivial compositional generalization property" — addresses the sharpest C3 attack a
reviewer can pose.

---

## §3 Tier narrative (conditional contribution claims)

### Tier 1 (strong)
Trigger: Test 3 Tier 1 condition met AND final Phase 2 SR ≥ 75%.

Main claim: *Two-layer CFM collapse as a generic failure mode of stacked flow matching,
fixed by condition-faithful source distributions and task-discriminative auxiliary
signal at each layer.*

### Tier 2 (medium)
Trigger: Test 3 shows F̄ improvement 2×–5× OR final SR ∈ [40%, 75%).

Main claim: *Theorem 2 with C1/C2/C3 as a formal framework for modality-faithful VLA,
demonstrated on stacked flow matching architectures.*

### Tier 3 (conservative)
Trigger: Test 3 fails Tier 1 + Test 4 shows capacity sufficient + Path B final SR ≥ 70%.

Main claim: *Object-conditioned trajectory bottleneck as an architectural primitive for
language-faithful VLA, with deterministic trajectory representation and stochastic action
decoder.*

### Cross-tier core (all tiers)
Theorem 2 + Lemma 1 + Remark 1 + ANOVA diagnostic framework + unseen instruction
phrasing ablation. The latter is the key OTP-Soft vs OFT differentiation invisible to
OFT's reported 97% SR — addresses reviewer's "why not just use OFT" pre-emptively.

---

## §4 Implementation timeline (with engineering overhead, [S3])

| Day | Milestone |
|---|---|
| Day 1 | Test 0 (sanity) + Test 1 (head identifiability) + Test 2 (spatial alignment) |
| Day 2 | Test 3 (2×2 ablation, 4 toy training runs) |
| Day 3 | Test 4 (capacity, if triggered) + Path decision committed |
| **Day 4 AM** | **Fix V3 ckpt-save bug: filter requires_grad=True before save; ckpt drops 16GB → ~185MB** |
| **Day 4 PM** | **Implement Cocos source module + cross-attention conditioning (if Path A) OR deterministic head (if Path B)** |
| **Day 5 AM** | **Sanity check: 50 demos × 200 steps single-task overfitting; verify new modules behave** |
| Day 5 PM – Day 8 | Full retrain round 1 (~9-10h wall time, monitoring) |
| Day 9 | Diagnostic + hyperparameter adjustment (α, β, λ, τ scan) |
| Day 10–13 | Retrain round 2 (if needed) + supplementary cross-attention |
| Day 14–17 | Sim eval + final diagnostic battery on production checkpoint |
| Day 18–27 | Paper writing |
| Day 28–30 | Buffer + RA-L submission (late June 2026) |

PhD application this cycle: under review status, no accepted paper. Trade-off documented
and accepted per Q1 timeline choice on 2026-05-11.

---

## §5 Pre-Registration declaration

**FROZEN** (no post-test modification without V7 document):

- §2.2 critical values F_{0.05}=2.469, F_{0.001}=5.128
- §2.3 BH-FDR q=0.05 across 40 slices; no FDR in Layer 2
- §2.4 permutation iterations=1000, α=0.05 threshold, p-only verdict
- §2.5 ablation 4-condition design; task subset {0, 5}; warm-start from V3 ckpt;
  hyperparameters α=1.0 β=1.0 τ=0.07 λ=1.0; F summary triple (median, max, 95-pctile);
  Tier 1 trigger = (median improvement ≥ 5× AND max F̄ ≥ 5.128)
- §2.6 action-level F̄ᵃ critical values same as §2.2; verdict on backbone signal sufficiency
- §3 tier triggers

**Soft** (boundary cases discussable, rationale required):
- Branch selection in decision tree
- Hyperparameters within chosen Path
- Number of retrain rounds
- Final paper §V structure

**Mandatory documented-rationale ranges**:
- Test 1: BH-adjusted p ∈ [0.04, 0.06] for any slice
- Test 2: permutation p ∈ [0.04, 0.06]
- Test 3: median improvement ∈ [4×, 6×] OR max F̄ ∈ [4.5, 5.5]
- Test 4: F̄ᵃ ∈ [4.5, 5.5]

Each must include: (a) decision taken, (b) alternative considered, (c) cost-benefit,
(d) confidence assessment. Written to notes/phase0_results.md.

---

## §6 Branch decision tree

```
Test 1 + Test 2 jointly
├─ Test 1: ≥1 slice BH-sig AND Test 2: p < 0.01
│  → head spatial-aware, problem in decoder
│  → Path A' (decoder-only Cocos + cross-attention retrain)
│
├─ Test 1: all slices non-sig OR Test 2: p ≥ 0.05
│  → Test 3 (2×2 ablation)
│  │
│  ├─ Tier 1 triggered (median ≥ 5× AND max F̄ ≥ 5.128)
│  │  → full Path A retrain (both layers Cocos + contrastive at head + cross-attention at decoder)
│  │
│  └─ Tier 1 NOT triggered
│     → Test 4 (capacity, action-level F̄ᵃ)
│     │
│     ├─ F̄ᵃ ≥ 5.128: Path B (deterministic head + decoder Cocos)
│     ├─ F̄ᵃ ∈ [2.469, 5.128): document rationale, lean Path B with width scaling
│     └─ F̄ᵃ < 2.469: backbone projection insufficient; reassess hidden state extraction
│                      or escalate to Path α (workshop venue)
│
└─ Borderline (Test 1 OR Test 2 in [0.01, 0.05]) → joint judgement, document
```

---

End of V6 pre-registration. This document is the artifact-of-record for the Phase 0
methodology. Commit immediately before running any Test.
