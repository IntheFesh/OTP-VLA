# OTP-VLA Phase 0 Pre-Registration V7

**Status**: V7. Commit date: 2026-05-11. Supersedes V6 (`notes/2026-05-11_phase0_preregistration_v6.md`).

**Author**: Yue, IEEE Robotics and Automation Letters (rolling submission).

**Purpose**: Pre-register forward-looking decisions following completion of the Phase 0 diagnostic battery (V6, frozen 2026-05-11). All decision rules below are committed *before* OFT baseline implementation and ablation results are observed.

---

## §I. Background: Phase 0 Findings (Reference)

All diagnostic data finalized in `notes/phase0_results.md` (committed 2026-05-11).

Three-layer empirical chain established:

1. **Backbone faithful**: $r(h_{\text{OFT}}, \text{language}) = +0.76, p < 0.001$ (Gate 1, replicated SC3)
2. **Demonstration language-orthogonal**: $r(\text{demo\_traj}, \text{language}) \in [-0.27, -0.20]$ across 3 trajectory representations (M3 robust)
3. **Policy unfaithful**: $r(z_{\text{OTP}}, \text{language}) = -0.19$, η² concentrated on non-target object slots (C1, C2)

Capacity ruled out as cause: $\bar{F}^a = 11.30$ in Test 4.

---

## §II. Theoretical Framework (Frozen)

### §II.A. Theorem 2 reformulated

Let $\pi_\theta = f_{\text{dec}} \circ f_{\text{head}} \circ f_{\text{back}}$ be a stacked VLA policy mapping observation $o$ and instruction $\ell$ to action $a$, with intermediate bottleneck $z = f_{\text{head}}(f_{\text{back}}(o, \ell))$. Let $\tau = \psi(\ell)$ denote task identity under semantic paraphrase equivalence, where

$$\ell \sim \ell' \iff p(a \mid o, \ell) = p(a \mid o, \ell') \text{ on the demonstration distribution.}$$

$\pi_\theta$ is **modality-faithful** if all three conditions hold:

- **C1 (bottleneck necessity)**: $z$ is task-determined, i.e., the decoder cannot ignore $z$ and still produce $a$.
- **C2 (task-information preservation)**: $I(z; \tau \mid o) > 0$.
- **C3 (decoder language-agnosticism)**: $\phi_\theta = f_{\text{dec}}$ does not take $\ell$ as input.

**Note on C2 reformulation**: V6 used $H(z \mid o) > 0$, which only requires entropy (satisfiable by noise). V7's $I(z; \tau \mid o) > 0$ requires $z$ to carry task-discriminative information beyond observation — a sharper condition that is empirically testable.

### §II.B. Lemma 1 (C3-preservation under contrastive training)

[Frozen from V5, unchanged] If $f_{\text{dec}}$ does not take $\ell$ as input during training and contrastive auxiliary signal is applied only on bottleneck representations $z$, then C3 is preserved.

### §II.C. Theorem 3 (Supervision-induced faithfulness collapse) — NEW in V7

Let $\pi_\theta$ be trained on $\mathcal{D} = \{(o_i, \ell_i, a_i)\}$ by minimizing $\mathbb{E}_{\mathcal{D}}[\mathcal{L}(\pi_\theta(o, \ell), a)]$, where $\mathcal{L}$ is a per-sample distribution-matching objective (L2, L1, conditional flow matching, or any other proper scoring rule). If demonstrations satisfy

$$I(a; \tau \mid o) = 0,$$

then the optimal policy $\pi^* = \arg\min \mathbb{E}_\mathcal{D}[\mathcal{L}]$ satisfies

$$I(\pi^*; \tau \mid o) = 0,$$

regardless of backbone capacity to represent $\tau$.

**Proof sketch.** $a \perp \tau \mid o$ implies $p(a \mid o, \tau) = p(a \mid o)$ pointwise. For distribution-matching $\mathcal{L}$, optimal $\pi^*(\cdot \mid o, \tau) = p(a \mid o, \tau) = p(a \mid o)$. Hence $\pi^*$ is conditionally independent of $\tau$ given $o$, giving $I(\pi^*; \tau \mid o) = 0$.

(Full proof in paper appendix, covering L2/L1/CFM/ShortcutFM specializations.)

### §II.D. Corollary (necessity of language-aware supervision)

Modality-faithfulness (Theorem 2) requires $I(z; \tau \mid o) > 0$. Under Theorem 3, this is unachievable via per-sample demonstration supervision alone when $I(a; \tau \mid o) = 0$. Therefore, modality-faithful VLA policies trained on language-orthogonal demonstrations require auxiliary signal beyond per-sample action loss.

### §II.E. Remark (compositional faithfulness)

By Data Processing Inequality, $I(a; \tau \mid o) \leq I(z; \tau \mid o) \leq I(h_{\text{OFT}}; \tau \mid o)$. A faithful backbone is therefore necessary but not sufficient for policy faithfulness; signal preservation must hold at every stage of the pipeline. This is a trivial DPI implication; the contribution is the empirical instantiation in §V.B.

---

## §III. OFT Unseen-Phrasing Ablation Protocol (Frozen)

### §III.A. Implementation requirements

1. **Predictor**: `OpenVLAOFTPredictor` class in `otp/eval/predictor.py` (currently stub, to be implemented). Uses HuggingFace `OpenVLAForActionPrediction.predict_action` interface on `moojink/openvla-7b-oft-finetuned-libero-spatial`.
2. **Determinism**: Bit-exact reproducibility verification before any SR measurement. Same `(episode_seed, step_counter, sample_idx)` must produce identical action chunks across runs.
3. **LIBERO env interface**: `libero.envs.OffScreenRenderEnv` per `scripts/04_validate_decoder_learning.py` Stage 2 sketch. Episode termination conditions and action chunking must match OTPSoftPredictor's V3 pipeline for fair comparison.
4. **3-stage SR parsing**: For each episode, record (i) `reach_success` (end-effector within 5cm of target object at any time), (ii) `grasp_success` (target object lifted ≥ 5cm above table), (iii) `place_success` (target object placed within target plate region, LIBERO standard).

### §III.B. Paraphrased instruction set (frozen below; no post-hoc modification)

10 LIBERO-Spatial tasks indexed 0-9. Each task has a base instruction of form `"pick up the black bowl {SPATIAL} and place it on the plate"`, where `{SPATIAL}` is task-specific.

| Task | SPATIAL phrase |
|---|---|
| 0 | between the plate and the ramekin |
| 1 | from table center |
| 2 | in the top drawer of the wooden cabinet |
| 3 | next to the cookie box |
| 4 | next to the plate |
| 5 | next to the ramekin |
| 6 | on the cookie box |
| 7 | on the ramekin |
| 8 | on the stove |
| 9 | on the wooden cabinet |

6 phrasings (1 identity + 5 paraphrased, all semantic-preserving):

| Phrasing ID | Template |
|---|---|
| P0 (identity) | `pick up the black bowl {SPATIAL} and place it on the plate` |
| P1 (word order) | `place on the plate the black bowl that is {SPATIAL}` |
| P2 (synonym) | `grab the dark bowl {SPATIAL} and put it on the plate` |
| P3 (passive voice) | `the black bowl {SPATIAL} should be picked up and placed on the plate` |
| P4 (verb change) | `move the black bowl {SPATIAL} onto the plate` |
| P5 (compact) | `transfer the black bowl {SPATIAL} to the plate` |

10 tasks × 6 phrasings = 60 instruction strings. Frozen.

### §III.C. Episode protocol

- 50 episodes per (task, phrasing) cell
- Total: 50 × 10 × 6 = 3000 episodes
- Episode seeds: episode i in cell (task t, phrasing p) gets seed `1000000 + t × 10000 + p × 1000 + i` (deterministic, reproducible)
- Per-task per-phrasing 3-stage SR recorded

### §III.D. Identity-SR replication gate (mandatory before applying conditional rules)

OFT identity SR is the placement-stage SR on phrasing P0 (identity).

**Pass criterion**: identity SR ≥ 92% on full 10-task × 50-episode subset (n=500 total).

**Computation**: pass criterion derives from literature OFT SR ≈ 97% with binomial SE for n=500 of ≈ 0.78pp. 92% is 6σ below literature, allowing for implementation variance while detecting severe bugs.

**Failure cases**:
- Identity SR ∈ [80%, 92%]: marginal. Up to 8 hours of debugging permitted. After 8 hours without resolution, escalate to severe-bug branch.
- Identity SR < 80% or unresolved after 8 hours: severe bug. Up to 16 hours total debugging permitted. If unresolved at 16 hours cumulative, activate **Plan B** (§III.G).

**Rule adjustment if 92% ≤ identity SR < 97%**: Subtract $(97 - \text{observed\_identity})$ from all paraphrased SR comparison thresholds. E.g., if identity SR = 94%, "≥ identity − 2pp" becomes "≥ 92pp" instead of "≥ 95pp".

### §III.E. Per-task McNemar test protocol

For each task $t$ and each paraphrased phrasing $p \in \{P1, ..., P5\}$:

Construct paired data: 50 episodes under identity P0 vs 50 episodes under paraphrasing $p$, with episode seeds aligned (same seed for paired episodes). McNemar test on paired success/failure outcomes:

$$\text{McNemar } \chi^2 = \frac{(|n_{01} - n_{10}| - 1)^2}{n_{01} + n_{10}}$$

where $n_{01}$ = succeed under identity but fail under paraphrasing, $n_{10}$ = vice versa.

Per-task per-phrasing $p$-value recorded.

### §III.F. Conditional framing rules (mutually exclusive, frozen)

Applied **after** §III.D gate passes (identity SR ≥ 92%).

**Rule A (visual-grounded)**: For all 5 paraphrased phrasings, mean paraphrased SR ≥ (identity SR − 2pp); AND McNemar $p > 0.1$ for all 50 (task × paraphrasing) cells.

Implication: OFT is robust to language phrasing; visual pathway dominates. M3 thesis strengthened to "demonstrations carry no language signal, and OFT does not rely on it." Paper §V.B writes contrastive (r_h, r_demo, r_z) chain + Theorem 3 instantiation as main story.

**Rule B (language-conditioned, brittle)**: Mean paraphrased SR ≤ (identity SR − 15pp); AND McNemar $p < 0.01$ in ≥ 3 paraphrased phrasings (averaged across tasks); AND failure pattern consistent across tasks.

Implication: OFT critically depends on language token specifics. M3 thesis narrowed to "demonstrations don't impose language conditioning during distillation, but OFT recovers it through architectural mechanism." Paper §V.B writes architectural-comparison framing.

**Rule C (mixed, per-task heterogeneous)**: Per-task SR drop variance > 20pp² across 10 tasks; AND neither Rule A nor Rule B uniformly apply.

Implication: Task-specific dependence. Some tasks robust, others brittle. M3 thesis qualified: language signal flows through task-conditional pathway (likely scene-query mechanism, explanation C). Paper §V.B writes nuanced framing with per-task breakdown table.

**Rule D (thesis falsification)**: For ≥ 8 of 10 tasks, mean paraphrased SR drop ≥ 50pp on all 5 paraphrasings; AND OFT identity SR ≥ 95%.

Implication: OFT is strongly language-conditioned despite demonstration-level orthogonality. This contradicts Theorem 3's prediction that demonstration supervision cannot transmit language signal. Paper main thesis fails. **Fallback to Tier 3**: reframe to "OFT and OTP-Soft differ architecturally in language signal preservation; mechanism remains open." Theorem 3 either has hidden assumption violations or inductive bias provides path beyond demonstration MI.

Prior assignment (for §III.G Plan B trigger analysis): P(A) = 0.50, P(B) = 0.12, P(C) = 0.30, P(D) = 0.08.

### §III.G. Plan B fallback (engineering failure path)

**Trigger**: cumulative debug time > 16 hours without passing identity-SR gate.

**Substitute experiment**: OTP-Soft V3 self-ablation. Use existing V3 checkpoint (no new implementation required) to run unseen-phrasing eval on identical paraphrased instructions. Expected result: V3 SR = 0/50 across all phrasings (V3 is 0/50 baseline). Data interpretation:

> "OTP-Soft V3 produces 0% SR across all 6 phrasings (identity + 5 paraphrased), confirming our diagnostic finding (§V.B C1) that V3's head is fully scene-reactive without language conditioning. OFT inference baseline implementation deferred due to engineering constraints; cross-architecture comparison in §V.D limited to within-OTP-Soft variants."

Paper §V.B writes M3-thesis without OFT comparison. Tier 3 sell-point.

This is honest acknowledgment of engineering constraints, not a failed experiment.

---

## §IV. Path B Implementation Plan (Frozen)

Path B: deterministic head + decoder Cocos. Does not depend on OFT ablation outcome.

### §IV.A. Steps

| Step | Description | Estimated time |
|---|---|---|
| 1 | Fix V3 ckpt save bug (filter `requires_grad=True` only; ~16GB → ~185MB per ckpt) | 30 min |
| 2 | Implement deterministic head: `OTPHead.__init__(deterministic=True)` bypasses flow_matcher; cross-attention output → linear projection to (N_obj, H, 6) | 2-3 hr |
| 3 | **z-distribution compatibility test (first-class)**: V3 CFM head vs Path B deterministic head (random-init) on 100 training demos. Compare z norm, per-dim std, covariance structure. Decide whether V3 decoder weights can be inherited or must retrain from scratch. Statistics collection model-agnostic for later OFT comparison reuse. | 1 hr |
| 4 | Implement decoder Cocos: source distribution N(α F_φ(c), β² I) replacing N(0, I). F_φ is new MLP module conditioned on z. | 3-4 hr |
| 5 | Single-task overfit sanity (task 0, 500 steps, L1 → 0.01) | 1 hr |
| 6 | Multi-seed full retrain (3 seeds × 50 epoch) | overnight per seed |

### §IV.B. Architectural commitments (locked)

- Backbone: OpenVLA-OFT (SigLIP+DINOv2 dual encoder + Llama2-7B), frozen, accessed via internal modules per `openvla_wrapper.py`
- OTP head: cross-attention with 5 learnable object queries (`num_objects=5`), 4 layers (`num_layers=4`), hidden_dim 1024, `deterministic=True`
- Decoder: LanguageAgnosticDecoder with explicit C3 guards (static signature inspection + dynamic behavioral tests)
- Proprioception: pos(3) + quat(4) + gripper=0(1) = 8d (gripper structurally zero per V3 memory)

### §IV.C. Multi-seed retrain protocol

- Seeds: {42, 137, 2026}
- Epochs: 50 each
- Learning rate: 1e-4 (per V3 memory; 1e-3 causes head divergence)
- Batch size: 16 (per V3 config)
- Checkpoint save filter: only `requires_grad=True` params (~185MB instead of ~16GB)

### §IV.D. Path B evaluation protocol

For each of 3 seeds:
- Sim eval on LIBERO-Spatial: 10 tasks × 50 episodes = 500 episodes
- Diagnostic suite repeated: Gate 1 + C1 + η² on Path B model
- Component ablation: drop mesh / drop grasp / drop proprio (each retrained 1 seed × 50 epoch)

Aggregate: mean ± std SR across 3 seeds. Report per-task SR. Required for paper §V.C.

---

## §V. Paper Framing Direction (Conditional on §III.F outcome)

This section sketches paper §V structure under each outcome. Specific paragraph writing deferred to paper writing phase.

### Outcome A (Rule A fires, visual-grounded)

- Title candidate: "Tracing Conditioning Failure in Stacked Flow Matching VLA Policies: Language Signal Loss in Pretraining-to-Demonstration Distillation"
- §V.B: three-layer diagnostic + Theorem 3 instantiation + Rule A SR data
- §V.D: OFT vs Path B unseen-phrasing comparison (both retain SR under paraphrasing if Rule A true)
- Sell point: strongest. M3 + OFT both support thesis.

### Outcome B (Rule B fires, language-conditioned brittle)

- Title candidate: "Stacked Flow Matching Heads Fail to Preserve Backbone Language Signal in VLA Policies"
- §V.B: three-layer diagnostic + Theorem 3 + architectural failure framing
- §V.D: OFT brittleness (SR drop under paraphrasing) contrasts with Path B robustness
- Sell point: narrower scope, architectural comparison.

### Outcome C (Rule C fires, mixed)

- Title candidate: "Quantifying Modality Contributions in VLA Policies: Per-Task Language Dependence on LIBERO-Spatial"
- §V.B: three-layer diagnostic + per-task heterogeneity table + Theorem 3 with task-conditional discussion
- §V.D: per-task OFT/Path B comparison table
- Sell point: nuanced, requires careful Theorem 3 adaptation for task-conditional case.

### Outcome D (Rule D fires, thesis falsification)

- Title candidate: "An Architectural Comparison of OFT and OTP-Soft for Language-Conditioned Manipulation" (Tier 3 framing)
- §V.B: three-layer diagnostic + architectural-difference observation (Theorem 3 retracted or reframed as conjecture)
- §V.C: Path B as alternative architecture; SR comparison
- §V.D: ablation
- Sell point: weak. Paper accepts that Theorem 3's prediction is empirically violated; mechanism open question.

### Outcome Plan B (engineering failure path)

- Title candidate: same as Outcome A
- §V.B: three-layer diagnostic + Theorem 3 + V3 self-ablation as OFT-baseline substitute
- §V.D: within-OTP-Soft variants only (V3 vs Path B)
- Acknowledgment in §VI: "OFT inference baseline deferred"
- Sell point: weakened by missing cross-architecture comparison.

---

## §VI. Day-By-Day Execution Schedule

### Day 1 (2026-05-11, today, afternoon-evening)

**Track 1**: Implement OFT predictor (Stage 1 of OFT ablation)
- `OpenVLAOFTPredictor` three methods
- Determinism verification (1 task × 5 episode × 2 runs, bit-exact)
- Estimated: 5-8 hours

**Track 2**: Path B Step 1 + Step 2
- Ckpt save bug fix
- Deterministic head implementation
- Estimated: 3-4 hours

**Track 3**: V7 commit (this document)
- Estimated: 0.5 hour

### Day 2 (2026-05-12, tomorrow)

- Morning: OFT identity sanity (Stage 2). 5 task × 10 episode = 50 episodes. Check identity SR ≥ 92%.
- If pass: launch OFT full ablation (Stage 3) overnight, 3000 episodes
- If fail (8 hr debug): activate Plan B; instead run V3 self-ablation on paraphrased instructions
- Concurrent: Path B Step 3 (z-distribution test) + Step 4 (decoder Cocos)

### Day 3 (2026-05-13)

- OFT Stage 3 results in (or Plan B V3 data)
- Apply §III.F conditional rules; lock paper framing
- Path B Step 5 (single-task overfit sanity)

### Day 4-7 (2026-05-14 to 05-17)

- Path B multi-seed retrain (Step 6) + component ablation
- Diagnostic suite on Path B model

### Day 8-14 (2026-05-18 to 05-24)

- Paper writing
- §V.B according to locked framing
- Theorem 3 full proof appendix

### Day 15+ (2026-05-25 onward)

- Paper revision, submission preparation
- IEEE RA-L rolling submission

---

## §VII. Open Items (Deferred to Paper Writing)

1. Theorem 3 full proof for each loss type (L2, L1, CFM, ShortcutFM). V7 commits the statement; appendix proof in paper writing.
2. $\psi$ (instruction equivalence class function) precise definition for paper. V7 commits the conceptual definition; concrete operational form in paper §III.
3. Title finalization (after §V.B framing locked).
4. Component ablation prioritization (mesh / grasp / proprio drop order).
5. LIBERO-Goal / LIBERO-10 cross-validation: explicit future work in §VI Discussion.

---

## §VIII. References

- V6 pre-registration: `notes/2026-05-11_phase0_preregistration_v6.md` (frozen)
- Phase 0 historical record: `notes/phase0_results.md` (frozen)
- Phase 0 raw data: `results/phase0/*.json`, `results/phase0/*.npy`
- Diagnostic scripts: `scripts/diagnostic/06_test0_sanity.py` through `scripts/diagnostic/17_oft_phrasing_robustness.py`

---

*V7 frozen 2026-05-11. Subsequent updates via amendment with timestamp.*
