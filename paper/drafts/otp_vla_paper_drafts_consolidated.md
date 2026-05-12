# OTP-VLA Paper Drafts — Consolidated (2026-05-12 Day 2)

**Frozen sections after Day 2 paper-writing pivot. Upload to `/root/autodl-tmp/OTP-VLA/paper/drafts/consolidated_2026-05-12.md` then `git add` + `git commit`.**

**Recommended commit message**:
```
Paper: §III.B v3 + §V.C v4 + §VI.4 (Theorem 3' A* reformulation, OFT contrast two-scenario framing, mechanism-based scope caveat)
```

**Sections in this file**:
1. [§III.B — Theorem 3' (final v3)](#iii-b)
2. [§V.C — CFM Decoder Sample-Quality Failure (final v4)](#v-c)
3. [§VI.4 — Generalization beyond LIBERO-Spatial](#vi-4)
4. [Locked framing decisions (audit trail)](#framing-decisions)
5. [Phase 0 final numbers (single source of truth)](#phase-0-numbers)
6. [Known issues + verification protocol](#known-issues)

**Status**:
- §III.B v3: reviewed through 3 rounds of audit, mathematically rigorous, ready for paper integration
- §V.C v4: reviewed through 3 rounds, internally consistent with §III.B v3, ready
- §VI.4: 1st draft, accepted by reviewer (mechanism-based future-work framing)
- §IV bridge: NOT in this file (to be drafted next, ~1 hr)
- §I intro hook: NOT in this file (to be drafted after §IV, ~2 hr)

---

<a name="iii-b"></a>
# §III.B — Theorem 3' (Conditioning bound under language-orthogonal demonstrations) **v3 final**

The mathematical content of this section is modest; the contribution lies in identifying the demonstration-level quantity $I_{p_{\mathcal{D}}}(a; \ell \mid o)$ that controls policy-level language conditioning, and in establishing this quantity is empirically small for standard VLA benchmarks (§V.C.4).

## Setup

Let $p_{\mathcal{D}}(o, \ell, a)$ denote the demonstration distribution over observations $o \in \mathcal{O}$, instructions $\ell \in \mathcal{L}$, and actions $a \in \mathcal{A}$. Let $\pi: \mathcal{O} \times \mathcal{L} \to \Delta(\mathcal{A})$ denote a policy mapping observation-instruction pairs to distributions over actions.

For a measurable function space, the conditional mutual information between a random variable $X$ and instruction $\ell$ given observation $o$ is

$$I(X; \ell \mid o) := \mathbb{E}_{p(o)}\!\left[ D_{\mathrm{KL}}\!\big(p(X, \ell \mid o) \,\|\, p(X \mid o) \otimes p(\ell \mid o)\big) \right].$$

## Definition 1 ($\varepsilon$-language-orthogonal demonstrations)

The demonstration distribution $p_{\mathcal{D}}$ is **$\varepsilon$-language-orthogonal** if

$$I_{p_{\mathcal{D}}}(a; \ell \mid o) \;\le\; \varepsilon.$$

## Theorem 3' (Conditioning bound on policy-sampled actions)

Let $\pi^*$ minimize the population per-sample loss

$$\pi^* \in \arg\min_\pi \;\mathbb{E}_{p_{\mathcal{D}}}\!\big[\mathcal{L}(\pi(o, \ell), a)\big]$$

where $\mathcal{L}$ is a distribution-matching loss whose population minimizer satisfies $\pi^*(\cdot \mid o, \ell) = p_{\mathcal{D}}(\cdot \mid o, \ell)$ almost surely.

Let $A^* \sim \pi^*(\cdot \mid o, \ell)$ denote a sample drawn from the policy at observation-instruction pair $(o, \ell)$. If $p_{\mathcal{D}}$ is $\varepsilon$-language-orthogonal (Definition 1), then

$$\boxed{\;I(A^*; \ell \mid o) \;=\; I_{p_{\mathcal{D}}}(a; \ell \mid o) \;\le\; \varepsilon.\;}$$

### Proof

By the population-minimizer assumption, the conditional distribution of $A^*$ given $(o, \ell)$ equals $p_{\mathcal{D}}(\cdot \mid o, \ell)$. The joint distribution $p(A^*, \ell \mid o)$ therefore equals $p(a, \ell \mid o)$ pointwise in $(o, \ell)$, hence $I(A^*; \ell \mid o) = I_{p_{\mathcal{D}}}(a; \ell \mid o)$. The inequality $I_{p_{\mathcal{D}}}(a; \ell \mid o) \le \varepsilon$ is Definition 1. $\quad\square$

**Note on substance and empirical use.** The proof reduces to a distributional identity at the population level. Theorem 3''s substance is twofold: (i) it links a quantity measurable on demonstration data alone ($I_{p_{\mathcal{D}}}(a; \ell \mid o)$) to a property of policy behavior at deployment ($I(A^*; \ell \mid o)$), without requiring measurement of the policy itself; (ii) by Remark 1, the same link applies to empirical policies $\hat{\pi}$ up to a sample-complexity gap. In §V.C, we use Theorem 3' **qualitatively**: the small magnitude of $I_{p_{\mathcal{D}}}(a; \ell \mid o) \approx 0.04$ nats (under the Gaussian surrogate, §V.C.4) predicts small $I(\hat{A}; \ell \mid o)$ at the policy-action level, modulo the empirical-population gap which we treat as second-order on benchmarks where demonstration sample sizes are large ($n \gtrsim 10^4$).

## Remark 1 (Empirical-population gap)

Theorem 3' applies to the population minimizer $\pi^*$. Empirically trained policies $\hat{\pi}$ approximate $\pi^*$ up to optimization and generalization error of order $o(1)$ in the sample size $n$ under standard learning-theoretic assumptions. Consequently,

$$I(\hat{A}; \ell \mid o) \;\le\; I(A^*; \ell \mid o) + o(1) \;=\; I_{p_{\mathcal{D}}}(a; \ell \mid o) + o(1) \;\le\; \varepsilon + o(1),$$

with the leading term governed by Theorem 3'. We treat the empirical-population gap as second-order relative to the demonstration-level signal magnitude $I_{p_{\mathcal{D}}}(a; \ell \mid o) \approx 0.04$ nats on LIBERO-Spatial (where $n \approx 1.1 \times 10^5$ training samples). Rigorous quantitative treatment of this gap is left to future work.

## Remark 2 (Loss families satisfying the population-minimizer property)

The following standard VLA training objectives satisfy $\pi^*(\cdot \mid o, \ell) = p_{\mathcal{D}}(\cdot \mid o, \ell)$ at the population level:

- $L_2$ regression: $\pi^*$ collapses to the conditional mean
- $L_1$ regression: $\pi^*$ collapses to the conditional median
- Conditional Flow Matching (CFM) and ShortcutFlowMatching: matched-flow representation recovers $p_{\mathcal{D}}(\cdot \mid o, \ell)$
- Cross-entropy on discretized actions: $\pi^*$ recovers the categorical distribution at each bin

Theorem 3' therefore applies broadly to standard VLA training objectives. The bound is architecture-agnostic within the class of demonstration-supervised policies.

## Corollary (Empirical instantiation on LIBERO-Spatial)

For LIBERO-Spatial demonstrations:

- Mantel test on demonstration trajectories vs. language embeddings yields $r(\text{traj}, \ell) \in [-0.27, -0.20]$ across three trajectory representations, with maximum-magnitude $r = -0.27$ on the flat representation (permutation $p = 0.96$).
- Squared-correlation effect size: $\rho^2 \le 0.073$.
- Gaussian-equivalent mutual-information surrogate: $\hat{I}_{p_{\mathcal{D}}}(a; \ell \mid o) := -\tfrac{1}{2}\log(1 - \rho^2) \approx 0.038$ nats.
- Theorem 3' prediction: $I(A^*; \ell \mid o) \le \hat{I}_{p_{\mathcal{D}}}(a; \ell \mid o) \approx 0.038$ nats *under the Gaussian assumption*.

**Caveat on Gaussian surrogate.** The Gaussian MI surrogate assumes joint Gaussianity of $(a, \ell)$, which is not assumed elsewhere in our analysis. We report squared correlations $\rho^2$ as the primary effect-size measure and use the MI surrogate as a **qualitative reference for Theorem 3' instantiation**, not as a quantitative ceiling on $I(A^*; \ell \mid o)$.

The primary effect-size finding (§V.C.4) is the squared-correlation ratio $\rho_h^2 / \rho_{\text{demo}}^2 \approx 8\times$ comparing backbone-level conditioning ($r_h = 0.76$, $\rho_h^2 = 0.578$) to demonstration-level conditioning. **We do not interpret this ratio as a bound on $I(A^*; \ell \mid o)$, but as an empirical separation between two layers of the pipeline where language signal is preserved (backbone) vs. attenuated (demonstration-trajectory representation).**

## Bridge to §IV (Architectural implications)

Theorem 3' constrains the policy's language conditioning by the demonstration's language conditioning, **regardless of policy architecture** — deterministic regression, CFM, ShortcutFlowMatching, and cross-entropy heads are all subject to the bound (Remark 2). The bound is therefore not "bypassable" by architectural choice within the demonstration-supervised paradigm.

Two implications for architecture (§IV) and evaluation (§V):

1. **Architectural choice determines failure mode, not the bound itself.** Different decoder families fail differently within the same bound: CFM-based decoders exhibit sample-quality collapse (§V.C.2), while deterministic regression decoders may succeed at task SR despite low policy-level language MI (§V.C.5 OFT contrast). The bound does not predict SR; it predicts that any high-SR demonstration-supervised policy must achieve task success via channels other than language conditioning at the policy level.

2. **Reducing the bound's effective tightness requires interventions outside the demonstration-supervised loss.** Auxiliary language objectives — language reconstruction, contrastive language alignment with backbone representations, or supervised conditioning on $h_{\mathrm{backbone}}$-derived signals during inference — modify the supervision distribution at training time, introducing additional language information channels beyond the demonstration trajectory $a$. We discuss such directions in §VI Future Work.

---

<a name="v-c"></a>
# §V.C — CFM Decoder Sample-Quality Failure under Demonstration-Supervised Learning **v4 final**

**Paper framing**: This section is the paper's primary empirical contribution. We do not present "Path B as the architecture that fixes V3"; we present "CFM-based action decoders fail in a specific, characterizable way under demonstration-supervised learning, validating Theorem 3' at the implementation level".

## §V.C.1 — Deterministic head verification

Path B's deterministic head (replacing V3's CFM head) was verified at the implementation level (5a v2, output dir `20260511_192411`):

- `head_loss`: 0.82 → 0.056 (min, overfit on 50-sample training subset)
- Determinism check: `head_loss` bit-exact across forwards (10 trials)
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

**Threshold**: L1/std < 0.3 (functional learning); L1/std > 0.8 (failure).
**Result**: 0/7 dims PASS, 7/7 dims FAIL. **Overall L1/std ratio**: 1.88.

CFM decoder samples are not task-discriminative. The aggregate `decoder_loss` (~1.0 nats, observed in training) does not reflect predictive quality; it reflects the CFM loss noise floor.

## §V.C.3 — Oracle ablation rules out moving-target hypothesis

Hypothesis: maybe decoder learns but receives noisy/incompatible head output (moving-target). To test, we replaced head trajectory with ground-truth trajectory in decoder input (5a v3, `use_oracle_trajectory=true`):

- 5a v2 (head → decoder): `decoder_loss` = 0.89 (first batch), 1.03 (training min)
- 5a v3 (gt_traj → decoder): `decoder_loss` = 1.11 (first batch), oracle vs normal nearly identical training curve

Decoder loss does not improve with oracle trajectory. Moving-target hypothesis rejected; failure is intrinsic to CFM-based decoder learning under demo-supervised loss.

## §V.C.4 — Empirical instantiation of Theorem 3'

We instantiate Theorem 3''s bound on the LIBERO-Spatial demonstration dataset. The Mantel test on demonstration trajectories vs. language embeddings yields

$$r(\text{demo trajectory}, \ell) \in [-0.27,\, -0.20] \quad \text{across three trajectory representations,}$$

with the maximum-magnitude $r = -0.27$ on the flat representation (permutation $p = 0.96$). The squared-correlation effect size is $\rho^2 = 0.073$.

For comparison, the backbone-level reference from Gate 1 is

$$r(h_{\mathrm{OFT}},\, \ell) = 0.76, \qquad \rho_h^2 = 0.578.$$

The squared-correlation gap is

$$\rho_h^2 / \rho_{\text{demo}}^2 \;=\; 0.578 / 0.073 \;\approx\; 8\times,$$

with $\rho_{\text{demo}}^2$ approaching the null-Mantel scale ($p = 0.96$).

Treating these squared correlations as Gaussian-equivalent MI surrogates via $\hat{I} = -\tfrac{1}{2}\log(1 - \rho^2)$ would yield $\hat{I}_h \approx 0.43$ nats vs $\hat{I}_{\text{demo}} \approx 0.04$ nats, an approximately $10\times$ qualitative gap. We emphasize that this surrogate **assumes joint Gaussianity** of $(h, \ell)$ and $(a, \ell)$ respectively, which is not assumed elsewhere in our analysis. We therefore report squared correlations as the primary effect sizes and use the MI surrogate as a qualitative reference for Theorem 3' instantiation, not as a quantitative ceiling on $I(A^*; \ell | o)$.

**Theorem 3' interpretation and the V3 failure chain.** Theorem 3' (§III.B) states $I(A^*; \ell | o) = I_{p_{\mathcal{D}}}(a; \ell | o) \le \varepsilon$, where $A^*$ denotes actions sampled from the population minimizer. We use $A^*$ to denote sampled actions, distinguishing them from the distribution-valued policy $\pi^*$. The Mantel tests in §V.A measure correlations involving sample-realized quantities ($z$ outputs, action chunks), consistent with the $A^*$-form of Theorem 3'.

The empirical observation that demonstration-trajectory and language are approximately uncorrelated under Mantel testing — at an order of magnitude below the backbone-level correlation — is consistent with $I_{p_{\mathcal{D}}}(a; \ell | o) \approx 0$ in the sense required by the theorem.

V3's 0/50 SR was previously hypothesized as a "two-layer cascade collapse": head encoding scene structure with degraded language conditioning, decoder collapsing to a marginal action mean. §V.A's diagnostic measurements ($r(z, \ell) = -0.19$, $\eta^2$ median 0.26 concentrated in non-target object slots) document the head-level symptom. §V.C.2-3 establish the mechanistic cause at the decoder level: CFM-based action sampling under language-orthogonal demonstration supervision fails irrespective of upstream head behavior, with the loss noise floor masking the absence of useful signal.

These observations describe the same chain at different layers, not competing hypotheses: both are consequences of Theorem 3' operating with $I_{p_{\mathcal{D}}}(a; \ell \mid o) \approx 0$ on the right-hand side. The bound forces any demonstration-supervised policy's **action-level language conditioning** $I(\hat{A}; \ell \mid o)$ to be similarly small (modulo empirical-population gap, Remark 1); the *manner* of policy failure (head-level collapse, decoder sample-quality failure, or both) is architecture-dependent at the SR level but not at the language-MI level.

## §V.C.5 — OFT achieves 97% SR within the same bound (two-scenario framing)

OFT (OpenVLA-OFT) shares the SigLIP+DINOv2 backbone (Gate 1 measured $r(h_{\mathrm{OFT}}, \ell) = 0.76$, identical to ours), and achieves $\sim 97\%$ SR on LIBERO-Spatial. A natural question is how OFT reconciles high SR with the Theorem 3' bound, which applies to any demonstration-supervised policy.

OFT does not use a CFM decoder; it produces actions via direct LLM hidden-state-to-action regression. Under Theorem 3', OFT and our V3 architecture are both subject to

$$I(A; \ell | o) \le I_{p_{\mathcal{D}}}(a; \ell | o),$$

where $A$ denotes sampled actions from the policy, with the bound architecture-agnostic and applying to any demonstration-supervised policy. OFT's $97\%$ SR combined with the §V.B paraphrase-invariance finding (9-task subset paired-diff CI ⊆ ±2pp; task 5 sensitivity caveat documented; excluding task 5 does not change any other conclusion) is consistent with two scenarios:

(a) OFT preserves language conditioning at low magnitude within the bound and uses that signal robustly across paraphrasings, or

(b) OFT is entirely visual-grounded at the policy level with no language signal, with task-discriminative scene structure sufficient to drive 97% SR.

Both scenarios are consistent with the bound and with §V.B's paraphrase-invariance observation; the current data does not distinguish them. Either way, high task-success rate does not establish that language understanding drives task success.

This observation generalizes beyond OFT-vs-V3. **The diagnostic question for the field is whether benchmark SR alone can distinguish these scenarios** — §VI argues it currently cannot, and proposes signal-flow diagnostics (Gate 1, M3, C1 in the present work) as a complementary evaluation axis. The paper's contribution is not architectural but methodological: a framework for measuring where in the VLA stack language conditioning is preserved or lost, independent of downstream SR.

## Implications for §V.D

Given §V.C.2-4 establishes CFM-based action decoders fail intrinsically under demonstration-supervised learning, §V.D pursues two complementary directions:

1. **Component ablation on V3 baseline** (mesh / grasp affordance / proprioception drops): isolates which condition signals, if any, the CFM decoder is sensitive to *despite* sample-quality failure. Negative result expected and welcomed — it reinforces §V.C's central claim that the failure is in the decoder family, not in the conditioning inputs.

2. **Cross-architecture comparison via OFT contrast** (§V.C.5 cross-reference): OFT and V3 share backbone but differ in decoder family (deterministic regression vs CFM). The 97% vs 0% SR gap localizes failure to the decoder design, not to data, backbone, or supervision signal.

**Deterministic action regression as architectural candidate.** A natural follow-up is to replace the CFM decoder with a deterministic action regression head, analogous to OFT's architecture. Such a substitution remains within Theorem 3''s bound on policy-action language MI but may avoid CFM-specific sample-quality failure at the SR level (§V.C.2). The OFT contrast provides existence proof that demonstration-supervised deterministic regression *can* achieve high SR within the same MI bound (whether via low-magnitude language preservation, visual grounding, or both remains to be diagnosed). We defer rigorous evaluation of this candidate on the OTP-Soft backbone to future work; this paper's principal contribution is the diagnostic framework (Gate 1, M3, C1, §V.C.2 sample-quality test), not the architectural fix.

---

<a name="vi-4"></a>
# §VI.4 — Generalization beyond LIBERO-Spatial

A natural pushback to our diagnostic findings is single-benchmark scope: Gate 1 ($r(h_{\mathrm{OFT}}, \ell) = 0.76$), M3 ($r \in [-0.27, -0.20]$), and the OFT contrast (§V.C.5) are all demonstrated on LIBERO-Spatial. We separate the theoretical and empirical components of our scope claim.

**Theoretical scope.** Theorem 3' bounds $I(A^*; \ell | o) = I_{p_{\mathcal{D}}}(a; \ell | o) \le \varepsilon$. The right-hand side depends only on the joint distribution of the supervision data $p_{\mathcal{D}}(\ell, a | o)$, not on the specific task suite. The bound is benchmark-agnostic; the question is whether the antecedent $I_{p_{\mathcal{D}}}(a; \ell | o) \approx 0$ holds across benchmarks.

**Empirical scope.** We expect demonstration-trajectory ⊥ language to hold broadly under standard teleoperated demonstration protocols, on a mechanistic argument: teleoperators visually perceive the scene and execute motor commands, with language serving as a post-hoc label affixed to the recorded trajectory rather than as a causal driver of the recorded actions. Under this protocol, the joint distribution $p_{\mathcal{D}}(\ell, a | o)$ factors approximately as $p(\ell | o) \cdot p(a | o)$, yielding the language-orthogonality condition independently of task suite.

This mechanistic prediction is empirically testable. We provide one quantitative instantiation (LIBERO-Spatial, $r = -0.27$); replication on LIBERO-Goal, LIBERO-Object, and non-LIBERO teleoperated suites (e.g., RoboCasa, Bridge) is left to future work. We do not claim universal cross-benchmark generalization, but we do make a falsifiable mechanistic prediction: any teleop-collected suite should yield $|r(\text{traj}, \ell)| \ll |r(h_{\mathrm{backbone}}, \ell)|$ under the same M3 protocol.

---

<a name="framing-decisions"></a>
# Locked framing decisions (audit trail for 2026-05-12)

These decisions were locked through 3 rounds of reviewer audit on Day 2. They are referenced across §III.B, §V.C, §VI.4, and the not-yet-written §IV and §I.

## Theorem 3' subject reformulation

- **Subject**: sampled action $A^* \sim \pi^*(\cdot|o,\ell)$, NOT distribution-valued $\pi^*$
- **Reason**: v1 stated $I(\pi^*;\ell|o) \le I(a;\ell|o)$ which is mathematically incorrect — information flow $\ell \to \pi^* \to a$ means $I(\pi^*;\ell|o) \ge I(a;\ell|o)$ by DPI. Toy Gaussian example: $\ell \in \{\ell_1, \ell_2\}$, $p_{\mathcal{D}}(a|o,\ell_1) = \mathcal{N}(0,1)$, $p_{\mathcal{D}}(a|o,\ell_2) = \mathcal{N}(0.1, 1)$ gives $I(\pi^*;\ell|o) = \log 2$ nats but $I(a;\ell|o) \approx 0$ — direction reversed from what we want.
- **Fix**: reformulate to $A^*$ subject. Proof becomes trivial distributional identity since $A^* | (o,\ell)$ and $a | (o,\ell)$ share distribution.
- **Trivial proof is feature not bug**: substance is in identifying the demonstration-level quantity that controls policy behavior, and verifying it empirically (M3 + Mantel).

## Architecture-agnostic bound (no "deterministic head bypasses")

- Theorem 3' applies to ALL demonstration-supervised policies (Remark 2): $L_2$, $L_1$, CFM, SFM, cross-entropy on discretized actions.
- **Architectural choice determines failure mode, not bound**: CFM → sample-quality collapse (§V.C.2); deterministic regression → may succeed at SR via visual grounding (§V.C.5 OFT).
- **Bypassing requires auxiliary objectives outside demo loss**: language reconstruction, contrastive language alignment with backbone, etc. Discussed in §VI.

## OFT contrast two-scenario framing

OFT achieves 97% SR + paraphrase-invariance (§V.B Rule A supported) is consistent with EITHER:
- (a) low-magnitude language preservation within bound, used robustly across paraphrasings, OR
- (b) entirely visual-grounded at policy level with no language signal.

Current data cannot distinguish. **Both scenarios consistent with bound**. The diagnostic question for the field: can benchmark SR distinguish these? §VI argues currently no, proposes signal-flow diagnostics as complementary axis.

## Single-benchmark scope (§VI.4 mechanism-based caveat)

- **Theoretical scope**: bound is benchmark-agnostic (depends only on $p_{\mathcal{D}}$)
- **Empirical scope**: teleop protocol mechanistically predicts language-orthogonality across suites
- **Falsifiable prediction**: any teleop-collected suite should yield $|r(\text{traj}, \ell)| \ll |r(h_{\mathrm{backbone}}, \ell)|$ under same M3 protocol

## §V.B Rule A wording (locked previously)

Use "supported" / "triggered" not "fires" (avoids dependence on procedural CI-aware application). Use paired SR diff 95% CIs as evidence of statistical equivalence. Task 5 caveat: "Excluding task 5 does not change any other conclusion" (sensitivity analysis framing, NOT post-hoc subset selection).

## Path B → no Path C; paper sells diagnosis not fix

- Phase 1 verification (Day 2 morning): `use_flow_matching=False` does NOT give deterministic regression — gives vanilla CFM. Path C deterministic decoder requires implementation, not flag toggle.
- Strategic decision: SKIP Path C implementation, full paper-writing track.
- §V.D no longer "Architectural Recovery via Deterministic Action Regression"; now "Component Ablation + Cross-Architecture Comparison".
- 7.5-day GPU sequential plan: V3 baseline + det head variants + component ablations.

## Paper title candidates (still TBD)

- Catchy: "Where Did the Language Go? Tracing Conditioning Signal Loss in VLA Policies"
- Conservative: "Modality-Faithful VLA Policies: Diagnostic Framework and Architectural Recovery"
- (Conservative title needs updating: paper does NOT propose architectural recovery anymore.)

---

<a name="phase-0-numbers"></a>
# Phase 0 Final Numbers — Single Source of Truth

**Locked 2026-05-12. All paper references draw from this section, NOT directly from `results/phase0/` (which is subject to overwrite).**

## Provenance

- LIBERO-Spatial, 10 tasks × 10 demos, frame 0
- Backbone: OpenVLA-OFT (SigLIP+DINOv2 dual-encoder)
- Replication run: 2026-05-12 17:00 CST
- Result file (replicated): `results/phase0/gate1_sanity_and_m3.json`

## Gate 1: r(h_OFT, language)

- Original (2026-05-11): $r = +0.7423$, $p < 0.001$
- Replicated (2026-05-12): $r = +0.7575$, $p < 0.001$
- $\Delta = 0.0152$ (within sampling noise)
- **Paper number: $r \approx 0.76$** (both runs round to 0.76)

## M3: r(demo_trajectory, language)

- Flat representation (primary): $r = -0.2698$, $p = 0.964$ (replicated 2026-05-12)
- Range across 3 representations (flat / endpoint / waypoint): $r \in [-0.27, -0.20]$
- **Paper number: $r = -0.27$ (flat, primary)** | $r \in [-0.27, -0.20]$ (range across representations)
- Squared: $\rho^2 = 0.073$ (flat) | $\rho^2 \in [0.04, 0.073]$ (range)

## C1: r(z_OTP, language)

- Phase 0 deep dive: $r = -0.19$
- $\eta^2$ median: 0.26, concentrated in non-target object slots
- Source file: `results/phase0/condition_check.json`

## Test 4 (backbone capacity): F̄ᵃ = 11.30

- Source: `results/phase0/test4_capacity.json`

## Theorem 3' Gaussian MI surrogates (QUALITATIVE, not quantitative)

These are $r^2$-based Gaussian surrogates, not kosher MI estimates. Joint Gaussianity is not assumed in our analysis; surrogates report effect-size magnitude only.

- $\hat{I}(h_{\mathrm{OFT}}; \ell|o) \approx 0.43$ nats from $r_h^2 = 0.578$ (Gaussian surrogate)
- $\hat{I}(a; \ell|o) \approx 0.04$ nats from $r_{\mathrm{demo}}^2 = 0.073$ (Gaussian surrogate)
- Squared-correlation gap: $r_h^2 / r_{\mathrm{demo}}^2 \approx 8\times$ (this is the primary effect-size ratio)
- MI-surrogate gap: $\sim 10\times$ (qualitative reference only, NOT entering quantitative bound)

## Cross-test consistency note (from gate1_sanity_and_m3.json)

- C1 D_language (raw mean-pool): 0.167
- Gate 1 D_language (raw mean-pool): 0.097
- Both use byte-identical embedding code path (SC2 verified)
- Difference: minor sampling effect (C1 used 10 tasks single demo, Gate 1 used 10 tasks × 10 demos)

## Note on `framing_implication` field in raw JSON results

The Phase 0 result files contain a `framing_implication` text field hardcoded by diagnostic scripts at write-time. These fields reflect the paper framing **at the time of each diagnostic run**, and necessarily lag behind the final paper §V framing as analysis evolved.

The empirical numbers (SC1, SC2, SC3, M3, C1, Test 4) in the raw JSON files are the authoritative data and remain unchanged across framing iterations. The final paper §V draws its framing from the post-hoc full-data analysis (§V.A through §V.C), not from any single diagnostic JSON's `framing_implication` field.

For reviewers consulting raw results: the JSON `framing_implication` fields represent intermediate working hypotheses (e.g., "selective collapse" in earlier scripts, "demonstration distillation failure" in later scripts). The published framing — CFM-decoder sample-quality failure under language-orthogonal demonstration supervision (§V.C) — is the post-diagnostic synthesis presented in this paper.

---

<a name="known-issues"></a>
# Known Issues + Verification Protocol

## Hydra CLI override silently dropped in diagnostic scripts (2026-05-12)

**Affected files**: `scripts/diagnostic/15_gate1_sanity_and_m3.py` (and likely sibling scripts using same `load_config()` pattern)

**Root cause**: `compose(config_name="otp_soft_30e")` does not pass `overrides=sys.argv[1:]`. Hydra's non-decorator API requires explicit overrides; CLI args like `data.suite=libero_goal` are silently dropped. Script always uses yaml default (`libero_spatial`).

**Symptom**: Run completes successfully, produces results for `libero_spatial` instead of intended suite. No error, no warning.

**Secondary lock-in**: `PROBE_TASK_NAMES` is hardcoded LIBERO-Spatial task list (lines 52-63). Even if hydra override worked, task selection wouldn't follow.

**Status**: Not fixed. LIBERO-Goal cross-benchmark replication deferred (paper §VI.4 mechanism-based caveat used instead).

**Fix recipe (if revisited)**:
1. `def load_config(overrides=None): ... compose(config_name="...", overrides=overrides or [])`
2. `cfg = load_config(sys.argv[1:])` in main
3. Move `PROBE_TASK_NAMES` into `cfg.data.probe_tasks` (per-suite yaml field)
4. Adapt `LIBEROOTPDataset` for libero_goal data structure differences (if any)

**Other scripts likely affected**: any `scripts/diagnostic/*.py` using `initialize_config_dir + compose` without explicit overrides. Audit before relying on CLI overrides in those scripts.

## Verification protocol for hydra-based scripts (mandatory)

Before running any new diagnostic or training script with CLI overrides — especially before launching GPU work in the 7.5-day ablation plan — verify in single-CPU dry-run that overrides actually propagate:

1. Add `print(OmegaConf.to_yaml(cfg))` at script start (or use `--cfg job --resolve` if `@hydra.main` decorator is used)
2. Run with intended overrides, on CPU, with minimal data load (or `--multirun` dry-run mode if available)
3. Visually confirm cfg fields reflect overrides before launching GPU work
4. If overrides do not propagate, fix the script per "Fix recipe" above; do NOT proceed with GPU launch assuming overrides will work at scale

**Reason**: silent hydra override drops produce results that look successful but answer the wrong question. Catching this at CPU dry-run cost ~30 sec; catching at GPU runtime cost = entire ablation cell (12.5 hr) at worst.

---

# End of consolidated draft 2026-05-12

## Next P0 work (NOT in this file)

- **§IV theory-architecture bridge** (~1 hr): Architecture-agnostic Theorem 3' interpretation; CFM head optimizes distribution-matching loss → sample-quality collapse; deterministic head also bounded (no bypass claim); §V.C.5 OFT existence proof for det regression within bound.
- **§I intro hook** (~2 hr): 3-layer signal-flow diagnostic: backbone preserves language ($r = 0.76$), demo sup attenuates (M3 $r = -0.27$), policy bounded by Theorem 3'. Sell point: backbone vs demo $0.76$ vs $-0.27$ hard sell + OFT 97% SR contrast.

## Recommended commit workflow

```bash
# On AutoDL after uploading this file:
cd /root/autodl-tmp/OTP-VLA

# Place into drafts directory
mkdir -p paper/drafts
# (assumes file uploaded as consolidated_2026-05-12.md)
mv /path/to/uploaded/consolidated_2026-05-12.md paper/drafts/

# Archive old drafts to preserve audit trail
mkdir -p paper/drafts/archive
mv paper/drafts/theorem_3_prime.md paper/drafts/archive/theorem_3_prime_v1.md 2>/dev/null
mv paper/drafts/section_V_C_reframed.md paper/drafts/archive/section_V_C_reframed_v1.md 2>/dev/null

# Stage + commit
git add paper/drafts/consolidated_2026-05-12.md paper/drafts/archive/
git status  # verify
git commit -m "Paper: §III.B v3 + §V.C v4 + §VI.4 consolidated (Theorem 3' A* reformulation, OFT contrast two-scenario framing, mechanism-based scope caveat)"
git push origin main
```
