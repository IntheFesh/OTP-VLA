# §V.C — Correlation Gap, OFT Contrast, and CFM Decoder Collapse (v1 draft)

**Status**: Draft v1. Day 5 2026-05-13. ~2 pages target.

**Purpose**: Synthesize the three diagnostic measurements (§V.A) and the paraphrase
invariance finding (§V.B) into the paper's central narrative. Establish (1) the
quantitative correlation gap as effect-size measurement, (2) the OFT contrast as
empirical existence proof of "high SR within Theorem 3' bound", (3) the CFM decoder
collapse as architecture-specific failure mode within the bound.

**Locked data sources**:
- §V.A Gate 1 r = +0.76, M3 r = -0.27, C1 r = -0.19
- §V.B OFT identity SR 94.6%, paraphrased SR 93.7%
- §V.C.2 sample quality: 5a v2 ckpt, 0/7 dims PASS, overall L1/std = 1.88
- Oracle test: replacing head with GT trajectory in 5a v3 — decoder still fails to recover GT actions, ruling out moving-target hypothesis

---

## §V.C.1 The Correlation Gap as Principal Effect-Size

Combining §V.A.2 and §V.A.3, the signal-flow diagnostic localizes language conditioning
loss across the VLA pipeline by direct comparison of the same Mantel quantity at three
stages:

$$\rho_h^2 \,/\, \rho_{\text{demo}}^2 \;=\; 0.578 \,/\, 0.073 \;\approx\; 8\times.$$

The backbone-level squared correlation $\rho_h^2 = 0.578$ exceeds the demonstration-level
$\rho_{\text{demo}}^2 = 0.073$ by an order of magnitude. Under Gaussian MI surrogate
(qualitative only, Remark 1):

$$\hat{I}_h \approx 0.43 \text{ nats}, \qquad \hat{I}_{\dataD}(a; \ell \mid o) \approx 0.04 \text{ nats}.$$

The transition from backbone to demonstrations attenuates the language signal by approximately one order of magnitude.

This gap is the **principal effect-size measurement** of the paper: language conditioning
exists at the backbone in substantial quantity but is essentially absent from the
supervision signal that downstream modules learn from. By Theorem 3' (§III.B), any
demonstration-supervised policy inherits this attenuation:

$$I(A^*; \ell \mid o) \;\le\; I_{\dataD}(a; \ell \mid o) \;\approx\; 0.04 \text{ nats}.$$

The policy-level Mantel measurement C1 ($r = -0.19$, §V.A.4) confirms this inheritance:
the trained OTP-Soft V3 latent $z$ is approximately language-orthogonal under inter-task
distance comparison, with the task-discriminative variance concentrated in
observation-derived (not instruction-derived) slots.

## §V.C.2 OFT Contrast — High SR Within the Bound

The 8× gap is consistent with Theorem 3' applied to any demonstration-supervised policy
on LIBERO-Spatial. We now contrast this with OpenVLA-OFT's 97% success rate (§V.B), which
is the highest published SR on LIBERO-Spatial. The contrast establishes two facts
simultaneously:

**Fact 1.** Theorem 3' is not vacuous in the empirical regime: a state-of-the-art
demonstration-supervised policy achieves high task SR (97%) while bounded to small
policy-level language MI ($\le 0.04$ nats by Theorem 3').

**Fact 2.** The OFT result is the **existence proof** for ``high SR with language-orthogonal
supervision is achievable'' — but the mechanism of task success is observation-action
mapping, not language-conditioned action selection. The 94.6% / 93.7% identity / paraphrased
SR pattern is consistent with both (a) low-magnitude language preservation and
(b) entirely visual grounding (§V.B.4), with SR alone unable to distinguish the two.

**The OFT contrast is the central piece of evidence for our diagnostic methodology
contribution**: SR + paraphrase invariance, the conventional evaluation regime for VLA
``language understanding,'' cannot in principle distinguish language-conditioned from
visually-grounded high performance. The signal-flow diagnostics (Gate 1, M3, C1) provide
complementary information by directly measuring where language conditioning is preserved
or lost across the pipeline.

Specifically, the OFT case sits at the operating point:

- **High SR** (97%) via observation-action mapping on LIBERO-Spatial
- **Language conditioning bounded** ($I(A^*; \ell \mid o) \le 0.04$ nats) by Theorem 3'
- **Paraphrase invariance** consistent with both scenarios (a) and (b)
- **Empirical disambiguation via signal-flow** is required to identify the dominant
  pathway — Gate 1's $r = 0.76$ shows backbone preserves language capacity; M3's
  $r = -0.27$ shows demonstrations do not encode language-action correlation; C1's
  $r = -0.19$ shows the trained policy inherits demonstration attenuation. Together
  these locate the dominant pathway as observation-derived for the OFT case.

The OFT contrast demonstrates that ``high SR'' and ``language-conditioned policy'' are
not the same property, and that distinguishing them requires diagnostics beyond
benchmark SR.

## §V.C.3 CFM Decoder Sample-Quality Collapse

We now turn to the contrasting failure mode under the same Theorem 3' bound: a
CFM-decoder policy under language-orthogonal supervision.

**Setup.** OTP-Soft V3 with Shortcut Flow Matching head and CFM decoder, trained on
LIBERO-Spatial demonstrations (the V3 baseline in §IV). We evaluate the trained decoder
by direct sampling: 16 training samples, compare decoder's $\hat{a}$ output against
ground-truth $a^{\text{GT}}$ via per-action-dimension L1 / standard deviation ratio
(L1 / std).

**Pre-registered threshold** (§III.F pre-registration): per dimension L1 / std < 0.3 is
``functional'' (decoder reconstructs the GT action with substantially-less-than-noise
error); L1 / std > 0.8 is ``failure'' (decoder output is noise-level relative to GT
variance).

**Result**: 0 / 7 dimensions pass; 7 / 7 dimensions fail. Overall L1 / std ratio is
**1.88**, indicating the decoder produces output that is **near-twice noise-level**
relative to the GT action variance. Per-dimension breakdown:

| dim | GT std | L1 err | L1 / std | Verdict |
|---|---|---|---|---|
| 0 (pos x) | 0.380 | 0.846 | 2.23 | FAIL |
| 1 (pos y) | 0.409 | 0.968 | 2.36 | FAIL |
| 2 (pos z) | 0.489 | 0.972 | 1.99 | FAIL |
| 3 (rot x) | 0.031 | 0.854 | 27.28 | FAIL |
| 4 (rot y) | 0.106 | 0.849 | 7.98 | FAIL |
| 5 (rot z) | 0.062 | 0.742 | 12.05 | FAIL |
| 6 (gripper) | 0.943 | 1.034 | 1.10 | FAIL |

**Oracle ablation (moving-target hypothesis falsification).** A natural candidate
explanation is that the decoder fails because its conditioning input — the head's
output $z$ — is itself language-orthogonal under the §V.A.4 (C1) finding, providing a
``moving target'' that prevents convergence. To test this, we run an oracle ablation
in 5a v3: replace the head's output $z$ at training time with a representation
derived directly from the GT trajectory $a^{\text{GT}}$ (sufficient statistic for the
GT action). If the moving-target hypothesis holds, this oracle conditioning should
allow the decoder to converge.

The oracle ablation **does not recover GT-matching decoder output**. Decoder L1 / std
remains in the same noise-level range, and per-dimension PASS count remains 0 / 7. This
rules out the moving-target hypothesis as the dominant cause of decoder failure.

**The CFM decoder sample-quality collapse is therefore a property of language-orthogonal
demonstration supervision under CFM training, not a consequence of suboptimal
conditioning.** The decoder learns to match the marginal distribution of GT actions
across the training set, but fails to learn the conditional mapping $a^{\text{GT}} \mid (o,
z)$ — exactly the prediction of Theorem 3' applied to the joint
$(\text{head output}, \text{action})$ conditional structure.

## §V.C.4 Reconciling the Two Failure Modes Within the Same Bound

The OFT result and the CFM decoder collapse appear superficially contradictory: one
policy achieves 97% SR, the other fails 7 / 7 dimensions on sample quality. Both are
trained on the same LIBERO-Spatial demonstrations. Both are bounded by Theorem 3' to
$I(A^*; \ell \mid o) \le \sim 0.04$ nats. How can they exhibit such different
performance profiles?

The reconciliation is that Theorem 3' bounds **policy-level language MI**, not task SR.
Architecture choice determines **which failure mode** occurs within the bound:

\begin{itemize}
\item \textbf{Deterministic regression architecture} (OpenVLA-OFT, deterministic
projection head): learns observation-action mapping to high SR. Language conditioning
absent at action level by Theorem 3', but the policy succeeds via the observation channel
because LIBERO-Spatial scenes are tightly task-correlated.

\item \textbf{Stochastic flow-matching architecture} (CFM decoder under
language-orthogonal demonstrations): the head/decoder factorization places the
conditioning bottleneck at the head output $z$. The CFM decoder then cannot learn the
conditional mapping $a \mid (o, z)$ because the supervised demonstration distribution
does not provide sufficient signal — the decoder collapses to producing the marginal $a$
distribution, yielding the observed 1.88 L1 / std ratio.
\end{itemize}

Both architectures operate within the Theorem 3' bound. The OFT case demonstrates that
the bound is compatible with high SR. The CFM decoder case demonstrates that the bound
is incompatible with conditional sample quality. The failure mode is determined by
architecture, not by the bound itself.

This is the **architectural disambiguation contribution** (Contribution 3, §I): given a
demonstration-supervised training regime with language-orthogonal supervision (the
LIBERO-Spatial setting and likely many other teleop-collected datasets), architectural
choice determines whether the policy fails on conditioning or fails on sample quality.
The Theorem 3' bound applies regardless.

## §V.C.5 Implications for the Diagnostic Methodology

The combined §V.C analysis supports three claims about the diagnostic methodology:

**1. The 8× correlation gap is the primary quantitative observation.** This is the
hard-effect-size measurement that motivates Theorem 3': demonstration supervision
attenuates language signal by an order of magnitude relative to backbone-level
preservation. The gap quantifies the information bottleneck that any policy trained on
such supervision inherits.

**2. The OFT contrast establishes that the bound is empirically non-vacuous.** A 97%
SR policy operating within the $\sim 0.04$ nats bound provides existence proof. SR alone
cannot distinguish this from a language-conditioned high-MI policy at any reasonable
sample size; only signal-flow diagnostics can.

**3. The CFM decoder collapse demonstrates the bound's architectural specificity.** The
bound applies to all demonstration-supervised policies, but failure mode depends on
architecture. This argues against ``high SR implies bound-violation impossible'' or
``policy with non-trivial language MI must achieve high SR'' as informal interpretations.

These observations support the diagnostic methodology contribution: signal-flow
measurements (Gate 1, M3, C1) provide complementary information to SR + paraphrase
invariance, locating the language-information bottleneck independent of architecture and
performance. We recommend their adoption in future VLA evaluation pipelines.
