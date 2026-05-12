# §VI — Discussion (v1 draft)

**Status**: Draft v1. Day 3 P1. ~2 pages target.

**Structure** (per userMemories #16 — §VI must explicitly address 4 pushbacks):
- §VI.1 Alternative interpretations
- §VI.2 Pushback 1 — Mantel test setup
- §VI.3 Pushback 2 — Path B is also demo-supervised, why does any architecture work?
- §VI.4 Pushback 3 — Auxiliary language losses as bypass route (already drafted in consolidated v1, this version refines)
- §VI.5 Pushback 4 — Single benchmark scope (already drafted as §VI.4 in consolidated; renumbering needed)
- §VI.6 Future work — auxiliary objective interventions
- §VI.7 Limitations

**Important note**: The §VI.4 that exists in `paper/drafts/otp_vla_paper_drafts_consolidated.md` is the "single-benchmark scope" content. In the final paper organization, single-benchmark scope is §VI.5 (the 4th of 4 pushbacks). The numbering in this draft follows the final order; the prior §VI.4 content is preserved verbatim as §VI.5 here for continuity.

---

## §VI.1 Alternative interpretations of the §V findings

The empirical findings of §V admit alternative interpretations beyond the one offered by Theorem 3'. We address the three most plausible:

**Alternative interpretation 1: Demonstration data is biased toward visually-distinctive tasks, masking language-action correlation.** It is conceivable that LIBERO-Spatial demonstrations exhibit near-zero language-trajectory correlation (M3) not because demonstrations are intrinsically language-orthogonal, but because the 10 tasks happen to differ visually in ways that dominate the inter-task distance matrix on trajectories. Under this interpretation, a demonstration suite with visually-similar tasks differing only in instruction semantics would show different M3 results.

This interpretation is partially testable. The mechanistic argument (§I.3, §VI.4 of this section) predicts that demonstration-trajectory correlation depends on the teleop protocol — operators executing motor commands based on visual perception, with language as label rather than driver — not on task visual distinctiveness. Within the present work, we cannot conclusively distinguish this interpretation, since all our data comes from LIBERO-Spatial. We note however that the §V.A.5 Test 4 backbone-capacity check rules out the related concern that high $\rho^2$ at the backbone reflects visual rather than language structure: backbone-level $h_{\mathrm{OFT}}$ correlates with text-token-derived language embeddings (Gate 1 protocol), and the F-statistic on action prediction confirms the backbone can produce task-discriminative actions when explicitly probed.

**Alternative interpretation 2: The Gaussian MI surrogate underestimates the true MI, and the actual demonstration-level language information is substantial.** The surrogate $\hat{I} = -\tfrac{1}{2}\log(1-\rho^2)$ assumes joint Gaussianity. If demonstration trajectory and language are dependent through nonlinear or non-Gaussian structure, the Mantel correlation $\rho$ underestimates the dependence, and $\hat{I}$ correspondingly underestimates the true MI.

This is a real limitation acknowledged in §III.B Caveat on Gaussian surrogate and §V.C.4. We report the squared correlation $\rho^2$ as the primary effect size, with the MI surrogate as qualitative reference. The interpretation we offer rests on the *gap* between backbone-level and demonstration-level squared correlation ($8\times$ ratio), which is invariant to monotone transformations of the underlying MI: if the Gaussian surrogate underestimates true MI, it likely underestimates it at both backbone and demonstration levels, preserving the ratio. The substantive claim — that language conditioning attenuates between backbone and demonstration — is a relative observation that does not depend on absolute MI estimates.

**Alternative interpretation 3: Theorem 3' is correct as stated but vacuous, because no realistic policy achieves the bound.** It is logically possible that demonstration-supervised policies all achieve $I(A^*; \ell \mid o)$ far below the bound, making the bound non-binding and uninformative. This would render Theorem 3' a theoretical curiosity rather than an explanatory tool.

Our empirical findings argue against this. The Gate 1 measurement shows the backbone has $\rho_h^2 = 0.578$, equivalent to a Gaussian MI surrogate of $\sim 0.43$ nats. If the bound were vacuous, policy-level language MI would substantially exceed the demonstration bound of $\sim 0.04$ nats by leveraging the backbone's preserved language signal at no cost. The C1 measurement on OTP-Soft V3 instead shows $r(z, \ell) = -0.19$ — policy latent inherits the demonstration-level near-orthogonality, not the backbone-level strong correlation. The bound appears to be binding in practice.

## §VI.2 Pushback 1 — Mantel test setup validity

A skeptical reviewer may question the Mantel test methodology underpinning Gate 1, M3, and C1: (a) the choice of distance metric (Euclidean) on potentially high-dimensional representations; (b) the use of mean-pooled language embeddings as the language-side distance matrix; (c) the 10-task LIBERO-Spatial set as a small-N inter-task distance test.

We address these in turn:

**(a) Distance metric choice.** Euclidean distance on per-task mean representations is standard for representation-distance probing (Kriegeskorte et al., 2008, representational similarity analysis). The Mantel test is invariant to monotone transformations of the distance matrices, so cosine distance produces qualitatively identical patterns; we report Euclidean for consistency with prior probing literature.

**(b) Language embedding choice.** We use mean-pooled token embeddings from the OpenVLA-OFT tokenizer applied to the raw instruction prompt. This choice keeps the language-side measurement aligned with what the backbone-internal text-processing pipeline produces; alternative choices (CLIP text encoder, BERT pooling, instruction-tuned sentence embeddings) would test slightly different language representations. The SC2 sanity check (§V.A.2) verifies our pipeline is deterministic and consistent between Gate 1 and C1 measurements.

**(c) Inter-task N = 10.** The Mantel test with $N = 10$ tasks produces $\binom{10}{2} = 45$ inter-task pairs and an effective sample size suitable for permutation testing with $N_{\text{perm}} = 1000$. The replicated Gate 1 result ($\Delta r = 0.0152$ between two independent runs of the same protocol on the same data) is well within sampling noise for this $N$; we estimate sampling noise empirically rather than relying on asymptotic distributional assumptions. The replication agreement supports the robustness of the protocol at this sample size.

## §VI.3 Pushback 2 — How does any VLA architecture achieve high SR if Theorem 3' applies universally?

A natural objection to Theorem 3' is that it appears too strong: if all demonstration-supervised architectures are bounded by demonstration-level language MI, and demonstrations are near-language-orthogonal, then no demonstration-supervised policy should perform well at language-conditioned tasks. Yet OFT achieves 97% SR. How does this fit?

The answer is in §V.C.5's two-scenario framing. OFT's 97% SR is achieved consistently with the bound: under scenario (a), OFT carries small but non-zero language MI within the bound, sufficient for task discrimination; under scenario (b), OFT carries zero language MI and achieves SR via observation-action mapping alone. Both scenarios are consistent with the bound and the §V.B paraphrase invariance pattern.

The conceptual error a reader might fall into is identifying "task success" with "language conditioning use." Task success only requires the policy to produce correct actions given the observation; it does not require the language instruction to causally drive action selection. In LIBERO-Spatial (and many similar benchmarks), tasks are visually distinctive — each task has a unique object layout that can be discriminated from observation alone. A policy that maps each visual scene to its correct action will achieve high SR without using language. Theorem 3' bounds the language-driven contribution to policy actions; it does not bound the SR-driven contribution from observation processing.

We argue this explains why the standard SR + paraphrase-invariance evaluation regime cannot detect the distinction. Both visually-grounded (b) and weakly-language-grounded (a) policies produce the same SR profile under standard evaluation. Signal-flow diagnostics (Gate 1, M3, C1) are needed to distinguish.

## §VI.4 Pushback 3 — Auxiliary language losses as bypass route

We claim Theorem 3' is architecture-agnostic for any demonstration-supervised policy, and that bypassing the bound requires interventions outside the demonstration loss (§III.B Bridge implication 2; §IV.A; §IV.C). A reviewer may ask: what specifically counts as "outside the demonstration loss," and can it actually achieve the bypass?

Several auxiliary objectives have been proposed in adjacent literature that, under our framework, would modify the Markov chain underlying Theorem 3' and thereby alter the bound:

- **Language reconstruction loss.** Training the policy to additionally reconstruct the instruction from intermediate representations (Lynch et al., 2020-style multi-task objectives) introduces a direct $\ell$-supervision signal that does not flow through demonstration trajectories. Under our framework, this modifies the supervision distribution from $p_{\mathcal{D}}(o, \ell, a)$ to a joint $p_{\mathcal{D}}(o, \ell, a) \cdot p_{\text{aux}}(\ell, z)$, lifting the Theorem 3' bound on $I(A^*; \ell \mid o)$ by the auxiliary signal's information.

- **Contrastive language-representation alignment.** Aligning policy hidden states with backbone-derived language embeddings via contrastive objectives (CLIP-style training, but for VLA hidden states) introduces a non-demonstration supervision pathway. Backbone-preserved language information (Gate 1 $r = 0.76$) becomes available to the policy through alignment rather than through demonstration mediation.

- **Inference-time conditioning on backbone signals.** At test time, augmenting policy inference with explicit backbone hidden state injection (analogous to retrieval-augmented generation in NLP) bypasses the demonstration-trained policy's bound by drawing on the backbone's preserved language signal directly.

These interventions are all *outside* the demonstration-supervised loss in the sense Theorem 3' targets. Our paper does not implement or empirically evaluate any of them — they are pointers to a research direction informed by Theorem 3', not solutions delivered by it.

A common potential confusion: does *fine-tuning* the backbone on demonstrations count as outside the demonstration loss? No — fine-tuning incorporates the backbone into the demonstration-supervised optimization, making it subject to Theorem 3' just as the action head is. The Gate 1 measurement on the *frozen* OpenVLA-OFT backbone evidences the preserved language signal; fine-tuning that backbone on demonstrations would degrade Gate 1 toward the demonstration-level signal magnitude.

## §VI.5 Pushback 4 — Single-benchmark scope

A natural pushback to our diagnostic findings is single-benchmark scope: Gate 1 ($r(h_{\mathrm{OFT}}, \ell) = 0.76$), M3 ($r \in [-0.27, -0.20]$), and the OFT contrast (§V.C.5) are all demonstrated on LIBERO-Spatial. We separate the theoretical and empirical components of our scope claim.

**Theoretical scope.** Theorem 3' bounds $I(A^*; \ell | o) = I_{p_{\mathcal{D}}}(a; \ell | o) \le \varepsilon$. The right-hand side depends only on the joint distribution of the supervision data $p_{\mathcal{D}}(\ell, a | o)$, not on the specific task suite. The bound is benchmark-agnostic; the question is whether the antecedent $I_{p_{\mathcal{D}}}(a; \ell | o) \approx 0$ holds across benchmarks.

**Empirical scope.** We expect demonstration-trajectory ⊥ language to hold broadly under standard teleoperated demonstration protocols, on a mechanistic argument: teleoperators visually perceive the scene and execute motor commands, with language serving as a post-hoc label affixed to the recorded trajectory rather than as a causal driver of the recorded actions. Under this protocol, the joint distribution $p_{\mathcal{D}}(\ell, a | o)$ factors approximately as $p(\ell | o) \cdot p(a | o)$, yielding the language-orthogonality condition independently of task suite.

This mechanistic prediction is empirically testable. We provide one quantitative instantiation (LIBERO-Spatial, $r = -0.27$); replication on LIBERO-Goal, LIBERO-Object, and non-LIBERO teleoperated suites (e.g., RoboCasa, Bridge) is left to future work. We do not claim universal cross-benchmark generalization, but we do make a falsifiable mechanistic prediction: any teleop-collected suite should yield $|r(\text{traj}, \ell)| \ll |r(h_{\mathrm{backbone}}, \ell)|$ under the same M3 protocol.

## §VI.6 Future work

Three directions follow from this work:

**1. Cross-benchmark empirical validation of M3.** Applying the M3 protocol to LIBERO-Goal, LIBERO-Object, LIBERO-90, RoboCasa, and Bridge would test the mechanistic prediction of §VI.5 across suites differing in task structure, instruction complexity, and demonstration source. If the prediction holds, the §VI.5 caveat strengthens to a general empirical claim about teleop-collected VLA training data. If it fails on some suite, the failure would itself be informative — identifying which protocol features matter for demonstration-level language conditioning.

**2. Auxiliary-objective interventions to bypass Theorem 3'.** The three intervention classes identified in §VI.4 (language reconstruction, contrastive alignment, inference-time conditioning) admit concrete experimental designs. The most promising near-term direction is contrastive alignment of policy hidden states with frozen backbone language embeddings — this leverages the Gate 1 result (backbone preserves language) without requiring new demonstration collection. We expect this to materially lift the Theorem 3' bound on policy-level language MI; the empirical question is whether the resulting policy achieves higher language-driven SR distinguishable from visually-grounded high SR under signal-flow diagnostics.

**3. Diagnostic-methodology adoption in VLA evaluation.** Beyond the present paper's findings, we propose that Gate 1, M3, and C1 measurements be reported alongside standard SR + paraphrase-invariance results in future VLA evaluations. The three measurements are cheap (each takes < 1 hour on a single GPU); they require only access to backbone hidden states, demonstration trajectories, and policy latents, which are available for nearly all published VLA architectures. Routine reporting would localize language-conditioning behavior across the field and inform whether SR improvements correspond to genuine language conditioning gains or to better visual grounding.

## §VI.7 Limitations

We acknowledge five limitations explicitly:

1. **Single benchmark.** §VI.5 discusses the scope caveat in detail. The empirical findings rest on LIBERO-Spatial; cross-benchmark validation is future work.

2. **Single backbone.** Gate 1 measurements use the OpenVLA-OFT backbone exclusively. Other VLAs (RT-2, π₀, generic LLM-based VLAs) might show different Gate 1 magnitudes. We have not measured. The theoretical bound applies regardless of backbone; only the empirical instantiation is backbone-specific.

3. **Gaussian MI surrogate.** The quantitative MI estimates ($\hat{I}_h \approx 0.43$, $\hat{I}_{\mathrm{demo}} \approx 0.04$) rely on a Gaussian surrogate and should be read as qualitative effect-size references rather than tight MI ceilings. The primary effect-size finding is the squared-correlation ratio.

4. **OTP-Soft architectural complexity.** OTP-Soft was originally designed under a working hypothesis subsequently falsified (§IV.C). It serves the present work as a diagnostic platform rather than as an optimal architectural recommendation. Readers interested in implementing the diagnostic methodology should adapt it to simpler VLA architectures (OFT, RT-2, π₀) for which the architectural overhead is lower.

5. **OFT scenario disambiguation.** §V.C.5 leaves open which of two scenarios (low-magnitude language preservation vs entirely visual grounding) describes OFT's high SR. Distinguishing requires either an OFT-specific signal-flow diagnostic measurement (which we have not performed) or controlled architecture experiments aligned with OFT's internal structure (future work).

These limitations bound the strength of our claims to: a theoretical bound with broad applicability under stated conditions, one quantitative empirical instantiation, and a diagnostic methodology proposed for broader adoption. The paper's contribution is methodological and theoretical rather than a complete empirical resolution of the language-conditioning question in VLAs.
