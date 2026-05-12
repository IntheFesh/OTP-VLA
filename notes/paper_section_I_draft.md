# §I — Introduction (v2 final)

**Status**: Draft v2 final. Frozen 2026-05-12 evening after 5-fix reviewer audit on §I v1.

**Reviewer fixes integrated** (v1 → v2):
1. **Hook**: "97%" → "up to 97%" (numerical precision; OFT 97.1% is 4-suite LIBERO average, not single-benchmark single-number)
2. **Contribution 1**: Reframe "novel methodological piece" with proper scope ("To our knowledge, prior probing literature in language-conditioned robotics..."), fix reference class (NLP probing citations belong in §II, not §I.2)
3. **Contribution 2**: Remove "earlier formulations" reformulation framing (paper-evolution leakage); keep only forward-looking "modest math + substantive bridge" statement
4. **Contributions 4+5**: Split single conflated Contribution 4 into separate Contributions 4 (diagnostic insight) and 5 (CFM characterization) — aligns with outline intent, lets each contribution stand alone
5. **§I.3**: Soften "any teleop suite" universal claim to falsifiable-on-any-teleop-suite framing (one quantitative confirmation, broader validation = future work)

**Citations verified** (Day 2 batch):
- ✅ Kim et al., 2024 — OpenVLA (arXiv:2406.09246, CoRL 2024)
- ✅ Kim et al., 2025 — OpenVLA-OFT (arXiv:2502.19645, Feb 2025)
- ✅ Black et al., 2024 — PI-0 (verified via web_search; achieves 94.2% on LIBERO per OFT comparison)
- ⏳ Driess et al., 2023 — PaLM-E (defer to §II batch)
- ⏳ Karamcheti et al., 2024 — Prismatic VLMs (defer to §II)
- ⏳ Brohan et al., 2023 — RT-2 (defer to §II)
- ⏳ Tenney et al., 2019 — BERT probing (defer to §II; removed from §I.2 per Fix 3)
- ⏳ Hewitt and Manning, 2019 — structural probing (defer; removed from §I.2)
- ⏳ Belrose et al., 2023 — tuned lens (defer; removed from §I.2)

**§V.C.5 alignment**: Sync patch already applied via commit `00d9363` — §V.C.5 末段 already says "CFM decoder is *sufficient*, not *isolated* cause". §I Contribution 5 matches.

---

## §I.0 Hook

Vision-Language-Action models (VLAs) achieve **up to 97%** task success rate on standard simulation manipulation benchmarks (Kim et al., 2025; Black et al., 2024). They are typically evaluated by task SR and paraphrase invariance — the latter interpreted as evidence that the model "understands" instructions rather than memorizing surface forms. Together, these benchmarks suggest that contemporary VLAs have largely solved the language-conditioning problem in robot manipulation.

This paper questions that interpretation. Through a three-layer signal-flow diagnostic on a strong VLA backbone (OpenVLA-OFT) trained on LIBERO-Spatial demonstrations, we find:

$$r(h_{\mathrm{backbone}},\, \ell) \;=\; \mathbf{+0.76} \;\;\;\text{(backbone)}, \qquad r(\text{demo trajectory},\, \ell) \;=\; \mathbf{-0.27} \;\;\;\text{(demonstrations)}.$$

The pretrained backbone preserves language conditioning at strong correlation; the demonstration data on which VLAs are trained is approximately language-orthogonal. **The information loss happens during demonstration supervision, not during pretraining.** Theorem 3' (§III.B) formalizes the consequence: any policy trained by distribution-matching losses on language-orthogonal demonstrations is bounded by the demonstration-level signal — regardless of architecture, regardless of decoder family, regardless of representation size. High task SR is then achieved despite — not because of — language conditioning at the policy level.

## §I.1 Background

VLAs combine pretrained vision-language backbones (Driess et al., 2023; Karamcheti et al., 2024) with action heads trained on robot demonstration datasets (Brohan et al., 2023; Kim et al., 2024; Kim et al., 2025; Black et al., 2024). State-of-the-art models — OpenVLA-OFT, RT-2, PI-0 — achieve high task SR on simulation benchmarks (LIBERO, RoboCasa) and increasingly on real-world bimanual platforms.

The dominant evaluation axis for whether a VLA "understands" language has been task SR combined with paraphrase invariance: a model that succeeds at the task and remains robust under instruction rewording is interpreted as having learned to ground language in action. Recent improvements in VLA architecture — parallel decoding, action chunking, continuous representations (Kim et al., 2025) — have driven SR from ~76% to ~97% on LIBERO-Spatial while preserving paraphrase robustness.

Two assumptions underlie this evaluation regime. First, that task SR and language conditioning are coupled — high SR implies that language is being used. Second, that paraphrase invariance distinguishes "genuine grounding" from "surface memorization." This paper provides evidence that neither assumption holds in the current evaluation paradigm: paraphrase-invariant VLAs achieving 97% SR can do so without policy-level language conditioning, as long as scene structure is sufficient for task discrimination.

## §I.2 Contributions

We make five contributions:

**1. Signal-flow diagnostic methodology (§V.A).** We introduce three measurements that together localize where in the VLA stack language conditioning is preserved or lost:

- **Gate 1**: Mantel correlation $r(h_{\mathrm{backbone}}, \ell)$ between backbone hidden states (last text token) and language embeddings across tasks. Measures backbone-level preservation.
- **M3**: Mantel correlation $r(\text{demonstration trajectory}, \ell)$ across tasks. Measures whether demonstration data encodes language-action correlation.
- **C1**: Mantel correlation $r(z_{\mathrm{policy}}, \ell)$ between policy latent representations and language embeddings. Measures policy-level conditioning.

Among these, **M3 is the principal methodological contribution**. To our knowledge, prior probing literature in language-conditioned robotics focuses on model-internal representations after training (probing learned policies for instruction sensitivity, e.g., via attribution or representation analysis); M3 instead probes the supervision data itself for instruction-trajectory correlation, prior to and independent of any specific policy. We are not aware of analogous protocols applied to robot demonstration datasets in the language-conditioning literature.

**2. Theorem 3' (§III.B).** A formal bound on policy-level language mutual information by demonstration-level language mutual information. For policies $\pi^*$ minimizing distribution-matching losses with population minimizer $\pi^*(\cdot \mid o, \ell) = p_{\mathcal{D}}(\cdot \mid o, \ell)$ a.s., the sampled-action MI satisfies

$$I(A^*; \ell \mid o) \;=\; I_{p_{\mathcal{D}}}(a; \ell \mid o) \;\le\; \varepsilon.$$

The bound is architecture-agnostic: it applies uniformly across $L_2$ regression, $L_1$ regression, Conditional Flow Matching, Shortcut Flow Matching, and cross-entropy on discretized actions. The theorem's mathematical content is modest — the proof reduces to a distributional identity at the population level — but the substance lies in identifying the demonstration-level quantity $I_{p_{\mathcal{D}}}(a; \ell \mid o)$ that controls policy behavior, and in establishing this quantity is empirically small for standard VLA benchmarks (Contribution 3).

**3. Empirical instantiation (§V.C).** On LIBERO-Spatial, the squared-correlation gap between backbone-level and demonstration-level language conditioning is

$$\rho_h^2 \,/\, \rho_{\text{demo}}^2 \;=\; 0.578 \,/\, 0.073 \;\approx\; 8\times,$$

with the demonstration-level correlation $\rho^2_{\text{demo}} = 0.073$ approaching the null-Mantel scale (permutation $p = 0.96$). This is the first quantitative measurement, to our knowledge, of the demonstration-level information bottleneck that constrains downstream VLA policies under Theorem 3'.

**4. Diagnostic insight: high task SR does not establish language conditioning (§V.C.5).** OpenVLA-OFT achieves 97% SR on LIBERO-Spatial with paraphrase invariance (§V.B: 9/10 tasks paired-diff CI within ±2pp; task 5 sensitivity caveat documented). Under Theorem 3', this benchmark performance is consistent with two scenarios: (a) OFT preserves language conditioning at low magnitude within the bound and uses that signal robustly, or (b) OFT is entirely visual-grounded at the policy level with no language signal. Standard task-SR benchmarks cannot distinguish these scenarios; signal-flow diagnostics (Contribution 1) can. This reframes a long-standing claim in the VLA literature: paraphrase-invariant high SR does not imply that the policy uses language information at the action level.

**5. Empirical characterization of CFM-decoder sample-quality collapse (§V.C.2-3).** Under language-orthogonal demonstration supervision, our OTP-Soft V3 architecture (Shortcut Flow Matching head and decoder, §IV) exhibits a distinctive failure mode at the implementation level: 7/7 action dimensions fail the per-dimension sample-quality threshold (§V.C.2, L1/std ratio 1.88 overall). An oracle ablation replacing head output with ground-truth trajectory rules out the moving-target hypothesis (§V.C.3): the failure persists with perfect upstream signal, localizing it to the decoder family under per-sample distribution-matching loss. The CFM decoder is sufficient to produce this failure under language-orthogonal demonstrations; whether it is the isolated cause of the SR gap with OFT requires controlled architecture experiments (§IV.C, future work).

## §I.3 Scope

The empirical findings in this paper are demonstrated on LIBERO-Spatial, a 10-task simulated manipulation benchmark with teleoperated demonstrations. The theoretical bound (Theorem 3') is benchmark-agnostic — it depends only on the joint distribution $p_{\mathcal{D}}(\ell, a \mid o)$ of the supervision data, not on the specific task suite. We expect the empirical phenomenon (demonstration-trajectory $\perp$ language) to hold broadly under standard teleoperated demonstration protocols on the mechanistic argument that teleoperators visually perceive the scene and execute motor commands, with language serving as a post-hoc label rather than a causal driver of recorded trajectories (§VI.4).

This prediction is falsifiable on any teleop-collected suite under the same M3 protocol: it can be confirmed by observing $\lvert r(\text{traj}, \ell)\rvert \;\ll\; \lvert r(h_{\text{backbone}}, \ell)\rvert$, or refuted by observing comparable magnitudes. We provide one quantitative confirmation on LIBERO-Spatial (§V.A); broader empirical validation across suites is left to future work (§VI).

This paper does not propose a new architectural fix. The OTP-Soft architecture we evaluate (§IV) was originally designed under a working hypothesis — since falsified by Theorem 3' — that a language-agnostic decoder boundary would bypass demonstration-induced conditioning collapse. We retain OTP-Soft as the experimental platform on the basis of its diagnostic properties (clean component-ablation structure, decoder-family contrast with OFT, generalizable measurement infrastructure) rather than as a proposed solution. Bypassing Theorem 3''s bound requires interventions outside the demonstration-supervised loss — auxiliary language objectives, contrastive language alignment, inference-time conditioning on backbone-derived signals — and we discuss such directions only as future work (§VI).

## §I.4 Paper organization

§II reviews related work in CFM-based generative modeling, VLA architecture, and language-grounding probing. §III formalizes the setup, introduces the ε-language-orthogonality condition, and proves Theorem 3' (§III.B). §IV describes the OpenVLA-OFT backbone, the OTP-Soft architecture, and the diagnostic value of this architecture for the present study. §V presents the three signal-flow measurements (Gate 1, M3, C1) and the CFM decoder failure characterization. §VI discusses scope, alternative interpretations, single-benchmark caveats, and future directions including auxiliary-objective interventions that could bypass the bound established here.
