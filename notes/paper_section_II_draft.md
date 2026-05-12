# §II — Related Work (v1 draft)

**Status**: Draft v1. Day 3 P1. ~1.5 pages target.

**3 families covered**:
1. CFM and flow matching in generative modeling (~0.4 page)
2. Vision-Language-Action architectures (~0.6 page, with comparison table)
3. Language-grounding probing methodology (~0.4 page)

**Citation status (verified Day 2-3)**:
- ✅ Lipman et al., 2023 — Flow Matching for Generative Modeling (ICLR 2023)
- ✅ Liu et al., 2022 — Rectified Flow
- ✅ Frans et al., 2024 — Shortcut models (arXiv:2410.12557)
- ✅ Kim et al., 2024 — OpenVLA (arXiv:2406.09246, CoRL 2024)
- ✅ Kim et al., 2025 — OpenVLA-OFT (arXiv:2502.19645)
- ✅ Black et al., 2024 — π₀ flow model (arXiv:2410.24164, RSS 2025)
- ⏳ Brohan et al., 2023 — RT-2 (well-known, defer batch validate)
- ⏳ Driess et al., 2023 — PaLM-E (well-known)
- ⏳ Tenney et al., 2019 — BERT probing
- ⏳ Hewitt and Manning, 2019 — structural probing
- ⏳ Belrose et al., 2023 — Tuned Lens
- ⏳ Octo team, 2024 — Octo diffusion policy
- ⏳ Chi et al., 2023 — Diffusion Policy

---

## §II.A Conditional Flow Matching and shortcut variants

Conditional Flow Matching (Lipman et al., 2023; Liu et al., 2022) is a generative modeling framework that learns a continuous-time velocity field transporting samples from a simple source distribution (typically isotropic Gaussian) to the target data distribution conditioned on auxiliary context. Unlike score-based diffusion models, CFM directly parameterizes the velocity field rather than the score function, yielding simpler training objectives and tighter connections to optimal transport. The CFM training loss
$$\mathcal{L}_{\text{CFM}}(\theta) = \mathbb{E}_{t,\,x_0,\,x_1}\!\left[\| v_\theta(x_t, t \mid c) - (x_1 - x_0) \|^2\right]$$
where $x_t = (1-t) x_0 + t x_1$ and $c$ is the conditioning signal, admits the population minimizer $v^*(x_t, t \mid c) = \mathbb{E}_{x_1 \sim p_{\mathcal{D}}(\cdot|c)}[x_1 - x_0 \mid x_t, t, c]$ recovering the matched-flow velocity field; the induced sampling distribution at $t=1$ matches $p_{\mathcal{D}}(x_1 \mid c)$ at convergence. This recovery property places CFM within the family of distribution-matching losses to which Theorem 3' (§III.B, Remark 2) applies.

A practical limitation of CFM is that high-quality sampling typically requires many Euler integration steps. Shortcut models (Frans et al., 2024) address this by additionally conditioning the velocity network on a desired step size, enabling few-step or single-step sampling via a self-consistency objective that ties large-step prediction to composed small-step trajectories. The Shortcut Flow Matching variant (the head and decoder loss of OTP-Soft, §IV.B) inherits the Lipman-style distribution-matching structure while accelerating training-time convergence relative to vanilla CFM.

Within image generation, CFM and its variants underpin state-of-the-art models (Stable Diffusion 3, FLUX) and achieve high sample quality on language-conditioned generation. Applications to robot action sequences are more recent: π₀ (Black et al., 2024) uses vanilla flow matching for action prediction in a generalist robot policy, and several follow-up works (including OTP-Soft in this paper) explore variants. Our finding (§V.C.2) is that the demonstration-supervised setting introduces a failure mode not encountered in image-generation CFM: under language-orthogonal demonstration distributions, the matched-flow representation cannot encode language-discriminative action distributions because such distributions are absent in the supervision data. This is a setting-specific failure rather than an architectural defect of CFM as a generative framework.

## §II.B Vision-Language-Action architectures

Modern VLAs are built by integrating a pretrained vision-language model (VLM) with an action head trained on robot demonstration datasets (Brohan et al., 2023; Kim et al., 2024). The pretrained VLM contributes broad semantic and visual knowledge from internet-scale data; the action head learns to map VLM hidden states to robot control commands using teleoperated demonstrations as supervision. This decomposition allows VLAs to inherit the language-following capability of the underlying VLM while specializing to embodied control tasks.

VLA designs differ primarily in the action head architecture and loss family. We summarize representative architectures and their relationship to Theorem 3' below:

| Architecture | Action head | Action loss family | Status under Theorem 3' |
|---|---|---|---|
| OpenVLA (Kim et al., 2024) | Autoregressive token output | Cross-entropy on discretized actions | Bounded (Remark 2) |
| OpenVLA-OFT (Kim et al., 2025) | Parallel decoding head | $L_1$ regression on continuous actions | Bounded (Remark 2) |
| RT-2 (Brohan et al., 2023) | Co-fine-tuned VLM tokens | Cross-entropy on discretized actions | Bounded (Remark 2) |
| π₀ (Black et al., 2024) | Flow matching action head | Vanilla CFM | Bounded (Remark 2) |
| Octo (Octo team, 2024) | Diffusion action head | Denoising score matching | Bounded (Remark 2) |
| Diffusion Policy (Chi et al., 2023) | Conditional DDPM | Denoising score matching | Bounded (Remark 2) |
| OTP-Soft (this paper, §IV) | Object-pose head + language-agnostic decoder | Shortcut Flow Matching | Bounded (Remark 2) |

The diversity of action-head choices is typically motivated by efficiency considerations (parallel decoding versus autoregressive, continuous versus discretized actions) or by representational flexibility (flow matching and diffusion permit multi-modal action distributions; cross-entropy on discretized actions cannot). Comparative studies have evaluated these design choices primarily through task success rate on benchmark suites (LIBERO, RoboCasa, Bridge), often using paraphrase invariance as a secondary axis interpreted as evidence of "language understanding" (Kim et al., 2025).

Our work makes a different observation: under Theorem 3' (§III.B, Remark 2), all of the architectures in the table above are subject to the same bound $I(A^*; \ell \mid o) \le I_{p_{\mathcal{D}}}(a; \ell \mid o)$, because each uses a distribution-matching loss whose population minimizer recovers $p_{\mathcal{D}}(\cdot \mid o, \ell)$. The architectural choice does not change this bound; it changes the empirical character of policy behavior when the bound is tight. In §V.C.5 we discuss this in detail for the OFT contrast: OFT's 97% SR on LIBERO-Spatial is consistent with two scenarios within the bound (low-magnitude language preservation or entirely visual grounding), neither of which can be distinguished by benchmark SR alone.

## §II.C Language-grounding probing methodology

Probing studies (Tenney et al., 2019; Hewitt and Manning, 2019) measure where linguistic structure is encoded in the internal representations of a neural network. The canonical setup trains a lightweight classifier on top of fixed model representations to predict linguistic targets (part-of-speech tags, syntactic dependency structure, semantic roles); high classifier accuracy is interpreted as evidence that the underlying representation encodes the target. More recent variants (Tuned Lens, Belrose et al., 2023; Logit Lens) probe layer-wise predictions in large language models without auxiliary classifier training.

In language-conditioned robotics, probing has been applied to learned policies for instruction sensitivity: e.g., measuring policy behavior under input perturbations, attribution analyses on attention weights, or representation-level inspection of policy hidden states. The shared assumption is that the policy is the object of inquiry — probing happens *after* training to assess what the model learned.

Our diagnostic methodology departs from this convention in one substantive way. We retain backbone-level probing (Gate 1, §V.A: a Mantel test on backbone hidden states versus language embeddings, analogous to standard NLP representation probing), but we additionally probe the **supervision data itself** (M3, §V.A: a Mantel test on demonstration trajectories versus language embeddings, prior to and independent of any specific policy). To our knowledge, this protocol — probing the language-trajectory correlation structure of the demonstration dataset directly — has not previously been applied in the language-conditioned robotics literature. The novelty is methodological rather than technical: the Mantel test is standard, but its application to demonstration data as a diagnostic layer separate from model-internal probing reveals where language conditioning is preserved (backbone) versus lost (demonstration data), prior to any policy being trained.

We also retain policy-level probing (C1, §V.A: Mantel test on policy latent representations versus language embeddings, analogous to standard policy probing), so that the three-layer diagnostic (Gate 1 / M3 / C1) localizes language information at each stage of the VLA pipeline. The relationship between these three levels is then formalized by Theorem 3' (§III.B).

## §II.D Transition to §III

The next section formalizes the conditions on demonstration data (Definition 1 in §III.B) and the bound on policy-level language conditioning (Theorem 3' in §III.B) that organize the empirical findings of §V. §III.A introduces the C2 and C3 conditions on policy architecture that underlie the analysis, and §III.B states and proves Theorem 3'.


