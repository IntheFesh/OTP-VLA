# Paper Abstract (v1 draft)

**Status**: Draft v1. Day 3 P1. Target ~200 words.

---

## Abstract

Vision-Language-Action (VLA) models achieve up to 97% task success rate on simulation manipulation benchmarks, with paraphrase invariance interpreted as evidence of language understanding. We question this interpretation. Through a three-layer signal-flow diagnostic on a strong VLA backbone trained on LIBERO-Spatial, we find that backbone hidden states correlate strongly with instruction embeddings ($r = +0.76$, $p < 0.001$) while demonstration trajectories correlate only weakly with the same embeddings ($r = -0.27$, $p = 0.96$) — an approximately $8\times$ squared-correlation gap localizing language information loss to the demonstration-supervision stage rather than to backbone capacity. We formalize this via Theorem 3', an architecture-agnostic upper bound: any demonstration-supervised policy minimizing a distribution-matching loss is bounded by $I(A^*; \ell \mid o) \le I_{p_{\mathcal{D}}}(a; \ell \mid o)$. The bound applies uniformly across L1/L2 regression, Conditional Flow Matching, Shortcut Flow Matching, and discretized cross-entropy heads. OpenVLA-OFT's 97% SR is consistent with two scenarios within the bound — low-magnitude language preservation or entirely visual grounding — neither distinguishable by benchmark SR alone. We document a Conditional Flow Matching decoder sample-quality collapse failure mode (7/7 action dimensions fail under language-orthogonal supervision, oracle ablation rules out moving-target hypothesis), characterizing how decoder family choice determines failure mode within the bound. Our contribution is a diagnostic methodology and theoretical bound; we explicitly do not propose an architectural fix, as bypassing the bound requires interventions outside the demonstration-supervised loss.

---

## Word count
~210 words. RA-L abstract limit typically 200-250 words. Adjust if needed for final format.

## Alternative versions

### Version B — More concise (~150 words)

VLA models achieve 97% task success on LIBERO with paraphrase invariance, suggesting language understanding. We show this interpretation is inadequate. A three-layer signal-flow diagnostic localizes language conditioning to the backbone ($r = +0.76$) while finding demonstration trajectories are approximately language-orthogonal ($r = -0.27$, an $8\times$ squared-correlation gap). Theorem 3' formalizes an architecture-agnostic bound: any demonstration-supervised policy satisfies $I(A^*; \ell \mid o) \le I_{p_{\mathcal{D}}}(a; \ell \mid o)$, applying to L1/L2 regression, CFM, Shortcut FM, and cross-entropy. OFT's 97% SR is consistent with two within-bound scenarios — low-magnitude language preservation or entirely visual grounding — indistinguishable by SR alone. We further document CFM decoder sample-quality collapse (7/7 action dimensions fail under language-orthogonal supervision; oracle ablation rules out moving-target hypothesis). Our contribution is diagnostic methodology and theoretical bound, not an architectural fix; bypassing the bound requires interventions outside the demonstration loss.

### Version C — More expansive (~280 words, more for journal extended format)

[For RA-L extended-format / journal submission; identical content to Version A with each contribution slightly elaborated. Generate on demand if needed.]


