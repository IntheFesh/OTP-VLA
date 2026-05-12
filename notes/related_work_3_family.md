# Related Work — 3 Family Outline

**Status**: P1 outline (writing anchor for §II Related Work section, ~1 hr writing budget after §IV/§I).

**Three families to cover** (per userMemories #21 P1 audit):

## Family 1: CFM in image generation (CFM-image)

**Relevance**: Establishes CFM as well-studied generative framework for distribution matching. We use SFM (variant) in our decoder. Reviewer will want to see this lineage.

**Key papers to cite**:
- **Lipman et al. 2023** — Original Conditional Flow Matching paper
- **Liu et al. 2022** — Rectified Flow (relevant variant)
- **Tong et al. 2023** — Improving CFM with simulation-free training
- **(if applicable) Shortcut Flow Matching paper** — Direct citation for our head/decoder framework

**Framing**: CFM family has demonstrated state-of-the-art in image generation (Stable Diffusion 3, FLUX). Application to robot action sequences is more recent (PI-0, OTP family). **Our finding (§V.C.2) shows demonstration-supervised CFM has fundamental sample-quality issues distinct from image-generation CFM** — image CFM has cleaner ground-truth distributions; robot demonstrations have language-orthogonal action distributions that CFM cannot leverage.

**Critical not-claim**: Do NOT claim "CFM doesn't work for robots" generally — vanilla CFM-action might work under different supervision regimes (e.g., language-rich captions). Our finding is specific to **demonstration-supervised** CFM under language-orthogonal demos.

## Family 2: VLA architecture (VLA-arch)

**Relevance**: Position our architectural choices against existing VLA designs.

**Key papers to cite**:
- **OpenVLA** (Kim et al. 2024) — backbone lineage
- **OpenVLA-OFT** (Kim et al. 2024 follow-up) — direct backbone we use; OFT contrast in §V.C.5
- **RT-2** (Brohan et al. 2023) — language-action coupling, tokenization-based
- **PI-0** (Black et al. 2024) — flow-matching VLA, large-scale
- **Octo** (Octo team 2024) — diffusion-based action head
- **OTP / OTP-VLA family** — prior pose-trajectory work this paper builds on
- **(if applicable) Diffusion Policy** — non-CFM diffusion baseline

**Framing**: VLA architecture has converged around "pretrained vision-language backbone + small action head/decoder". Variations in decoder design (deterministic regression vs CFM vs diffusion vs tokenization) have been treated as engineering choices. **Our finding (§V.C.5 + Theorem 3' Remark 2) shows these decoder choices have predictable failure-mode differences within the same theoretical bound**.

**Comparison table** (proposed for §II.B):

| Architecture | Decoder family | Action loss | Status under Theorem 3' |
|---|---|---|---|
| OpenVLA, RT-2 | Cross-entropy on discretized actions | CE | Bounded; visual-grounded SR possible |
| OFT | Deterministic regression | L2 | Bounded; 97% SR via visual grounding (§V.C.5) |
| PI-0 | Flow matching (vanilla CFM) | CFM | Bounded; SR via either path |
| OTP-Soft (this paper) | Shortcut Flow Matching | SFM | Bounded; **CFM-family sample-quality collapse documented (§V.C.2)** |
| Diffusion Policy | Denoising diffusion | DDPM loss | Bounded; failure mode unanalyzed |

## Family 3: Language grounding probing (probing)

**Relevance**: Methodologically closest. Probing studies measure where in a neural network language information is preserved.

**Key papers to cite**:
- **Tenney et al. 2019** — BERT probing (canonical reference for layer-wise probing methodology)
- **Belrose et al. 2023** — Logit Lens / Tuned Lens (representation-level probing for LLMs)
- **(any VLA-specific probing work)** — e.g., probing CLIP for compositional understanding
- **Hewitt & Manning 2019** — Structural probing (Mantel-test-like methodology in NLP)
- **(if applicable) Manning et al. various** — Probing classifier methodology

**Framing**: Probing studies in NLP measure information at internal representations of neural networks. **Our methodology extends this to the supervised data itself (M3) and to the policy's roll-out behavior (C1)** — not just internal representations. The novel contribution methodologically is bridging:

1. Backbone representation probing (Gate 1, similar to NLP probing)
2. **Supervision data probing (M3, novel — measures correlation in training data)**
3. Policy behavior probing (C1, also extends probing methodology)

The M3 step is the novel methodological contribution — we are not aware of prior VLA work that measures language-trajectory correlation directly on the demonstration data.

## §II overall budget: ~1.5 pages

- §II.A CFM in generative modeling: ~0.4 page
- §II.B VLA architecture (with comparison table): ~0.6 page
- §II.C Language grounding probing: ~0.4 page
- Transition to §III: ~0.1 page

## Critical framing musts

- ✅ Position our work as **methodological + theoretical contribution**, not architectural
- ✅ M3 (demo-data probing) is the novel methodological piece — emphasize
- ✅ §V.C.5 OFT contrast positions us as "diagnosing existing strong architecture" not "competing with it"
- ❌ DO NOT claim our architecture is novel/better — it's a vehicle for the diagnostic study
- ❌ DO NOT cite work we haven't actually read — placeholders OK in outline, must be verified at writing time

## Bibliography action items

This outline lists ~15 candidate citations. Before writing §II:
1. Verify each paper exists + is correctly attributed (use web_search if uncertain)
2. Read abstract + key sections for each (not full paper)
3. Note which papers we *must* cite vs which are nice-to-have (Tier 1 vs Tier 2)
4. Flag any citations Yue has not read directly — Claude can help with abstract-reading via web_fetch

**Tier 1 must-cite** (paper viability):
- Lipman 2023 (CFM)
- OpenVLA + OpenVLA-OFT (architecture lineage + §V.C.5 contrast)
- Tenney 2019 or similar (probing methodology lineage)

**Tier 2 nice-to-have**:
- Everything else above

---

# End of outlines

## Recommended writing order

1. **§IV.A + §IV.C** (architecture-agnostic framing + retroactive motivation, ~30 min)
2. **§IV.B** (architecture details, mostly factual, ~30 min)
3. **§I hook + §I.1 + §I.2** (intro + contributions, ~1.5 hr)
4. **§I.3 + §I.4** (scope + organization, ~0.5 hr)
5. Defer §II Related Work to Day 3 (P1, full reading required)

Total: §IV ~1 hr, §I ~2 hr, §II deferred. Today P0 finishable.
