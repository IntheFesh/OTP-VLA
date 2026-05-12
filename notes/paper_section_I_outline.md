# §I Outline — Introduction + Hook

**Status**: Outline (anchor for ~2 hr writing session). NOT a full draft.

**Frozen 2026-05-12 evening** after Theorem 3' v3 + §V.C v4 + §V.C.5 two-scenario framing all locked.

## Hook (1-2 paragraphs, MUST land hard)

**Opening question**: Vision-Language-Action models achieve $\sim 97\%$ task success on LIBERO and similar benchmarks. Do they *understand* the language instruction, or do they exploit visual cues correlated with instructions in training data?

**Quantitative answer (this paper)**:

A three-layer signal-flow diagnostic on a strong VLA backbone reveals:

1. **Backbone preserves language**: $r(h_{\mathrm{backbone}}, \ell) = 0.76$ ($p < 0.001$) on LIBERO-Spatial. The backbone hidden state at the last text token correlates strongly with instruction embedding across tasks.

2. **Demonstration data does not encode language conditioning**: $r(\text{demo trajectory}, \ell) = -0.27$ ($p = 0.96$) — demonstrations are approximately language-orthogonal under Mantel testing.

3. **Trained policies are bounded**: Theorem 3' (§III.B) shows that under standard distribution-matching losses, $I(A^*; \ell | o) \le I_{p_{\mathcal{D}}}(a; \ell | o)$. The bound is architecture-agnostic — applies to deterministic regression, CFM, cross-entropy, and any other demonstration-supervised policy.

**The contrast is the hook**: backbone-level language correlation $0.76$ vs. demonstration-level $-0.27$ is a hard-to-ignore signal that the **information loss happens during demonstration supervision**, not during pretraining. High SR is then achieved despite — not because of — language conditioning at the policy level.

## §I.1 — Background (~0.5 page)

VLA models combine pretrained vision-language backbones (CLIP, SigLIP+DINOv2, LLaMA-family) with action heads trained on robot demonstrations. State-of-the-art VLAs (OFT, RT-2, PI-0) achieve high task SR on LIBERO, RoboCasa, Bridge.

The dominant evaluation axis is task SR + paraphrase invariance (model robust to instruction rewording). This is interpreted as "the model understands the instruction."

**This paper questions that interpretation.**

## §I.2 — Contributions (~0.5 page, bullet form OK here despite paper style)

1. **Signal-flow diagnostic methodology**: Gate 1 (backbone-level language correlation), M3 (demonstration-level language-trajectory correlation), C1 (policy-level latent-language correlation). Together they localize where in the VLA stack language conditioning is preserved or lost.

2. **Theorem 3' (§III.B)**: Formal bound on policy-level language MI by demonstration-level language MI. Architecture-agnostic. Reformulated to address subject ambiguity (sampled action $A^*$ vs distribution-valued policy $\pi^*$).

3. **Empirical instantiation (§V)**: On LIBERO-Spatial, $\rho^2_{\text{backbone}} / \rho^2_{\text{demo}} \approx 8\times$ separation. Demonstrates the bound is **near-binding** on standard benchmarks.

4. **Diagnostic insight on OFT (§V.C.5)**: OFT's 97% SR + paraphrase invariance is consistent with two scenarios — low-magnitude language preservation OR entirely visual-grounded. Benchmark SR alone cannot distinguish; signal-flow diagnostics can.

5. **CFM decoder failure characterization (§V.C.2-3)**: CFM-based decoders exhibit sample-quality collapse independent of upstream head behavior. Documented via training-set sample-quality test (L1/std ratio 1.88, 7/7 dims fail) and oracle ablation (gt trajectory does not improve decoder loss).

## §I.3 — Scope (~0.25 page)

Demonstrated on LIBERO-Spatial. Theoretical bound (Theorem 3') benchmark-agnostic. Empirical mechanism-based prediction (§VI.4): any teleop-collected suite should yield $|r(\text{traj}, \ell)| \ll |r(h_{\text{backbone}}, \ell)|$.

## §I.4 — Paper organization (~0.25 page)

Standard "§II reviews related work, §III formalizes ..., §IV describes architecture, §V presents diagnostic findings, §VI discusses ..." paragraph.

## Critical framing musts (locked decisions)

- ✅ Lead with the contrast: $0.76$ vs $-0.27$
- ✅ Frame contribution as "diagnostic methodology + theoretical formalization + empirical validation", NOT "architectural fix"
- ✅ Acknowledge: paper does NOT propose a fix that bypasses Theorem 3' bound
- ❌ DO NOT promise an "architectural recovery" in the intro (paper §V.D no longer pursues this)
- ❌ DO NOT use language like "we solve the language conditioning problem" (overclaim; the paper diagnoses, doesn't solve)

## Word budget: ~2 pages target

## Title

Catchy: "Where Did the Language Go? Tracing Conditioning Signal Loss in VLA Policies"
Conservative: "Diagnostic Framework for Language Conditioning in Vision-Language-Action Policies"

**Recommend**: Catchy title (Anthropic-published RA-L papers tend toward this style). Reviewer #2 minor concern: "where did" implies the language *was* there and got lost — but this is exactly the paper's finding (backbone has it, demo loses it). The title is empirically accurate.

## Connection to other sections

- → §II Related Work (CFM-image probing, VLA-arch evaluation, language-grounding probing)
- → §III.A C2/C3 conditions (setup for Theorem 3')
- → §III.B Theorem 3' (formalization of the hook claim)
- → §IV (architecture — both ours and OFT, for §V.C.5 contrast)
- → §V.A-C (the empirical findings the intro previews)

---

