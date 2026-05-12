# §IV Outline — Theory-to-Architecture Bridge

**Status**: Outline (anchor for ~1 hr writing session). NOT a full draft.

**Frozen 2026-05-12 evening** after Theorem 3' v3 reformulation. Framing radically different from §IV v1 (which was Path B "deterministic head bypasses bound" — now known incorrect).

## Purpose of §IV

§IV serves as the bridge between §III.B (theorem) and §V (empirical findings). It must:

1. State what the bound implies for architecture design
2. Explain why CFM-based decoders are particularly vulnerable to the bound's implications
3. Justify why we still investigate the OTP-Soft architecture despite the bound
4. NOT claim any architecture "bypasses" the bound — Theorem 3' is architecture-agnostic

## Section structure (proposed)

### §IV.A — Implications of Theorem 3' for VLA architecture (~0.5 page)

**Key claim**: The bound $I(A^*; \ell | o) \le I_{p_{\mathcal{D}}}(a; \ell | o)$ applies to all demonstration-supervised policies (Remark 2). Architecture cannot reduce this bound; architecture can only determine **failure mode within the bound**.

**Two architectural axes**:

1. **Decoder family** (deterministic regression vs CFM vs cross-entropy): determines what kind of failure manifests when bound is tight. Deterministic regression → may succeed at SR via visual grounding (OFT existence proof). CFM → sample-quality collapse, fails at SR.

2. **Conditioning input richness** (mesh, grasp affordance, proprioception): determines what information the decoder *could* use, independent of language. Component ablations (§V.D) probe sensitivity to these inputs.

**Honest framing**: Architecture cannot make a demonstration-supervised policy "understand language" beyond what demos encode. It can only avoid pathological failure modes within the bound.

### §IV.B — OTP-Soft architecture (~1 page)

**Backbone**: OpenVLA-OFT pretrained (SigLIP+DINOv2 dual-encoder, fused to 6-channel internally; 7.6B params, 92M trainable frozen during our work).

**Head**: OTP head with **Shortcut Flow Matching** (NOT vanilla CFM; `use_shortcut: true`). Produces 8-step trajectory of object poses.

**Decoder**: LanguageAgnosticDecoder with C3 architectural guard ($\ell \notin \text{inputs}(\phi_\theta)$).
- Inputs: trajectory $z$, proprioception (pos 3 + quat 4 + gripper=0 dim 1 = 8D), grasp affordance, object geometry
- Decoder family: also Shortcut Flow Matching (matched to head)
- Explicitly excludes $\ell$ at decoder layer

**Mesh prior**: pose-invariant shape descriptor, closed-set object library; training/test share 5 meshes. Limitation goes to §V Discussion end, NOT §IV.

**Proprioception schema**: gripper dim is structurally zero by design ("gripper not part of proprioception by design; gripper actions predicted as part of 7-DoF output"). Architectural choice, not limitation.

### §IV.C — Why this architecture for the diagnostic study (~0.5 page)

**Honest motivation**: OTP-Soft was originally designed under the Path B framing (deterministic head supposedly "bypasses" demonstration bound). That framing is now known incorrect (§III.B v3, this paper).

**Retroactive justification for the diagnostic study**:
1. OTP-Soft's architectural choices (C3 boundary, $w(o)$ inputs) provide a clean separation between conditioning channels — allows component ablation (§V.D)
2. CFM-based architecture provides a contrast case to OFT's deterministic regression — the architectural delta isolates decoder family as failure-mode driver (§V.C.5)
3. The diagnostic infrastructure built for OTP-Soft (Gate 1, M3, C1 measurements) generalizes to any VLA architecture

**Reviewer-likely question**: "If you knew CFM was bound-vulnerable, why train OTP-Soft at all?"
**Answer in §IV.C**: We didn't know at design time; the bound's implications for CFM were discovered through this work (§V.C.2 sample-quality test + §V.A C1 diagnostic + Theorem 3' formalization). The paper documents both the architecture and the diagnostic insights that arose during its evaluation.

## Critical framing musts (locked decisions)

- ✅ DO say "Theorem 3' applies to all demonstration-supervised architectures"
- ✅ DO say "architectural choice determines failure mode within bound, not the bound itself"
- ✅ DO acknowledge OTP-Soft was designed under earlier (incorrect) framing — honest scientific narrative
- ❌ DO NOT say "deterministic head bypasses bound" (mathematically incorrect, contradicts §III.B Bridge)
- ❌ DO NOT say "OTP-Soft solves CFM failure" (it doesn't; §V.C.2 shows it inherits CFM failure)
- ❌ DO NOT use C-Lipschitz framing for the bound (dropped in §III.B v3; bound has no $C$ factor)

## Word budget: ~2 pages target

## Bibliography references needed (P1)

- OpenVLA-OFT paper (backbone)
- Original OpenVLA paper (lineage)
- Shortcut Flow Matching paper (head + decoder)
- CFM original paper (Lipman et al.) (decoder framework)
- C3-style architectural guards (if any prior work)

## Connection to other sections

- ← §III.B (Theorem 3' bound, Remark 2 loss families, Bridge): provides the constraint
- → §V.A (C1 diagnostic on $z$): tests whether decoder receives task-conditioned signal
- → §V.C.5 (OFT contrast): provides the cross-architecture comparison
- → §V.D (component ablation): tests sensitivity to conditioning inputs

---

