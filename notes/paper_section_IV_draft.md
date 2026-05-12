# §IV — Architecture and Theoretical Grounding (v2 final)

**Status**: Draft v2 final. Frozen 2026-05-12 evening after Theorem 3' v3 reformulation + 4-fix reviewer audit.

**Reviewer fixes integrated**:
1. §IV.A末段 OFT contrast claim weakened (confounded by multiple architectural differences, not isolated decoder-family effect)
2. §IV.B "improves sample quality" claim removed (contradicts §V.C.2 finding)
3. §IV.B `_check_forbidden` mention strengthened (verified to be both construction-time AND runtime guard per `grep` audit 2026-05-12)
4. §IV.C "designed under incorrect hypothesis" framing replaced with positive diagnostic-platform narrative

**Citations verified** (via web_search 2026-05-12):
- OpenVLA: Kim et al., 2024 (arXiv:2406.09246, CoRL 2024)
- OpenVLA-OFT: Kim et al., 2025 (arXiv:2502.19645) — **NOT 2024**
- Shortcut models: Frans et al., 2024 (arXiv:2410.12557) — applied to flow-matching variant
- Flow Matching: Lipman et al., 2023 (ICLR 2023)

---

## §IV.A Implications of Theorem 3' for VLA architecture design

Theorem 3' (§III.B) provides an architecture-agnostic upper bound on the language conditioning of any demonstration-supervised policy:

$$I(A^*; \ell \mid o) \;\le\; I_{p_{\mathcal{D}}}(a; \ell \mid o) \;\le\; \varepsilon.$$

By Remark 2 of §III.B, this bound applies uniformly across the standard space of VLA action-decoder families: $L_2$ regression, $L_1$ regression, Conditional Flow Matching, Shortcut Flow Matching, and cross-entropy on discretized actions. The bound is not bypassable through architectural choice within the demonstration-supervised paradigm; bypassing it requires interventions outside the demonstration loss itself (auxiliary language objectives, contrastive language alignment, or inference-time conditioning on backbone-derived signals — discussed in §VI).

This observation reorients the question that architectural design can answer. The bound determines what is *achievable* in terms of policy-level language conditioning; the architecture determines what failure mode manifests when the bound is tight. We distinguish two architectural axes that operate independently of the bound:

**Decoder family.** The choice between deterministic action regression, flow-matching-based generation, and cross-entropy over discretized actions does not change the bound but does change the empirical character of policy behavior when language signal is unavailable. Deterministic regression heads can still achieve high task SR by relying on visual grounding alone (§V.C.5, OFT existence proof); they appear to "work" at the SR level while satisfying the bound at the language-MI level. Flow-matching-based decoders, in contrast, exhibit a distinctive sample-quality collapse under language-orthogonal demonstrations: the matched-flow representation cannot learn to discriminate among task-conditioned action distributions when the conditioning signal is absent in the supervision data (§V.C.2-3 documents this empirically).

**Conditioning input richness.** Independent of decoder family, the set of inputs supplied to the decoder — observation features, proprioception, mesh-derived geometry, grasp affordance maps, instruction embeddings — determines what information the decoder *could* use to discriminate among tasks. Theorem 3' constrains the conditioning that flows through the *language channel*; it does not constrain conditioning that flows through observation-derived channels. Component ablations (§V.D) systematically remove conditioning inputs to probe the decoder's sensitivity along this axis.

These two axes motivate the architectural design of OTP-Soft, described next, and they motivate the comparative evaluation strategy of §V: OFT contrast as a cross-architecture qualitative comparison (§V.C.5), and component ablations for conditioning-input effects within the OTP-Soft architecture (§V.D).

## §IV.B OTP-Soft architecture

OTP-Soft inherits the OpenVLA-OFT pretrained backbone and replaces the action head with a structured pipeline producing object-pose trajectories followed by language-agnostic action decoding.

**Backbone.** We use the OpenVLA-OFT checkpoint (Kim et al., 2025), a 7.6B-parameter vision-language model with a SigLIP+DINOv2 dual-encoder visual frontend fused to a 6-channel representation internally and a LLaMA-2-derived language model trunk inherited from OpenVLA (Kim et al., 2024). We freeze the backbone (92M trainable parameters under our training configuration are confined to downstream modules). Following the OFT integration pattern, we access internal modules (`vision_backbone`, `projector`, `llm_backbone`) directly rather than through the outer forward path, bypassing OFT's discretized action mask logic.

**Object-pose trajectory head.** A 138.3M-parameter trajectory head consumes backbone hidden states and emits an 8-step trajectory of object poses ($N_{\text{obj}} \times H \times 7$ for $H=8$ horizon steps and 7-DoF pose per object). The head is trained with the shortcut model framework of Frans et al. (2024) applied to the flow-matching variant (denoted Shortcut Flow Matching throughout this paper), chosen at design time for its accelerated convergence property under the shortcut self-consistency objective relative to vanilla Conditional Flow Matching (Lipman et al., 2023). We note that the choice between Shortcut and vanilla CFM does not affect Theorem 3' applicability (Remark 2 of §III.B); §V.C.2 documents that sample-quality collapse occurs under Shortcut Flow Matching despite this design choice. We denote the head's stochastic output as the latent trajectory $z$.

**Language-agnostic decoder.** The decoder $\phi_\theta$ accepts the trajectory $z$, proprioception, grasp affordance maps, and object-geometry features, and emits a 7-DoF action chunk of horizon $H$. By architectural construction, the instruction $\ell$ is excluded from the decoder's input signature: $\ell$ does not appear in `forward()` arguments or as an input to any submodule. This realizes the C3 condition of §III.A at the implementation level: $\ell \notin \text{inputs}(\phi_\theta)$. The exclusion is enforced by a `_check_forbidden` guard that raises `AssertionError` if any forbidden name fragment appears among (i) `forward()` parameter names at module construction, (ii) submodule and parameter names at construction, or (iii) `locals().keys()` and `dir(self)` at every forward pass — providing both static and runtime verification of the C3 boundary.

The decoder also uses Shortcut Flow Matching as its loss family, matching the head. This choice was made for consistency at design time and is empirically consequential: §V.C.2 documents that the decoder inherits sample-quality collapse independently of the head's behavior (§V.C.3 oracle ablation).

**Proprioception schema.** Proprioception is encoded as an 8-dimensional vector: end-effector position (3) + orientation quaternion (4) + a structurally zero gripper dimension (1). The zero gripper dimension is by design — gripper actions are predicted as part of the 7-DoF action output rather than treated as a proprioception input, reflecting the LIBERO Panda gripper's discrete open/close semantics. We frame this as an architectural choice rather than a limitation: it preserves the action prediction's autonomy over gripper control while keeping the proprioception schema dimensionally consistent across rollouts.

**Mesh prior.** Object geometry is encoded via a pose-invariant shape descriptor computed offline from object meshes, forming a closed-set object library. Training and test splits in our LIBERO-Spatial evaluation share five mesh identities. This is a deliberate scoping choice — pose-invariant geometry encoding allows the decoder to reason about object affordances independently of pose without requiring per-test mesh generalization. The implications of this closed-set assumption for cross-suite generalization are discussed in §VI.

## §IV.C Diagnostic value of the OTP-Soft architecture for this study

The OTP-Soft architecture was originally designed as a candidate for addressing the language conditioning concerns motivating this work. The theoretical analysis presented here (§III.B) shows that no architecture-internal pathway can reduce the bound on policy-level language conditioning within the demonstration-supervised paradigm — a finding that is independent of any specific architecture. We retain OTP-Soft as the experimental platform for this paper on the basis of its diagnostic properties rather than its original architectural hypothesis:

1. **Architectural separation enables component ablation.** The explicit C3 boundary at the decoder, together with the well-defined conditioning input set $\{z, \text{proprioception}, \text{grasp affordance}, \text{object geometry}\}$, allows §V.D to systematically drop conditioning channels and isolate their contribution to decoder behavior. Architectures without such clean separation (e.g., end-to-end transformer decoders consuming arbitrary token mixtures) would conflate input-ablation effects with attention-allocation effects.

2. **Decoder-family qualitative comparison with OFT.** OTP-Soft and OFT share the same pretrained backbone but differ in multiple architectural dimensions: decoder family (Shortcut Flow Matching vs deterministic regression), intermediate representation (latent trajectory $z$ vs none), object query mechanism (5 object queries vs none), conditioning input set (mesh and grasp affordance vs single-image input), and proprioception schema (8-dim with structural zero vs 7-dim). We cannot fully isolate the decoder-family contribution from these other architectural differences within the present work. The contrast does inform our qualitative claim in §V.C.5 that at least one demonstration-supervised architecture (OFT) achieves high SR on LIBERO-Spatial while at least one other (OTP-Soft V3) fails on the same supervision protocol and same backbone — a contrast that complements but does not isolate the decoder-family axis. Controlled architecture experiments designed to isolate this axis are left to future work.

3. **Diagnostic infrastructure generalizes.** The Gate 1, M3, and C1 measurements developed for OTP-Soft (§V.A) do not depend on architectural specifics beyond access to backbone hidden states, demonstration trajectories, and policy latent representations. These measurements apply equally to OFT, RT-2, PI-0, or other VLA architectures sharing a comparable structural decomposition. The diagnostic methodology is the principal contribution of this paper; OTP-Soft is its first instantiation.

A reviewer might reasonably ask whether the architectural complexity of OTP-Soft is justified given that the architecture does not solve the conditioning problem it was originally designed to address. Our answer is twofold. First, the architecture's diagnostic value (above three reasons) holds independently of its original design hypothesis — it serves as a controlled experimental platform regardless of original motivation. Second, the absence of an architectural fix in this work — established theoretically by Theorem 3' and instantiated empirically by the §V.C.4 Mantel measurements and §V.C.2 sample-quality test — is itself a substantive finding. Future work should pursue auxiliary-objective interventions outside the demonstration-supervised loss (§VI), informed by the bound this paper establishes.

---

