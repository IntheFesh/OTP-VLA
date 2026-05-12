# §III.A — Conditions on Policy and Demonstration Data (v1 draft)

**Status**: Draft v1. Day 3 P1. ~1 page target.

**Purpose**: Introduces the C2 condition (instruction-equivalence-class informativeness of policy latent) and the C3 condition (architectural exclusion of $\ell$ from decoder input), and sets up notation used by Theorem 3' in §III.B. Precedes §III.B in paper organization.

**Locked decisions** (from userMemories #4, #9, #16):
- C2: $I(z; \tau \mid o) > 0$ where $\tau$ is instruction equivalence class under semantic paraphrase
- C3: architectural — $\ell \notin \text{inputs}(\phi_\theta)$
- C1 is not stated as a condition but as a measurement (§V.A) of whether C2 holds at the policy level after training

---

## §III.A.1 Setup and notation

We consider a policy $\pi: \mathcal{O} \times \mathcal{L} \to \Delta(\mathcal{A})$ mapping observation-instruction pairs to distributions over robot actions, trained on demonstration data $p_{\mathcal{D}}(o, \ell, a)$ over observation space $\mathcal{O}$, instruction space $\mathcal{L}$, and action space $\mathcal{A}$. We decompose the policy as

$$\pi(a \mid o, \ell) \;=\; \phi_\theta(a \mid z, w(o)),$$

where $w(o)$ denotes observation-derived features (visual encodings, proprioception, geometry, affordance maps — see §IV.B for instantiation), and $z = h_\psi(o, \ell)$ is an instruction-aware latent produced by an upstream module (the "head" in our architecture). The decoder $\phi_\theta$ is a stochastic mapping that produces actions conditional on the latent $z$ and observation features $w(o)$.

This decomposition is sufficiently general to cover most VLA architectures considered in §II.B: in OpenVLA (Kim et al., 2024) and RT-2 (Brohan et al., 2023), $z$ corresponds to the LLM hidden state at the last text token, $w(o)$ is implicit in visual tokenization, and $\phi_\theta$ is a softmax over discretized action tokens; in OFT (Kim et al., 2025), $z$ is the same LLM hidden state and $\phi_\theta$ is a deterministic regression head; in π₀ (Black et al., 2024), $\phi_\theta$ is a flow-matching head; in OTP-Soft (this paper, §IV.B), $z$ is an object-pose trajectory and $\phi_\theta$ is a Shortcut Flow Matching decoder. The decomposition does not assume any particular architectural form for $\phi_\theta$ or $h_\psi$.

We use $a \sim p_{\mathcal{D}}(\cdot \mid o, \ell)$ for action samples from the demonstration distribution conditional on $(o, \ell)$, and $A^* \sim \pi^*(\cdot \mid o, \ell)$ for action samples from a policy at the same conditioning.

## §III.A.2 Condition C2 — task-discriminative latent

The first condition concerns whether the latent $z$ carries task-discriminative information beyond what is determined by the observation alone:

**Condition C2 (task-discriminative latent).** Let $\tau(\ell)$ denote the instruction equivalence class of $\ell$ under semantic paraphrase — that is, $\tau(\ell) = \tau(\ell')$ if and only if $\ell$ and $\ell'$ are paraphrases referring to the same intended task. We require

$$I(z; \tau \mid o) > 0.$$

In words: the latent $z$ must carry information about the task identity (instruction equivalence class) beyond what the observation $o$ already provides. C2 is a necessary condition for the policy to behave differently across paraphrased instructions referring to different tasks while sharing the same observation context.

The use of $\tau$ rather than the raw instruction $\ell$ is important. Two paraphrased instructions referring to the same task ("pick up the black bowl" and "grasp the dark bowl") should produce the same conditioning on the policy, even though the language embeddings of $\ell$ and $\ell'$ differ in surface form. Requiring $I(z; \tau \mid o) > 0$ targets the *semantic* content of language, not the surface form. An earlier formulation requiring only $H(z \mid o) > 0$ is insufficient: it is satisfied by injecting any task-independent noise into $z$, which carries no information about the intended task.

C2 is a property of the policy under evaluation, not of the supervision data. In §V.A (C1 diagnostic) we measure a related quantity — the Mantel correlation $r(z, \ell)$ on policy latents — as an empirical proxy for whether C2 holds substantively after training.

## §III.A.3 Condition C3 — architectural language-agnosticism of the decoder

The second condition concerns whether the decoder $\phi_\theta$ receives the instruction $\ell$ directly as input, separately from the routing through $z$:

**Condition C3 (decoder language-agnosticism).** The decoder $\phi_\theta$ does not receive $\ell$ as an input:

$$\ell \;\notin\; \text{inputs}(\phi_\theta).$$

All language information that reaches the decoder must flow through the latent $z$. Equivalently, the decoder's action output $\phi_\theta(a \mid z, w(o))$ is conditionally independent of $\ell$ given $(z, w(o))$.

C3 is an architectural condition rather than an empirical one. In OTP-Soft (§IV.B), C3 is enforced by construction: the decoder's `forward()` signature accepts only $(z, w(o))$ inputs, and a `_check_forbidden` guard raises `AssertionError` at both module construction and every forward pass if any forbidden name fragment related to $\ell$ appears among parameter names, submodule names, or runtime locals. In other VLA architectures, C3 holds when the decoder consumes only LLM hidden states and visual features without re-injecting language tokens at the action prediction stage; this is the case for OFT (Kim et al., 2025) and π₀ (Black et al., 2024) under standard implementations.

The motivation for C3 is structural clarity rather than performance. By routing all language information through a single bottleneck ($z$), the architecture exposes a clean diagnostic target: measuring $I(z; \ell \mid o)$ (or its empirical proxy $r(z, \ell)$ in §V.A) directly assesses whether the upstream head $h_\psi$ has preserved language information for the decoder to use. Without C3, the decoder could potentially route language information through a second, unobserved pathway, conflating the diagnostic target.

C3 is not a sufficient condition for the policy to use language at the action level. As we show in §III.B (Theorem 3') and §V.C (empirical instantiation), C3 combined with demonstration-supervised training does not guarantee policy-level language conditioning: the bound on $I(A^*; \ell \mid o)$ depends on the demonstration data's language-action correlation structure, not on the architectural enforcement of C3.

## §III.A.4 Relationship between C2 and C3

C2 and C3 address different layers of the policy. C2 is a requirement on the latent $z$ — it must encode task-discriminative information. C3 is a requirement on the decoder $\phi_\theta$ — it must not bypass $z$ by reading $\ell$ directly. Together, they specify a clean architectural decomposition where:

- The head $h_\psi$ is the only module that consumes $\ell$
- The decoder $\phi_\theta$ depends on $\ell$ only indirectly through $z$
- The C2 condition's diagnostic ($I(z; \tau \mid o)$) and the policy's empirical language conditioning ($I(A^*; \ell \mid o)$) can be measured independently

Under this decomposition, Theorem 3' (§III.B) provides the bound on $I(A^*; \ell \mid o)$ in terms of demonstration-level signal. The §V.A diagnostic measurements (Gate 1 on $h_\psi$'s upstream backbone state, M3 on demonstration trajectories, C1 on $z$) then trace how language information propagates from the backbone, through the supervision data, into the trained policy's latent. The architectural conditions C2 and C3 ensure these measurements isolate the correct quantities; the empirical bound in §III.B Theorem 3' explains the relationship between them.


