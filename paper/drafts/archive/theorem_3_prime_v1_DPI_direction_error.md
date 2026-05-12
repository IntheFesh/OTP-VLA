# Theorem 3' (Approximate signal collapse via DPI) - PAPER DRAFT

**Setup**: Demonstration distribution p_D(o, ℓ, a) over observations o ∈ O, 
instructions ℓ ∈ L, actions a ∈ A. Policy π: O × L → Δ(A).

## Definition 1 (ε-language-orthogonal demonstrations)

The demonstration distribution is ε-language-orthogonal if
$$I_{p_D}(a; \ell \mid o) \leq \epsilon$$

## Theorem 3' (Approximate supervision-induced collapse)

Let π* minimize expected per-sample loss E_{p_D}[L(π(o, ℓ), a)]. If L is a
distribution-matching loss for which the population minimizer satisfies
π*(·|o, ℓ) = p_D(·|o, ℓ) almost surely, then
$$I(\pi^*; \ell \mid o) \leq I_{p_D}(a; \ell \mid o) \leq \epsilon$$

### Proof

Apply DPI to the Markov chain:
$$\ell \to (o, \ell) \to a \to \pi^*(\cdot \mid o, \ell)$$

Since π*(·|o,ℓ) = p_D(a|o,ℓ), the conditional MI I(π*; ℓ | o) is bounded by
I_{p_D}(a; ℓ | o). The second inequality is Definition 1. □

## Remark 1 (Loss families satisfying the population-minimizer property)

L2 regression (mean-conditional), L1 regression (median-conditional), CFM /
ShortcutFlowMatching (matched-flow representation), and cross-entropy on
discretized actions all satisfy π* = p_D(a|o,ℓ) at population level.
Theorem 3' therefore applies broadly to standard VLA training objectives.

## Corollary (Empirical instantiation)

For LIBERO-Spatial demonstrations:
- r(traj, ℓ) ∈ [-0.27, -0.20] across 3 trajectory representations
- ρ² ≤ 0.073
- Gaussian-surrogate bound: Î(a; ℓ | o) ≤ -½ log(1 - ρ²) ≤ 0.038 nats
- Theorem 3' predicts: I(π*; ℓ | o) ≤ 0.038 nats
- Empirically: r(z_V3, ℓ) = -0.19 → I ≈ 0.018 nats (consistent with bound)

## Bridge to Section IV (deterministic head)

Theorem 3' constrains policies whose ONLY source of language signal is
demonstration supervision. Architectures that admit a non-supervision-mediated
language pathway — e.g., a deterministic head that reads h_OFT (a backbone
hidden state) directly and propagates it through the decoder via fixed
differentiable transformations — induce a Markov chain

$$\ell \to h_{OFT} \to z \to \pi(a|o, \ell)$$

that does not factor through a ~ p_D. The DPI bound of Theorem 3' does not
apply to this pathway, motivating our deterministic-head design (§IV.B).
