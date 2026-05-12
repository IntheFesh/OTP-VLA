# Phase 0 Final Numbers — Single Source of Truth

**Locked 2026-05-12. All paper references draw from this file, NOT directly from `results/phase0/` (which is subject to overwrite).**

## Provenance
- LIBERO-Spatial, 10 tasks × 10 demos, frame 0
- Backbone: OpenVLA-OFT (SigLIP+DINOv2 dual-encoder)
- Replication run: 2026-05-12 17:00 CST
- Result file (replicated): `results/phase0/gate1_sanity_and_m3.json`

## Gate 1: r(h_OFT, language)
- Original (2026-05-11): r = +0.7423, p < 0.001
- Replicated (2026-05-12): r = +0.7575, p < 0.001
- Δ = 0.0152 (within sampling noise)
- **Paper number: r ≈ 0.76** (both runs round to 0.76)

## M3: r(demo_trajectory, language)
- Flat representation (primary): r = -0.2698, p = 0.964 (replicated 2026-05-12)
- Range across 3 representations (flat / endpoint / waypoint): r ∈ [-0.27, -0.20]
- **Paper number: r = -0.27 (flat, primary) | r ∈ [-0.27, -0.20] (range across representations)**
- Squared: ρ² = 0.073 (flat) | ρ² ∈ [0.04, 0.073] (range)

## C1: r(z_OTP, language)
- Phase 0 deep dive: r = -0.19
- η² median: 0.26, concentrated in non-target object slots
- Source file: `results/phase0/condition_check.json`

## Test 4 (backbone capacity): F̄ᵃ = 11.30
- Source: `results/phase0/test4_capacity.json`

## Theorem 3' Gaussian MI surrogates (QUALITATIVE, not quantitative)
These are r²-based Gaussian surrogates, not kosher MI estimates. Joint Gaussianity is not assumed in our analysis; surrogates report effect-size magnitude only.
- Î(h_OFT; ℓ|o) ≈ 0.43 nats from r_h² = 0.578 (Gaussian surrogate)
- Î(a; ℓ|o) ≈ 0.04 nats from r_demo² = 0.073 (Gaussian surrogate)
- Squared-correlation gap: r_h² / r_demo² ≈ 8× (this is the primary effect-size ratio)
- MI-surrogate gap: ~10× (qualitative reference only, NOT entering quantitative bound)

## Cross-test consistency note (from gate1_sanity_and_m3.json)
- C1 D_language (raw mean-pool): 0.167
- Gate 1 D_language (raw mean-pool): 0.097
- Both use byte-identical embedding code path (SC2 verified)
- Difference: minor sampling effect (C1 used 10 tasks single demo, Gate 1 used 10 tasks × 10 demos)

---

## Note on `framing_implication` field in raw JSON results

The Phase 0 result files contain a `framing_implication` text field hardcoded by diagnostic scripts at write-time. These fields reflect the paper framing **at the time of each diagnostic run**, and necessarily lag behind the final paper §V framing as analysis evolved.

The empirical numbers (SC1, SC2, SC3, M3, C1, Test 4) in the raw JSON files are the authoritative data and remain unchanged across framing iterations. The final paper §V draws its framing from the post-hoc full-data analysis (§V.A through §V.C), not from any single diagnostic JSON's `framing_implication` field.

For reviewers consulting raw results: the JSON `framing_implication` fields represent intermediate working hypotheses (e.g., "selective collapse" in earlier scripts, "demonstration distillation failure" in later scripts). The published framing — CFM-decoder sample-quality failure under language-orthogonal demonstration supervision (§V.C) — is the post-diagnostic synthesis presented in this paper.
