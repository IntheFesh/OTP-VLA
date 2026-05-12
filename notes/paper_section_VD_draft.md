# §V.D — Component Ablation and Cross-Architecture Comparison (placeholder for ablation results)

**Status**: Placeholder v1. Day 3 P1. ~1 page target.

**Note**: This section documents the §V.D experimental plan and pre-specified analysis protocol. Empirical results fill in after the 7.5-day GPU ablation plan completes (Day 4-10). The pre-specification here protects the reported results from post-hoc selection bias.

**Planned experiments (from userMemories #20)**:
- 4 architecture variants × 3 seeds + 3 conditioning-input ablations × 1 seed = 15 retraining runs
- Architecture variants: (a) Path B base (det head, $\mathcal{N}(0, I)$ decoder source); (b) Path B + Cocos; (c) CFM head + det decoder (reverse); (d) V3 baseline (CFM head + CFM decoder)
- Conditioning ablations: drop mesh / drop grasp affordance / drop proprioception (all on V3 baseline)
- Statistical reporting per V7 §III.F protocol: McNemar pairwise tests + BH-FDR adjustment + Wilson 95% CI on SR

---

## §V.D.1 Two-axis ablation design

To probe the relative contributions of the two architectural axes identified in §IV.A (decoder family and conditioning input richness) to policy behavior under the Theorem 3' bound, we conduct two complementary ablations:

**Axis 1 — Decoder family.** Four architecture variants holding backbone, supervision protocol, and conditioning inputs constant while varying the head and decoder loss family:

| Variant | Head loss | Decoder loss | Source distribution |
|---|---|---|---|
| (a) Path B base | Deterministic regression | Shortcut Flow Matching | $\mathcal{N}(0, I)$ |
| (b) Path B + Cocos | Deterministic regression | Shortcut Flow Matching | Cocos shifted source |
| (c) CFM head + det decoder | Shortcut Flow Matching | Deterministic regression | (not applicable) |
| (d) V3 baseline | Shortcut Flow Matching | Shortcut Flow Matching | $\mathcal{N}(0, I)$ |

Each variant is trained with 3 independent random seeds following the same training schedule and hyperparameter configuration as V3 (lr $= 10^{-4}$, full LIBERO-Spatial training set, 30 epochs).

**Axis 2 — Conditioning input richness.** Three single-input ablations on the V3 baseline architecture, removing one conditioning channel at a time:
- (e) V3 minus mesh: object-geometry input set to zero
- (f) V3 minus grasp affordance: grasp affordance map set to zero
- (g) V3 minus proprioception: proprioception input set to zero

Each ablation is trained with 1 seed using V3's training schedule.

The 4 + 3 = 7 configurations × seed counts produce 12 + 3 = 15 retraining runs. At 12.5 GPU-hours per training run, the full sweep requires approximately 7.5 GPU-days sequential.

## §V.D.2 Pre-specified analysis protocol

We pre-specify the analysis protocol prior to obtaining results to prevent post-hoc selection bias. All reported quantities and tests are determined here:

**Primary outcome.** Per-task success rate over $n_{\text{eval}} = 50$ rollout episodes per task per seed, on LIBERO-Spatial's 10 tasks.

**Pairwise comparisons.** For each axis, all pairwise differences between configurations:
- Axis 1: 6 pairwise comparisons across {a, b, c, d}
- Axis 2: 3 pairwise comparisons between each ablation and V3 baseline {e vs d, f vs d, g vs d}

**Statistical tests.** McNemar's paired test on per-episode success/failure outcomes, applied per task per pairwise comparison. P-values aggregated across the 10 tasks × N pairs using Benjamini-Hochberg FDR adjustment with target $q < 0.05$.

**Confidence intervals.** Per-cell Wilson 95% CI on SR. For paired aggregate differences, paired 95% CI via the Newcombe-Wilson hybrid method on per-task SR pairs.

**Practical-equivalence threshold.** ±2 percentage points absolute SR difference, applied to paired aggregate differences (matching §V.B Stage 3 protocol).

**Pre-specified decision rules.**
- If Axis 1 shows 0/6 statistically significant pairs at $q < 0.05$ AND all paired-diff 95% CIs include ±2pp: conclude **architecture-uniform failure** — all variants achieve equivalent SR, consistent with Theorem 3' bound dominating performance independent of decoder family.
- If Axis 1 shows ≥1 significant pair: report which variant differs and discuss in terms of decoder family or source distribution.
- If Axis 2 shows 0/3 significant pairs: conclude **conditioning-input-uniform failure** — V3 baseline decoder is insensitive to conditioning input identity, consistent with §V.C.2 sample-quality collapse dominating over any input-driven signal.
- If Axis 2 shows ≥1 significant pair: report which conditioning input matters and discuss in relation to §V.A.4 (C1 latent structure showing visually-grounded $\eta^2$ distribution).

## §V.D.3 Connection to §V.C and Theorem 3'

§V.C established two empirical findings: (§V.C.2) CFM decoder samples are not task-discriminative under language-orthogonal demonstration supervision; (§V.C.5) OFT achieves high SR without the bound being slack — its 97% SR is consistent with two scenarios both within the bound. §V.D's two-axis design probes:

- **Whether Axis 1 (decoder family choice) produces measurable SR differences within the Theorem 3' bound.** If decoder family is the dominant factor (as §IV.A predicts), we expect the CFM-family configurations (a, b, d) to cluster at low SR with the det-decoder configuration (c) producing higher SR. If Axis 1 shows architecture-uniform failure, the §V.C.2 sample-quality collapse is the unifying mechanism.

- **Whether Axis 2 (conditioning inputs) produces measurable SR differences.** If mesh, grasp affordance, or proprioception contribute substantively to V3's behavior despite the decoder's sample-quality collapse, ablating them should hurt SR. If Axis 2 shows conditioning-input-uniform failure, the decoder's failure mode (§V.C.2) overrides any conditioning-input contribution.

The pre-specified decision rules above are designed so that any of the four possible outcomes (architecture-uniform × conditioning-uniform, architecture-uniform × conditioning-sensitive, architecture-sensitive × conditioning-uniform, architecture-sensitive × conditioning-sensitive) maps to a substantive conclusion supporting the diagnostic narrative of §V.C — that the failure mode is decoder-family-locked and demonstration-distribution-induced rather than conditioning-input-driven.

## §V.D.4 Results

*[Results to be filled in after Day 4-10 GPU ablation completion. Pre-specified protocol §V.D.2 governs analysis.]*

**Reporting template** (to be populated):

### §V.D.4.1 Axis 1 results table

| Variant | Per-task mean SR (3 seeds) | 95% CI | Pairwise vs V3 baseline |
|---|---|---|---|
| (a) Path B base | TBD | TBD | TBD |
| (b) Path B + Cocos | TBD | TBD | TBD |
| (c) CFM head + det decoder | TBD | TBD | TBD |
| (d) V3 baseline | TBD | TBD | (reference) |

### §V.D.4.2 Axis 2 results table

| Variant | Per-task mean SR (1 seed) | 95% CI | Diff vs V3 baseline (95% CI) |
|---|---|---|---|
| (e) V3 − mesh | TBD | TBD | TBD |
| (f) V3 − grasp affordance | TBD | TBD | TBD |
| (g) V3 − proprioception | TBD | TBD | TBD |
| (d) V3 baseline | TBD | TBD | (reference) |

### §V.D.4.3 Decision-rule outcome

*[Selected from §V.D.2 pre-specified branches based on observed pairwise significance and CI widths.]*

## §V.D.5 Discussion of expected outcomes (pre-results)

Based on §V.C.2 finding (CFM decoder sample-quality collapse on language-orthogonal demonstrations), our pre-results expectation is:

- **Axis 1**: configurations (a), (b), (d) cluster at low SR (CFM-family decoder, all subject to §V.C.2 collapse). Configuration (c) (det decoder reversed pairing) may achieve higher SR via visual grounding analogous to OFT (§V.C.5), but is also subject to Theorem 3' bound on language MI.
- **Axis 2**: all three ablations show SR equivalent to V3 baseline within practical-equivalence threshold, because the CFM decoder's sample-quality collapse dominates over conditioning-input choices.

We emphasize this is a prediction, not a foregone conclusion. The pre-specified protocol §V.D.2 applies independently of expected outcome. Results contradicting the pre-results expectation will be reported faithfully and discussed in terms of what they reveal about the decoder-family vs conditioning-input axes.


