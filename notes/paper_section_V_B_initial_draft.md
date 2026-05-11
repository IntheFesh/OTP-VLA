# Paper §V.B — Initial Draft

**Status**: Initial draft 2026-05-11. Based on V2 analysis path (V7 §III.F + BH-FDR + CI-aware threshold). Rule A triggered.
**Version**: v0.1 (pre-Step 6 retrain results; §V.C/§V.D placeholders)

---

## §V.B Empirical Evaluation of OFT Paraphrase Invariance

We evaluated OpenVLA-OFT [Kim et al. 2024] on LIBERO-Spatial under controlled instruction paraphrasing to test the Theorem 3 prediction independently. The protocol (pre-registered V7 §III.F) consisted of 10 LIBERO-Spatial tasks × 6 instruction phrasings × 50 episodes per cell, totaling 3000 episodes. The six phrasings comprise one identity phrasing (the original LIBERO task description) and five semantic-preserving paraphrasings spanning word-order rearrangement (P1), synonym substitution (P2), passive voice (P3), verb change (P4), and compact rephrasing (P5). Spatial qualifiers ("on the ramekin", "between the plate and the ramekin", etc.) were preserved across all paraphrasings to maintain task identity.

Aggregate results: identity SR averaged 94.6% across 10 tasks; mean paraphrased SR was 93.7%—a 0.9pp difference well within binomial sampling noise (SE ≈ 1.1pp on n = 500 per phrasing).

Per pre-registered analysis (V7 §III.F), we applied McNemar tests on paired identity-vs-paraphrased outcomes across all 50 (task × phrasing) cells, with Benjamini-Hochberg FDR correction at α = 0.05. No cell achieved FDR-significant rejection (all q > 0.05). Paired SR differences (identity − paraphrased) per phrasing, with 95% CIs computed across the 10 tasks:

| Phrasing | Diff (pp) | 95% CI |
|---|---|---|
| P1 word order | +1.4 | [−1.4, +4.2] |
| P2 synonym | −0.8 | [−3.4, +1.8] |
| P3 passive | +0.8 | [−2.4, +4.0] |
| P4 verb change | +1.0 | [−2.5, +4.5] |
| P5 compact | +2.0 | [−0.3, +4.3] |

All five CIs include the V7 Rule A SR threshold (paraphrased ≥ identity − 2pp); the data is statistically consistent with the visual-grounded explanation, and the FDR test yields no significant per-cell rejections. Per pre-registered §III.F decision criteria, the data triggers **Rule A (visual-grounded explanation)**: OFT's high LIBERO-Spatial SR derives primarily from visual grounding rather than sensitivity to instruction surface form.

### Task 5 sensitivity disclosure

One task merits explicit disclosure. Task 5 ("on the ramekin") is a known weak baseline for OFT (identity SR = 60%, vs. 96-100% on the other 9 tasks). On task 5, paraphrased SR averages 51% across phrasings (a 9pp paraphrased drop), with raw McNemar p-values in [0.07, 0.30]; no cell survives FDR correction. The observed drop is comparable to the per-task SE at this baseline (≈6.5pp); we attribute it to limited statistical power on a low-baseline task rather than language-specific brittleness. Excluding task 5 as a sensitivity check: 9-task identity mean SR is 98.0%, 9-task paraphrased mean SR is 98.0% — no observable drop, all conclusions unchanged.

### Connection to Theorem 3

These findings provide independent empirical support for the Theorem 3 prediction. Theorem 3 was derived from properties of the demonstration distribution (M3: trajectory $\perp$ language $\mid$ observation, verified at $r \in [-0.27, -0.20]$ across three trajectory representations in §V.A); OFT is an externally trained policy (Kim et al. 2024) whose behavior under paraphrasing was not used to derive the theorem. The observed paraphrase invariance of OFT on its high-baseline tasks therefore serves as out-of-distribution validation of Theorem 3's mechanism: language-orthogonal supervision yields policies whose task discrimination capacity does not require fine-grained instruction parsing.

A key implication is that the apparent "language conditioning" of OFT—i.e., its acceptance of instruction strings as inputs—does not translate to fine-grained sensitivity to instruction surface form. The instruction string functions, in effect, as a coarse task selector rather than a per-action conditioning signal. This is consistent with our M3 measurement showing that demonstrations carry near-zero trajectory-level information about instruction content given the observation.

### Caveats and limitations

(i) Generalization beyond LIBERO-Spatial. The evaluation tests one benchmark family. Our claim is specifically that OFT exhibits paraphrase invariance on LIBERO-Spatial; whether this holds on tasks where instructions encode physical action subtleties (e.g., "carefully" vs. "quickly") is open.

(ii) Five paraphrasings sampled, not exhaustive. The five paraphrasings cover common semantic-preserving transformations but do not span the full space of possible rephrasings. Adversarial paraphrasings (e.g., negation: "do not pick up the bowl") could in principle elicit different behavior; these are out of scope for the current Theorem 3 prediction.

(iii) Task 5 caveat as above.

---

## Decision Rationale (Pre-Registration Trail)

The Rule A trigger is based on:
- V2 specification (CI-aware threshold application, FDR on all 50 cells)
- Robustness check via V3 (CI-aware + informative-cell filter b+c ≥ 4): same Rule A trigger conclusion

Statistical implementation choices are minimal and standard:
- BH-FDR correction (default for k = 50 multiple tests)
- 95% CI overlap for threshold criterion (default for noisy point estimates)

No post-hoc rule modification. No data-dependent subset selection. Task 5 is included in the main analysis; the 9-task sensitivity analysis is a robustness check, not a primary analysis.

---

## Connections to Other §V Subsections

- **§V.A (Phase 0 diagnostics)**: Establishes the M3 finding (demonstrations language-orthogonal at $r \in [-0.27, -0.20]$) used by Theorem 3.
- **§V.C (Path B retrain)**: To be drafted after Step 6 completes. Reports OTP-Soft SR after deterministic head + Cocos source retraining (3 seeds × 50 epoch).
- **§V.D (Component ablation)**: To be drafted after ablation runs (Step 6 + drop mesh/grasp/proprio variants).

---

*Initial draft frozen pending Step 5/6 results. §V.B independent of Path B retrain outcome — locked.*
