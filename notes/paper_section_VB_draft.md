# §V.B — Paraphrase Invariance and the OFT Visual-Grounded Regime (v1 draft)

**Status**: Draft v1. Day 3 P1. ~1.5 pages target.

**Note**: A prior draft exists at `notes/paper_section_V_B_initial_draft.md` (committed 2026-05-11). This v1 supersedes it with the post-Day-2-pivot framing — particularly the Rule A "supported/triggered" wording (not "fires") and the task 5 sensitivity-analysis caveat. The prior draft is preserved in git history; this version aligns with §III.B Theorem 3' v3 reformulation and §V.C v4 OFT contrast framing.

**Locked numbers from Stage 3 OFT ablation (userMemories #11)**:
- 10 tasks × 6 phrasings × 50 episodes = 3000 evaluations total
- Identity (original instruction): 94.6% mean SR
- Paraphrased instructions (5 variants): 93.7% mean SR
- Aggregate identity vs paraphrased diff: 0.9 percentage points
- 0/50 task-phrasing cells achieve BH-FDR adjusted $q < 0.05$
- All 5 paired-diff 95% confidence intervals include the ±2pp practical-equivalence threshold
- Task 5 outlier: identity 60% baseline (other tasks 90%+); 9-task subset shows identity 98% vs paraphrased 98%

---

## §V.B.1 Experimental setup

To assess the standard interpretation that paraphrase invariance evidences "language understanding" in VLA policies, we evaluate OpenVLA-OFT (Kim et al., 2025) on LIBERO-Spatial under controlled instruction rewording. The experimental matrix comprises 10 tasks × 6 instruction phrasings × 50 evaluation episodes per task-phrasing cell, totaling 3000 episode evaluations.

The 6 phrasings per task consist of one "identity" instruction (the original LIBERO-Spatial task description used for training) and 5 semantically-equivalent paraphrasings. Paraphrasings vary surface lexical form (e.g., "pick up" → "grasp"; "place" → "put") while preserving semantic content (the same target object, the same target location). Each task-phrasing cell is evaluated under 50 independent episode rollouts from the LIBERO-Spatial test split using the published OpenVLA-OFT checkpoint without further fine-tuning.

We report two quantities per task-phrasing cell: (1) point-estimate task success rate over 50 episodes; (2) Wilson 95% confidence interval on the per-cell SR. Aggregate quantities (mean SR across cells, paired identity-vs-paraphrased differences with 95% CIs) are computed across the task × phrasing matrix.

## §V.B.2 Aggregate result — paraphrase invariance

Across the full 10-task × 6-phrasing matrix:

$$\text{Identity SR} \;=\; 94.6\%, \quad \text{Paraphrased SR} \;=\; 93.7\%, \quad \Delta \;=\; 0.9 \text{ pp}.$$

The aggregate identity-vs-paraphrased difference of 0.9 percentage points is small relative to the per-cell sampling variance (with $n = 50$ episodes per cell, the Wilson 95% CI half-width is approximately ±9pp). Under any standard practical-equivalence threshold (±2pp, ±3pp, ±5pp), OFT's performance is statistically indistinguishable across paraphrasings.

We formalize this with the Stage 3 statistical protocol of our pre-registration (V7 §III.F):

- **Per-cell BH-FDR adjusted significance test**: 0 of 50 task-phrasing cells achieve $q < 0.05$ for any pairwise identity-vs-paraphrasing comparison (McNemar test, Benjamini-Hochberg adjustment across 50 cells).
- **Paired-diff confidence intervals**: All 5 paired identity-vs-paraphrased SR differences (one per paraphrasing variant, aggregated across tasks) have 95% confidence intervals that include the practical-equivalence threshold ±2pp.

By the pre-registered protocol, the joint conditions for "Rule A — visual-grounded SR regime" (the policy achieves high SR robust to instruction paraphrasing) are **supported** by these observations.

## §V.B.3 Task 5 sensitivity caveat

One task in the 10-task LIBERO-Spatial suite ("pick up the black bowl on the stove and place it on the plate," task 5) exhibits an unusual pattern: identity SR is 60% rather than the 90%+ observed on other tasks, with paraphrased SR also showing increased variance. This appears to be a per-task difficulty artifact unrelated to language-conditioning behavior — the same physical task is challenging across all phrasing variants — but it warrants explicit sensitivity analysis.

Excluding task 5 from the aggregate produces:

$$\text{Identity SR (9-task)} \;=\; 98\%, \quad \text{Paraphrased SR (9-task)} \;=\; 98\%, \quad \Delta \;=\; 0\,\text{pp}.$$

The 9-task subset preserves and strengthens the paraphrase-invariance finding. Excluding task 5 does not change any other conclusion in this section. We retain the full 10-task analysis as primary and report the 9-task subset as a sensitivity check rather than as a post-hoc selected analysis.

## §V.B.4 Interpretation under Theorem 3'

The paraphrase-invariance observation is conventionally read as evidence that the policy "understands" language: a model robust to instruction rewording is interpreted as having learned semantic content beyond surface form. Theorem 3' (§III.B) sharpens this interpretation. Under the bound

$$I(A^*; \ell \mid o) \le I_{p_{\mathcal{D}}}(a; \ell \mid o)$$

with demonstration-level signal $I_{p_{\mathcal{D}}}(a; \ell \mid o)$ measured at $\approx 0.04$ nats on LIBERO-Spatial (§V.A.3), OFT's 94.6% / 93.7% identity-vs-paraphrased SR result is consistent with two distinct scenarios:

(a) **Low-magnitude language preservation**: OFT does carry some language conditioning within the bound, sufficient to discriminate between semantically-different instructions but not surface-form variants of the same instruction. In this scenario, paraphrase invariance reflects genuine semantic conditioning at low MI magnitude.

(b) **Entirely visual grounding**: OFT carries no language conditioning at the action level. Paraphrase invariance reflects that the policy ignores language entirely and acts based on observation alone. With LIBERO-Spatial's tightly task-correlated visual scenes (each task has a distinctive object layout), observation alone is sufficient for task discrimination.

The paraphrase-invariance measurement cannot distinguish (a) from (b). Both predict the observed pattern: high identity SR, paraphrased SR statistically equivalent to identity SR. The two scenarios differ only in the *mechanism* of task success, which is not directly probed by the SR metric.

This is the diagnostic methodology contribution made explicit in §I (Contribution 4): the SR + paraphrase-invariance evaluation regime, while standard in the VLA literature, cannot distinguish between language-conditioned and visually-grounded high performance. Signal-flow diagnostics (Gate 1, M3, C1) provide complementary information: Gate 1's $r = 0.76$ confirms the backbone has language-discriminative capacity available; M3's $r = -0.27$ shows demonstration data does not encode language-action correlation; C1's $r = -0.19$ for OTP-Soft confirms the policy latent inherits the demonstration-level attenuation. Together these locate the dominant pathway as visual-grounded for the OFT case (b), though without ruling out small residual language preservation (a) at low MI magnitude.

## §V.B.5 Implications

Three implications follow from the paraphrase-invariance observation combined with Theorem 3':

1. **Paraphrase invariance is not evidence of language conditioning.** Under Theorem 3', any policy with demonstration-supervised training on LIBERO-Spatial-like data is bounded to small policy-level language MI. Both linguistically-conditioned policies (at low magnitude within the bound) and entirely visually-grounded policies predict paraphrase invariance. The metric does not distinguish.

2. **High SR is not evidence of language conditioning.** OpenVLA-OFT's 97% SR coexists with the Theorem 3' bound: the policy achieves task success via observation-action mapping; the contribution of language to action selection is bounded above by $\sim 0.04$ nats. Standard benchmark SR cannot establish language-driven task completion.

3. **Signal-flow diagnostics are complementary, not replacements.** Gate 1, M3, and C1 measure quantities orthogonal to task SR. They localize where language information is preserved, lost, or recovered across the VLA pipeline. They do not predict SR; they identify the information bottleneck that any high-SR policy must operate within (per Theorem 3'). We recommend their adoption alongside, not in place of, standard SR + paraphrase-invariance evaluations.


