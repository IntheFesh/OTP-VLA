# §I Re-pass — Fresh-eye Consistency Check (Day 3 P1)

**Purpose**: After all 6 sections drafted (§I, §II, §III.A, §III.B, §IV, §V.A, §V.B, §V.C, §V.D, §VI), re-read §I checking for inconsistencies introduced as later sections were written.

**§I version checked**: v2 final, committed `f4ecfd0` on 2026-05-12.

**Result**: No major inconsistencies found. 4 minor notes documented below. **§I v2 final stays as committed**; the minor notes can be folded into a future v3 if pre-submission polish requires.

---

## Cross-section consistency check

### §I.0 Hook ↔ §V.A.6 summary

**§I.0**: "We find: $r(h_{\mathrm{backbone}}, \ell) = +0.76$, $r(\text{demo trajectory}, \ell) = -0.27$. The pretrained backbone preserves language conditioning at strong correlation; the demonstration data on which VLAs are trained is approximately language-orthogonal."

**§V.A.6 table**:
- Gate 1: $+0.76$ — "Language preserved at strong correlation" ✅
- M3: $-0.27$ — "Language attenuated to near-null" ✅
- C1: $-0.19$ — "Policy latent inherits demonstration-level attenuation"

✅ Consistent. §I.0 numbers align with §V.A.6 measurements.

### §I.2 Contribution 2 ↔ §III.B Theorem 3' statement

**§I.2 C2**: "Theorem 3' formal bound ... satisfies $I(A^*; \ell \mid o) = I_{p_{\mathcal{D}}}(a; \ell \mid o) \le \varepsilon$. The bound is architecture-agnostic: it applies uniformly across $L_2$ regression, $L_1$ regression, Conditional Flow Matching, Shortcut Flow Matching, and cross-entropy on discretized actions."

**§III.B Theorem 3' statement** (consolidated v1, line 95-101): 
$$I(A^*; \ell \mid o) = I_{p_{\mathcal{D}}}(a; \ell \mid o) \le \varepsilon$$

**§III.B Remark 2 (loss families)**: lists L2, L1, CFM, SFM, cross-entropy on discretized actions ✅

✅ Consistent.

### §I.2 Contribution 3 ↔ §V.A.3 + §V.C.4

**§I.2 C3**: "$\rho_h^2 / \rho_{\text{demo}}^2 = 0.578 / 0.073 \approx 8\times$"

**§V.A.3**: $\rho^2 \le 0.073$ (range $[0.04, 0.073]$ across representations)
**§V.C.4** (consolidated v1): "$\rho_h^2 / \rho_{\text{demo}}^2 = 0.578 / 0.073 \approx 8\times$"

✅ Consistent.

### §I.2 Contribution 4 ↔ §V.C.5 + §V.B

**§I.2 C4**: "OpenVLA-OFT achieves 97% SR on LIBERO-Spatial with paraphrase invariance (§V.B: 9/10 tasks paired-diff CI within ±2pp; task 5 sensitivity caveat documented). Under Theorem 3', this is consistent with two scenarios: (a) low-magnitude language preservation, or (b) entirely visual-grounded."

**§V.B.2**: identity 94.6%, paraphrased 93.7%, 0.9pp ✅
**§V.B.3 task 5 caveat**: "Excluding task 5 does not change any other conclusion" ✅
**§V.C.5** (consolidated v1, two-scenario framing): scenarios (a) and (b) ✅

✅ Consistent.

### §I.2 Contribution 5 ↔ §V.C.2 + §V.C.3

**§I.2 C5**: "7/7 action dimensions fail the per-dimension sample-quality threshold (§V.C.2, L1/std ratio 1.88 overall). An oracle ablation replacing head output with ground-truth trajectory rules out the moving-target hypothesis (§V.C.3)..."

**§V.C.2** (consolidated v1): "Result: 0/7 dims PASS, 7/7 dims FAIL. Overall L1/std ratio: 1.88." ✅
**§V.C.3** (consolidated v1): "Decoder loss does not improve with oracle trajectory. Moving-target hypothesis rejected" ✅

✅ Consistent.

### §I.3 mechanism-based prediction ↔ §VI.5 (§VI single-benchmark scope)

**§I.3**: "We expect the empirical phenomenon (demonstration-trajectory $\perp$ language) to hold broadly under standard teleoperated demonstration protocols on the mechanistic argument that teleoperators visually perceive the scene and execute motor commands, with language serving as a post-hoc label..."

**§VI.5** (new draft) and **consolidated §VI.4**: same mechanistic argument articulated ✅

✅ Consistent.

### §I.3 "does not propose architectural fix" ↔ §IV.C + §V.C "Implications for §V.D"

**§I.3**: "This paper does not propose a new architectural fix. The OTP-Soft architecture we evaluate (§IV) was originally designed under a working hypothesis — since falsified by Theorem 3' — that a language-agnostic decoder boundary would bypass demonstration-induced conditioning collapse. We retain OTP-Soft as the experimental platform..."

**§IV.C** (`notes/paper_section_IV_draft.md`): "The OTP-Soft architecture was originally designed as a candidate for addressing the language conditioning concerns motivating this work. The theoretical analysis presented here (§III.B) shows that no architecture-internal pathway can reduce the bound..." ✅

**§V.C "Implications for §V.D"** (consolidated v1): "We do not propose a new architectural recovery in this work..." ✅

**§V.C v4 sync patches** (paper/drafts/section_V_C_v4_sync_patches.md): "Such a substitution remains within Theorem 3''s bound on policy-action language MI but may avoid CFM-specific sample-quality failure at the SR level (§V.C.2)..." ✅

✅ Consistent.

---

## Minor notes (defer to v3 if pre-submission polish)

### Note 1 — §I.2 Contribution 1 forward reference to NLP citations

**§I.2 C1 in v2 final**: "...prior probing literature in language-conditioned robotics focuses on model-internal representations after training... M3 instead probes the supervision data itself..."

This wording is consistent with §II.C (Related Work) which explicitly cites Tenney, Hewitt-Manning, Belrose for NLP probing and discusses M3's departure from policy-internal probing. ✅

However, in §I.2 v2 final we deliberately *removed* the NLP citations per Reviewer §三 feedback to fix reference class. The current text mentions "probing literature in language-conditioned robotics" without explicit citations. Reader following the §I → §II flow will encounter NLP probing references in §II.C, which is fine.

**Recommendation**: Keep §I.2 as is. The citation structure (§I cites VLA, §II.C cites NLP probing for methodology lineage) is correct.

### Note 2 — §I.2 Contribution 5 "isolated cause" language

**§I.2 C5 v2 final**: "The CFM decoder is sufficient to produce this failure under language-orthogonal demonstrations; whether it is the isolated cause of the SR gap with OFT requires controlled architecture experiments (§IV.C, future work)."

This aligns with §IV.C #2 "we cannot fully isolate the decoder-family contribution from these other architectural differences" and §V.C v4 sync patches. ✅

**Recommendation**: No change needed.

### Note 3 — §I.3 "auxiliary objectives" forward reference

**§I.3 v2 final**: "Bypassing Theorem 3''s bound requires interventions outside the demonstration-supervised loss — auxiliary language objectives, contrastive language alignment, inference-time conditioning on backbone-derived signals — and we discuss such directions only as future work (§VI)."

This aligns with §VI.4 (new draft) which discusses these three intervention classes explicitly, and §VI.6 future work direction 2. ✅

**Recommendation**: No change needed.

### Note 4 — Title placeholder

§I draft does not include a title line; title finalization (`paper_title_decision.md`) recommended Candidate 1 ("Where Did the Language Go? Tracing Conditioning Signal Loss in VLA Policies").

**Recommendation**: When integrating §I into final paper format (LaTeX), insert recommended title as the document title above the §I content.

---

## Summary

§I v2 final (commit `f4ecfd0`) is consistent with all later sections (§II, §III.A, §III.B, §IV, §V.A, §V.B, §V.C, §V.D, §VI). The 4 minor notes are formatting/structural items for pre-submission v3 polish; no content changes required for the current paper draft.

**Status**: §I v2 final retained as canonical. No commit needed from this re-pass — the result is "consistency verified."


