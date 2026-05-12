# §V.C v4 — Sync patches for §IV v2 (2026-05-12 evening)

**Status**: 2 spot patches to existing §V.C v4 content in `paper/drafts/otp_vla_paper_drafts_consolidated.md`. These patches sharpen "decoder family isolation" claims to acknowledge architectural confounding between OTP-Soft V3 and OFT, ensuring paper internal consistency with §IV.C #2 (which acknowledges OTP-Soft vs OFT differ along multiple architectural dimensions).

**Required for**: paper internal consistency. §V.C v4 currently over-claims decoder family as isolated cause of SR gap; §IV.C #2 now (correctly) acknowledges confounding. These patches align §V.C with §IV.C.

**Application**: When integrating §IV v2 + §V.C v4 + §III.B v3 into final paper draft, replace the noted spots in §V.C v4 with the patched versions below. Until then, the patches are documented here and the consolidated draft should be considered superseded in these 2 spots.

---

## Spot 1 — §V.C.5 末段

### Old (in `otp_vla_paper_drafts_consolidated.md` §V.C.5)

> OFT and our V3 architecture therefore do not differ in language preservation — both are bounded — but in **how they fail to use language**:
>
> - **OFT**: succeeds at SR via visual-grounded action regression; language signal lost at the policy level but task succeeds via observation-action correlation
> - **V3 (CFM head + CFM decoder)**: fails at both SR (0/50) and language conditioning (C1 $r = -0.19$); CFM sample-quality failure documented in §V.C.2-3 dominates over any potential language signal
>
> This OFT contrast strengthens the paper's central diagnostic claim: high task-success rate does not imply language understanding. The §V.B Rule A condition is triggered (9-task subset paraphrase-invariance), localizing OFT's success to the visual-grounded regime; §V.C.2-3 localizes V3's failure to the CFM decoder family.

### New (replacement)

> OFT and our V3 architecture therefore do not differ in language preservation — both are bounded — but in **how they fail to use language**:
>
> - **OFT**: succeeds at SR via visual-grounded action regression; language signal lost at the policy level but task succeeds via observation-action correlation
> - **V3 (CFM head + CFM decoder)**: fails at both SR (0/50) and language conditioning (C1 $r = -0.19$); CFM sample-quality failure documented in §V.C.2-3 dominates over any potential language signal
>
> This OFT contrast strengthens the paper's central diagnostic claim: high task-success rate does not imply language understanding. The §V.B Rule A condition is triggered (9-task subset paraphrase-invariance), localizing OFT's success to the visual-grounded regime; §V.C.2-3 documents V3's failure mode (CFM-family sample-quality collapse). The CFM decoder is *sufficient* to produce this failure under language-orthogonal demonstrations, but we do not claim it is the *isolated* cause of the SR gap with OFT, since OTP-Soft V3 and OFT differ along multiple architectural dimensions beyond the decoder family (§IV.C #2). Isolating decoder family from other architectural differences requires controlled experiments left to future work.

---

## Spot 2 — §V.C "Implications for §V.D" — Cross-architecture comparison bullet

### Old (in `otp_vla_paper_drafts_consolidated.md` §V.C ending)

> 2. **Cross-architecture comparison via OFT contrast** (§V.C.5 cross-reference): OFT and V3 share backbone but differ in decoder family (deterministic regression vs CFM). The 97% vs 0% SR gap localizes failure to the decoder design, not to data, backbone, or supervision signal.

### New (replacement)

> 2. **Cross-architecture comparison via OFT contrast** (§V.C.5 cross-reference): OFT and V3 share backbone and supervision protocol but differ along multiple architectural dimensions including decoder family. The 97% vs 0% SR gap, given that backbone and supervision are shared, is consistent with the decoder family being a sufficient contributor to the SR gap; full isolation from other architectural differences (intermediate representation, conditioning inputs, query mechanism) requires controlled architecture experiments left to future work.

---

## Patch impact summary

These two spots are the only places in §V.C v4 that over-claimed "decoder family isolated as failure cause". The Theorem 3' interpretation, sample-quality test, oracle ablation, and two-scenario OFT framing are all unaffected and remain authoritative in the consolidated draft.

**Action item for next consolidated draft revision**: When producing the next consolidated paper draft (presumably after §I writing complete), apply these 2 spot patches inline to §V.C v4 content, and archive this patch file to `paper/drafts/archive/` with a note explaining the v4 → v4.1 transition.
