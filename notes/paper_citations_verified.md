# Citation Audit — Published Venues vs arXiv-only (Day 3 P1 update)

**Status**: Updated 2026-05-13. Supersedes prior citation note.

**Reviewer concern**: arXiv preprints reduce paper authority. Where possible, cite the published venue (conference/journal); use arXiv only when no published venue exists yet.

---

## Published venue citations (use these in paper)

| Citation | Title | Published Venue | arXiv ID | Cite in paper as |
|---|---|---|---|---|
| Brohan et al., 2023 | RT-2: Vision-Language-Action Models Transfer Web Knowledge to Robotic Control | **CoRL 2023** (PMLR 229: 2165-2183) | 2307.15818 | "Brohan et al., CoRL 2023" |
| Octo Model Team et al., 2024 | Octo: An Open-Source Generalist Robot Policy | **RSS 2024** (Delft) | 2405.12213 | "Octo Model Team et al., RSS 2024" |
| Kim et al., 2024 | OpenVLA: An Open-Source Vision-Language-Action Model | **CoRL 2024** (pp. 2679-2713) | 2406.09246 | "Kim et al., CoRL 2024" |
| Black et al., 2024 | π₀: A Vision-Language-Action Flow Model for General Robot Control | **RSS 2025** | 2410.24164 | "Black et al., RSS 2025" |
| Frans et al., 2024 | One Step Diffusion via Shortcut Models | **ICLR 2025** | 2410.12557 | "Frans et al., ICLR 2025" |
| Lipman et al., 2023 | Flow Matching for Generative Modeling | **ICLR 2023** | (already pub) | "Lipman et al., ICLR 2023" |
| Tenney et al., 2019 | BERT Rediscovers the Classical NLP Pipeline | **ACL 2019** (pp. 4593-4601) | 1905.05950 | "Tenney et al., ACL 2019" |

---

## arXiv-only (use sparingly + cite alternative if possible)

| Citation | Title | Status | Strategy |
|---|---|---|---|
| Kim et al., 2025 (OFT) | Fine-Tuning Vision-Language-Action Models: Optimizing Speed and Success | arXiv 2502.19645, no peer-reviewed venue yet | See OFT handling below |

### OpenVLA-OFT handling strategy

The OFT paper is arXiv-only at submission time. It is also the only choice for citing the 97% SR result on LIBERO-Spatial that anchors §V.C.5. Three options:

**(A) Cite as arXiv preprint, single instance only**: cite "Kim et al., arXiv 2025" in §V.C.5 as the source of 97% number. Acknowledge in footnote that this is a preprint. Reviewer accepts because no peer-reviewed alternative exists for this specific result.

**(B) De-emphasize OFT specifically, cite OpenVLA published version**: For backbone description (§IV.B), cite "OpenVLA-OFT (Kim et al., arXiv 2025)" alongside "OpenVLA (Kim et al., CoRL 2024)" as the lineage. For §V.C.5 OFT contrast, rephrase to use OpenVLA-CoRL2024 as the contrast architecture instead of OFT. Issue: OpenVLA achieves ~76% on LIBERO-Spatial, not 97%; the 8× squared-correlation gap argument relies on the 97%+SR achiever existing within the bound — OpenVLA's 76% weakens the rhetorical sell.

**(C) Use both**: cite OpenVLA (CoRL 2024) as the architectural lineage; cite OpenVLA-OFT (arXiv 2025) only for the specific 97% SR number and the architectural specifics (parallel decoding, action chunking, L1 regression head). 1-2 arXiv-only citations total in paper.

**Recommendation: Option C**. Honest about preprint status while preserving the §V.C.5 argument. Total arXiv-only citation count: 1 (OFT). Total citations with published venue: 7+ (all others).

---

## Verification deferred (Day 4 .bib production)

These citations are listed in §I/§II/§IV/§V/§VI drafts but their venue should be checked when constructing the .bib file:

| Citation | Expected venue | Risk |
|---|---|---|
| Driess et al., 2023 | ICML 2023 (PaLM-E) | LOW |
| Karamcheti et al., 2024 | ICML 2024 (Prismatic VLMs) | LOW |
| Hewitt and Manning, 2019 | NAACL 2019 (structural probing) | LOW |
| Belrose et al., 2023 | arXiv 2303.08112 (tuned lens) — check if published since | MEDIUM |
| Chi et al., 2023 | RSS 2023 (Diffusion Policy) | LOW |
| Liu et al., 2022 | ICLR 2023 (Rectified Flow) | LOW-MEDIUM |
| Kriegeskorte et al., 2008 | Frontiers in Sys. Neuro. 2008 (RSA) | LOW (very old, established) |
| Lynch et al., 2020 | CoRL 2020 (multi-task language imitation) | MEDIUM (specific paper) |

**Action for Day 4**: 30-minute pass: web_search each entry, confirm venue, build .bib entry with proper venue field.

---

## Updated citation strategy for paper drafts

**Drafts to update** (replace year-only refs with venue-explicit):

1. `notes/paper_section_II_draft.md` — §II.A, §II.B citation table, §II.C
2. `notes/paper_section_IIIA_draft.md` — references to OpenVLA, OFT, π₀
3. `notes/paper_section_IV_draft.md` (committed Day 2) — §IV.B backbone citation
4. `notes/paper_section_I_draft.md` (committed Day 2) — §I.0 hook citation, §I.1 background

**Update specifics**:
- "Kim et al., 2024" → "Kim et al., CoRL 2024" (when referring to OpenVLA)
- "Kim et al., 2025" → "Kim et al., 2025 (preprint)" (when referring to OFT)
- "Brohan et al., 2023" → "Brohan et al., CoRL 2023"
- "Black et al., 2024" → "Black et al., RSS 2025" (using publication venue year)
- "Frans et al., 2024" → "Frans et al., ICLR 2025"
- "Lipman et al., 2023" → "Lipman et al., ICLR 2023"
- "Tenney et al., 2019" → "Tenney et al., ACL 2019"
- "Octo team, 2024" → "Octo Model Team et al., RSS 2024"

**Year convention for arXiv-published-later cases**: paper community varies. For consistency, recommend using the venue year (e.g., "Black et al., 2025" for π₀ since published at RSS 2025) with the .bib year=2025 field. This makes the citation feel more authoritative than arXiv year.

**Final arXiv-only count in paper**: 1 (OpenVLA-OFT). Acceptable.

---

## Summary

- Day 2-3 verified citations: 7 (all have published venues except OFT preprint)
- arXiv-only citations after updates: 1 (OpenVLA-OFT, justified by no alternative)
- Deferred to Day 4 .bib production: 8
- All paper drafts need a citation-format pass to insert venue-explicit references (handled in Day 4 .bib construction)

This reduces paper's arXiv-citation footprint substantially. Reviewer perception risk minimized.
