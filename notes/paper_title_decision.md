# Paper Title Finalization (v1 draft)

**Status**: Draft v1. Day 3 P1.

---

## Title candidates (userMemories #16 + new candidates from §I framing)

### Candidate 1 — Catchy (userMemories carry-over)
**"Where Did the Language Go? Tracing Conditioning Signal Loss in VLA Policies"**

Pros:
- Memorable, reviewer-quotable
- "Tracing" emphasizes diagnostic methodology contribution
- The question framing (where did X go) aligns with hook ($r = 0.76$ vs $-0.27$ findings)
- Distinctive — unlikely to confuse with other VLA papers

Cons:
- "Where Did the Language Go?" implies language *was* there and got lost — interpretively this is what we find (backbone has it, demonstration loses it), but assumes reader buys the framing immediately
- May be perceived as less serious by some venues' reviewers

### Candidate 2 — Conservative (userMemories carry-over, updated)
**"Modality-Faithful VLA Policies: Diagnostic Framework and Theoretical Bound"**

Pros:
- Professional, standard academic register
- "Diagnostic Framework" + "Theoretical Bound" accurately describe contributions
- "Modality-Faithful" hints at the C2/C3 condition framing without requiring reader to immediately grok the diagnostic

Cons:
- "Modality-Faithful" terminology was tied to the Path B fix-proposal framing; the paper now does NOT propose modality-faithful architecture as a solution
- Slightly generic compared to Candidate 1
- The original userMemories version said "Diagnostic Framework AND Architectural Recovery" — paper no longer does architectural recovery, so this version drops that

### Candidate 3 — New, methodological emphasis
**"Signal-Flow Diagnostics for Vision-Language-Action Policies: Why High Task Success Does Not Imply Language Conditioning"**

Pros:
- Two-part title structure (common in RA-L / IEEE journals)
- Main claim explicit in the subtitle
- "Signal-Flow Diagnostics" introduces our methodology name
- Direct, no implicit framing required

Cons:
- Long (15 words)
- Subtitle clause is declarative and may sound confrontational to some readers
- "Vision-Language-Action" spelled out adds length

### Candidate 4 — New, theory-first emphasis
**"Demonstration-Supervised VLA Policies Are Bounded by Demonstration-Level Language Information"**

Pros:
- States Theorem 3' directly as the title claim
- Theoretically precise
- Reviewers in theory-leaning venues may prefer

Cons:
- Long and dense
- Diagnostic-methodology contribution underemphasized
- Less "quotable" or memorable than Candidate 1

### Candidate 5 — New, hybrid
**"Tracing Language Loss in Vision-Language-Action Policies: A Diagnostic Framework with Theoretical Bound"**

Pros:
- Combines Candidate 1's "tracing" verb with Candidate 2's professional register
- "Diagnostic Framework with Theoretical Bound" accurately captures the two main contributions
- Length is moderate (15 words)

Cons:
- "Tracing Language Loss" is slightly more clinical / less memorable than "Where Did the Language Go?"
- Hybrid character may feel like a compromise

---

## Recommendation

**Primary: Candidate 1 — "Where Did the Language Go? Tracing Conditioning Signal Loss in VLA Policies"**

Reasoning:
- RA-L published papers and recent CoRL workshop papers tend toward more direct, question-based titles for empirical findings papers
- The hook ($r = 0.76$ vs $-0.27$ contrast) is the paper's quotable moment; the title should mirror its tone
- "Tracing Conditioning Signal Loss" precisely names the diagnostic methodology contribution
- Risk of "implies language was there and got lost" framing is actually accurate — the backbone preserves it, demonstrations attenuate it. Reviewer reading title sees framing immediately, can either buy in or push back, but no confusion

**Backup: Candidate 5 — "Tracing Language Loss in Vision-Language-Action Policies: A Diagnostic Framework with Theoretical Bound"**

Reasoning:
- If reviewer feedback suggests Candidate 1 is too informal for the target venue, Candidate 5 retains the "tracing" verb and main thrust while shifting to a more conservative two-part structure

**Submission target sequencing**:
- RA-L (rolling deadline, primary target): use Candidate 1
- CoRL Workshop (secondary, non-archival): use Candidate 1 (workshop tone permits catchy titles)
- If targeting NeurIPS Datasets and Benchmarks track in future: Candidate 5 or new Candidate emphasizing the M3 protocol as a benchmark-evaluation methodology

---

## Decision (await Yue over-rule)

**Recommended: Candidate 1**
**Backup: Candidate 5**

If you accept primary: Day 3 work proceeds with this title used in §I draft header and abstract framing.

If you prefer different candidate or want to draft a new one: tell me your preference, I update §I draft and abstract accordingly in next batch.


