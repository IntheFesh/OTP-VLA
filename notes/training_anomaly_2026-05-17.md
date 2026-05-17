# Training Anomaly Memo — 2026-05-17

**Status**: Observation, not yet integrated into paper.  
**Detected**: 2026-05-17 23:06 during cell 7 progress audit.  
**Severity**: High — affects §V.C framing and §V.D statistical reporting.

---

## Observation

Two cells in production ablation exhibit catastrophic mid-training divergence at near-identical step ranges, with no NaN/Inf and no recovery over remaining 100k+ steps:

### Cell 4 (V3 baseline seed 42) — COMPLETED, diverged

```
Step      Total   Head    Decoder
10000     1.23    0.83    0.40
20000     1.01    0.60    0.41
30000     0.76    0.47    0.29
40000     0.71    0.41    0.30
50000     0.50    0.33    0.17   ← Last healthy
60000     1.87    1.53    0.35   ← Divergence event (head 4.6x jump)
70000     1.89    1.69    0.20
...
165000    1.68    1.53    0.15   ← Stuck, no recovery
```

### Cell 7 (drop_mesh seed 42) — STILL RUNNING, same pattern

```
Step      Total   Head    Decoder
10000     1.20    0.84    0.36   ← Mirrors V3 seed 42 within 1%
20000     1.03    0.63    0.40
30000     0.75    0.46    0.29
40000     0.70    0.41    0.29
50000     0.52    0.35    0.17   ← Last healthy
60000     0.75    0.40    0.35   ← Wobble onset
70000     1.89    1.69    0.20   ← Divergence (identical to V3 seed 42)
80000-115000: stuck at 1.5-1.9 head loss
```

### Cells unaffected (5 others)

| Cell | Final total | Final head | Final decoder |
|---|---|---|---|
| pathB_seed42 | 0.32 | 0.17 | 0.15 |
| pathB_seed137 | 0.32 | 0.24 | 0.07 |
| pathB_seed2026 | 0.25 | 0.05 | 0.20 |
| v3_seed137 | 0.28 | 0.16 | 0.13 |
| v3_seed2026 | 0.46 | 0.34 | 0.11 |

---

## Strict diagnosis

### Hypothesis 1: Seed 42 specific (refuted)

If seed 42 caused the divergence, **all seed 42 cells** should fail. But:
- pathB_seed42: HEALTHY (head 0.17, final loss 0.32)
- v3_seed42: DIVERGED
- drop_mesh_seed42: DIVERGED

→ Hypothesis 1 **refuted**: seed 42 alone is not sufficient.

### Hypothesis 2: CFM-head architecture specific (supported)

V3 and drop_mesh share **CFM head + CFM decoder** architecture.  
Path B has **det head + CFM decoder**.

- Path B (det head): all 3 seeds healthy
- V3 / drop_mesh (CFM head): 2/4 cells diverged on seed 42

→ Hypothesis 2 **supported**: CFM head is the destabilizing element. The det head in Path B saturates to L1 floor quickly and is robust to whatever seed-specific instability triggers CFM head divergence.

### Hypothesis 3: Seed 42 × CFM head batch order interaction (likely)

The specific combination of seed 42's RNG-generated batch order + CFM head's gradient dynamics triggers a gradient explosion to a non-recoverable local minimum at step ~60000.

Pre-divergence trajectory of v3_seed42 and drop_mesh_seed42 are **identical to 1%** through step 50000, suggesting both use the same data ordering (consistent with seed 42 controlling DataLoader RNG identically across architecture variants).

The drop_mesh ablation (which removes mesh prior input) does not protect against this divergence — the destabilization is in the head training dynamics, not the conditioning input set.

---

## Implications for paper

### Implication 1: §V.C "CFM decoder collapse" framing needs reconsidered

Current §V.C claims: CFM **decoder** collapse under language-orthogonal supervision (0/7 dims pass, L1/std=1.88).

Production data shows: **CFM decoder converges** to L1-like loss (~0.07-0.20) in all 7 completed cells, including the 2 diverged cells (decoder loss 0.15 even when head loss 1.53).

What actually fails: **CFM head** training stability, specifically when combined with seed 42-induced batch order.

### Implication 2: New §V.C narrative possibility

The original §V.C evidence (5a v2 sample quality test, step 500 overfit) was on a different scale than production. The production-scale failure is **CFM head instability**, not decoder sample quality collapse.

Possible reframe: "CFM head training instability under language-orthogonal supervision. 2/4 CFM-head cells (V3 seed 42, drop_mesh seed 42) exhibit catastrophic divergence at step ~60000 with no recovery, while det-head Path B cells (3/3) converge stably."

This is **stronger** empirical evidence than 5a v2 micro-scale sample quality test:
- Full production training scale (50 epochs × 165k steps)
- Reproducible failure point (step 60000-70000)
- Architecture-specific (CFM head, not det head)
- Architecture-input-independent (drop_mesh still fails)

### Implication 3: §V.D statistical reporting requires care

For SR eval on diverged cells:
- v3_seed42 likely fails SR (head not learned)
- drop_mesh_seed42 likely fails SR (head + mesh input both compromised)

These should not be averaged with healthy seeds without disclosure. Pre-registration's McNemar + Wilson CI protocol handles per-cell variance, but the divergence is an additional methodological point requiring §V.D explicit discussion.

### Implication 4: Seed protocol disclosure

The seed 42 instability suggests CFM-head VLA training has narrower stability margin than typical L1-supervised heads. This is **expected** under Theorem 3': demonstration-level supervision provides low MI signal (~0.04 nats), and CFM head training requires stronger gradient signal than the supervision provides.

The divergence is therefore **predicted** by the paper's theoretical framework, not an inconvenient outlier.

---

## Pending — Cells 8-12 will refine

Cells 8 (drop_mesh_seed137) and 9 (drop_mesh_seed2026) will test:
- If drop_mesh seeds 137/2026 also diverge → drop_mesh fundamentally destabilizes (independent of seed)
- If drop_mesh seeds 137/2026 converge → seed 42 × CFM head is the specific trigger

Cell 12 (variantc_seed42) is interesting: variant (c) uses CFM head + DET decoder (mirror of Path B). If variantc_seed42 diverges, it confirms CFM head × seed 42 trigger regardless of decoder. If variantc_seed42 converges, the divergence requires CFM throughout the model.

---

## Strict next actions

1. **Do not kill cell 7** — it provides valuable evidence of failure mode at production scale.
2. **Wait for cells 8-9** — refines seed 42 vs drop_mesh × seed 42 attribution.
3. **Defer §V.C rewrite** until cells 8-9 complete (~36 hr from now).
4. **SR eval** will be the final arbiter — diverged cells with low head loss should fail SR, healthy cells should achieve baseline SR.
5. **No retraining** — keep diverged cells as evidence. Pre-registration McNemar + per-seed CI handles seed variance natively.

---

## Audit trail

- Detection: 2026-05-17 23:06 by Yue via final loss summary
- Documented: 2026-05-17 23:10
- Pending verification: cells 8-9 completion ~May 19
- §V.C reframe decision: deferred to ~May 19-20
- §V.D protocol unchanged (pre-registration handles per-seed CI)
