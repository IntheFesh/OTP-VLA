# Paper Figures Plan (v1 draft)

**Status**: Draft v1. Day 3 P1. Budget: 6 figures (RA-L typical for empirical paper).

**Data source for each figure listed**. Figure creation (matplotlib/seaborn scripts) deferred to Day 4-5 — this document is the design spec.

---

## Figure 1 — Three-layer signal-flow diagnostic schematic

**Section placement**: §I (alongside hook) or §V.A.1 introduction

**Purpose**: Schematic illustration of Gate 1 / M3 / C1 measurement pipeline. Conveys the conceptual structure of the diagnostic methodology before any numerical results.

**Content**:
- Three boxes representing pipeline stages: Backbone → Demonstration Data → Policy Latent
- Arrows showing data flow: $\ell \to h_{\mathrm{backbone}}$ (vision-language encoding); demonstration data box receives $(o, \ell)$ → produces $a$; policy latent box receives $(o, \ell)$ through head $h_\psi$, produces $z$
- Three measurement annotations: Gate 1 between backbone box and $\ell$; M3 between demonstration data box ($a$) and $\ell$; C1 between policy latent box ($z$) and $\ell$
- Each measurement annotation shows the Mantel correlation magnitude found ($+0.76$, $-0.27$, $-0.19$) as visual sizes (e.g., bar thickness or color intensity)

**Data source**: Phase 0 results (`results/phase0/gate1_sanity_and_m3.json`, `condition_check.json`)

**Style**: Black-and-white schematic, clean lines, suitable for two-column layout. No 3D effects or decorative shading. SVG or PDF format.

---

## Figure 2 — Backbone-vs-demonstration correlation comparison (the headline figure)

**Section placement**: §I.0 hook (alongside the $r = +0.76$ vs $r = -0.27$ displayed equation) OR §V.A.6 summary

**Purpose**: The paper's "money chart" — visualization of the $8\times$ squared-correlation gap.

**Content**:
- Two side-by-side scatter plots, or two columns:
  - Left: Gate 1 — scatter of inter-task distances $D^h_{ij}$ on x-axis vs $D^\ell_{ij}$ on y-axis, regression line with $r = +0.76$ overlaid
  - Right: M3 — same axes but for $D^{\mathrm{traj}}_{ij}$ vs $D^\ell_{ij}$, regression line with $r = -0.27$ overlaid
- Both panels share the same y-axis scale (language distances)
- Annotation in each panel: "$r = +0.76$, $p < 0.001$" / "$r = -0.27$, $p = 0.96$"
- Below: shared bar/text showing $\rho^2$ values: $0.578$ vs $0.073$, ratio $\approx 8\times$

**Data source**: `results/phase0/gate1_sanity_and_m3.json` Mantel test outputs, paired with corresponding distance matrix data

**Style**: Black-and-white preferred (RA-L print-friendly); points as small circles; regression line solid; color used minimally if at all to distinguish the two panels

---

## Figure 3 — CFM decoder sample-quality test (§V.C.2)

**Section placement**: §V.C.2

**Purpose**: Visualize the per-dimension failure pattern from the 5a v2 sample-quality test.

**Content**:
- Two-panel figure:
  - Panel (a): bar chart of L1/std ratio per action dimension (7 bars for pos x/y/z, rot x/y/z, gripper). Horizontal threshold lines at 0.3 (PASS threshold) and 0.8 (FAIL threshold). Bars colored by pass/fail status.
  - Panel (b): scatter plot of predicted vs ground-truth action per dimension (7 small subplots arranged in grid), showing predictions cluster at marginal mean rather than tracking ground truth.

**Data source**: 5a v2 sample quality test outputs (referenced in §V.C.2 table). Specific data should be regenerated from 5a v2 checkpoint forward pass for figure precision; the existing JSON contains the per-dimension L1/std numbers (line 21 of §V.C.2 in consolidated draft has values).

**Style**: Two-panel side-by-side. Panel (a) bar chart with threshold lines. Panel (b) grid of small scatter plots, identity line in light gray, predictions in solid color.

---

## Figure 4 — Oracle ablation (§V.C.3)

**Section placement**: §V.C.3

**Purpose**: Show that decoder loss does not improve with oracle trajectory (rules out moving-target hypothesis).

**Content**:
- Line plot, x-axis = training step / batch, y-axis = decoder loss
- Two curves: normal training (5a v2, head → decoder) and oracle training (5a v3, gt_traj → decoder)
- Both curves converge to similar values ($\sim 1.0$ nats), nearly indistinguishable across the training trajectory
- Annotated: "Decoder loss does not improve with oracle trajectory — failure is intrinsic to CFM decoder, not a moving-target artifact"

**Data source**: 5a v2 and 5a v3 training logs (decoder_loss curves). Need to extract these from log files.

**Style**: Single panel, two curves with different line styles (solid vs dashed) for print compatibility. Legend in corner. Optional: shaded confidence bands if multiple seeds available (likely not, since 1 seed each).

---

## Figure 5 — §V.D ablation results (placeholder)

**Section placement**: §V.D.4.1 + §V.D.4.2

**Purpose**: Visualize Axis 1 (4 architecture variants) and Axis 2 (3 conditioning ablations) results.

**Content (planned)**:
- Two-panel figure:
  - Panel (a) — Axis 1: per-task SR bars for 4 architecture variants (a/b/c/d), 10 tasks each, error bars from 3-seed variance. Optional: aggregate SR with 95% CI to the right of per-task panels.
  - Panel (b) — Axis 2: per-task SR bars for V3 baseline (d) vs three conditioning ablations (e/f/g), 1 seed each.

**Data source**: §V.D ablation results (TBD after Day 4-10 GPU runs)

**Style**: Two-panel side-by-side. Color or hatching to distinguish variants. Practical-equivalence threshold (±2pp) shown as shaded band around V3 baseline reference.

---

## Figure 6 — OTP-Soft architecture diagram (§IV.B)

**Section placement**: §IV.B

**Purpose**: Visual schematic of OTP-Soft's full architecture, showing the C3 boundary explicitly.

**Content**:
- Input boxes: image (multi-view), instruction text, proprioception, object meshes, grasp affordance maps
- Backbone: OpenVLA-OFT (SigLIP+DINOv2 dual-encoder + LLaMA-2 trunk), frozen
- Head $h_\psi$: takes backbone hidden states, produces latent trajectory $z$ (8-step × $N_{\mathrm{obj}}$ × 7-DoF). Internal: Shortcut Flow Matching head
- C3 boundary: dashed vertical line separating head from decoder. Annotation: "$\ell \notin \text{inputs}(\phi_\theta)$ enforced via `_check_forbidden`"
- Decoder $\phi_\theta$: takes $(z, w(o))$ — emphasize $\ell$ does NOT enter. Internal: Shortcut Flow Matching decoder. Output: 7-DoF action chunk (H=8 horizon)
- Loss arrows from training: $\mathcal{L}_{\mathrm{head}}$ on $z$ vs ground truth poses; $\mathcal{L}_{\mathrm{decoder}}$ on action vs ground truth action

**Data source**: §IV.B description (no numerical data needed)

**Style**: Schematic with clear box labels. Color or shading to indicate frozen vs trainable modules. Bold the C3 boundary line as key visual element.

---

## Optional Figure 7 — Supplementary

If extra space available (e.g., journal extended format or supplementary materials):

**§V.A.4 C1 $\eta^2$ distribution analysis**

Histogram of per-slot $\eta^2$ in OTP-Soft V3's latent $z$, showing bimodal distribution: most slots with low $\eta^2$ (noise-like) and a tail of high-$\eta^2$ slots (object-pose encoding). Helps make concrete the "task-discriminative variance concentrated in non-target object slots" finding.

**Data source**: `results/phase0/condition_check.json` C2_eta_squared section.

---

## Production notes

1. **Figure budget**: 6 figures fits standard RA-L. Figure 7 reserved for supplementary if space-constrained.
2. **Color usage**: RA-L permits color but prints both color and grayscale. Design for grayscale-readable; color is enhancement only.
3. **Style consistency**: All figures use same font (Arial/Helvetica, 10pt minimum), same line thickness, same axes formatting. matplotlib `rcParams` template to be defined in Day 4-5 figure-creation work.
4. **Reproducibility**: Each figure should have a matplotlib script stored at `paper/figures/figureN_script.py` so figures can be regenerated from raw data.
5. **Submission format**: Vector format (PDF for matplotlib output, SVG for schematics) preferred over raster (PNG).

---

## Day 4-5 figure creation plan

After Day 3 P1 completion, before GPU ablation launch:
- Day 4 morning: Figures 1, 2, 6 (schematic + headline + architecture; data already available)
- Day 4 afternoon: Figures 3, 4 (CFM failure + oracle; data already available)
- Day 5: Figure 5 placeholder + populate when ablation results arrive (Day 7-10)

Estimated total figure-creation time: ~6-8 hours across Day 4-5.


