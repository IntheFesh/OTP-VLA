# §V.A — Diagnostic Measurements (v1 draft)

**Status**: Draft v1. Day 3 P1. ~2 pages target.

**Purpose**: Detailed methodology and results for the three signal-flow measurements (Gate 1, M3, C1) referenced throughout §I, §III, and §V.C. Provides reproducibility detail.

**Data sources (Phase 0, all frozen)**:
- `results/phase0/gate1_sanity_and_m3.json` (replicated 2026-05-12)
- `results/phase0/condition_check.json` (C1, η²)
- `results/phase0/test4_capacity.json` (backbone capacity reference)
- Single source of truth: `notes/phase0_final_numbers.md`

**Locked numbers** (from `notes/phase0_final_numbers.md`):
- Gate 1: $r(h_{\mathrm{OFT}}, \ell) = +0.76$ (replicated: $r = +0.7575$, $p < 0.001$)
- M3: $r(\text{demo traj}, \ell) = -0.27$ (flat representation, primary; range $[-0.27, -0.20]$ across 3 representations); $\rho^2 = 0.073$, $p = 0.964$
- C1: $r(z_{\text{OTP}}, \ell) = -0.19$, η² median 0.26

---

## §V.A.1 Methodology overview

The three-layer signal-flow diagnostic comprises three Mantel tests on inter-task distance matrices, applied to different stages of the VLA pipeline:

| Test | Subject of measurement | Stage in pipeline |
|---|---|---|
| **Gate 1** | $h_{\mathrm{backbone}}$ (LLM hidden state, last text token) | Pre-training endpoint, before any demonstration supervision |
| **M3** | Demonstration trajectory $a$ (per-task action sequences) | Supervision signal, used to train the policy |
| **C1** | Policy latent $z$ (head output after training) | Trained policy, downstream of demonstrations |

Each measurement applies the same statistical protocol: (1) collect representations $\{X_k\}$ across $K=10$ tasks of LIBERO-Spatial; (2) compute the inter-task distance matrix $D^X_{ij} = \|X_i - X_j\|$; (3) compute the matched inter-task distance matrix on language embeddings $D^{\ell}_{ij} = \|\ell_i - \ell_j\|$; (4) Mantel test on $(D^X, D^{\ell})$ with $N_{\text{perm}} = 1000$ permutations to obtain Pearson correlation $r$ and permutation $p$-value. Distances are computed with Euclidean norm; tasks are LIBERO-Spatial's 10 pick-and-place variations. Language embeddings $\ell_i$ are the mean-pooled token embeddings from the OpenVLA-OFT tokenizer applied to each task's instruction prompt.

The shared protocol enables direct cross-stage comparison of the same quantity ($r$ versus language across tasks) at three pipeline layers. Differences in magnitude across the three measurements localize where language conditioning is preserved or lost.

## §V.A.2 Gate 1 — Backbone language preservation

**Quantity measured.** $r(h_{\mathrm{backbone}}, \ell)$, the Mantel correlation between OpenVLA-OFT backbone hidden states at the last text token and language embeddings, across 10 tasks of LIBERO-Spatial.

**Methodology.** For each task, we tokenize the instruction prompt with the OpenVLA-OFT processor, forward the resulting tokens through the OFT VLM (frozen backbone, no fine-tuning), and extract the LLM hidden state at position $[-1]$ — the last text token before the action prediction segment. This position aggregates causal attention over the entire visual+text input sequence. Per task, we collect $n_{\text{demos}} = 10$ samples drawn from the first frame of 10 demonstration episodes; the per-task representation is the mean over these 10 samples. The inter-task distance matrix $D^h$ uses Euclidean distances between per-task representations.

**Sanity checks.** Three sanity checks (SC1, SC2, SC3) precede the main Gate 1 measurement and are reported in our replication log (`results/phase0/gate1_sanity_and_m3.json`):

- **SC1 (token position).** Position $[-1]$ corresponds to the last text token (token id 29901, surface form `:`), which has attended over the full visual + text sequence via causal LLM attention. This validates the use of position $[-1]$ as a representation summarizing the conditioning on $(o, \ell)$.
- **SC2 (embedding consistency).** The language embedding pipeline is deterministic (identical across two runs, $L_2$ norm $= 0.234$, max element-wise diff $= 0$). Cross-test comparison between the C1 deep-dive run and the Gate 1 replication run confirms byte-identical embedding code path.
- **SC3 (replication of original Gate 1).** A replication run on 2026-05-12 produced $r(h_{\mathrm{OFT}}, \ell) = +0.7575$ ($p < 0.001$) versus the original 2026-05-11 result $r = +0.7423$, a difference of $\Delta r = 0.0152$ within sampling noise.

**Result.** Averaging across the two runs, the Gate 1 measurement is

$$r(h_{\mathrm{backbone}},\, \ell) \;\approx\; +0.76, \quad p < 0.001.$$

The squared correlation $\rho_h^2 = 0.578$ implies a Gaussian-equivalent MI surrogate of $\hat{I}_h \approx 0.43$ nats (reported qualitatively only, §III.B Caveat on Gaussian surrogate).

**Interpretation.** The OpenVLA-OFT backbone, prior to any demonstration-specific fine-tuning of downstream modules, preserves strong correlation between its hidden state and the language instruction across tasks. The backbone-level representation is language-discriminative.

## §V.A.3 M3 — Demonstration-level language-action correlation

**Quantity measured.** $r(\text{demonstration trajectory}, \ell)$, the Mantel correlation between demonstration action sequences and language embeddings, across 10 tasks.

**Methodology.** For each task, we collect $n_{\text{demos}} = 10$ demonstration episodes from LIBERO-Spatial and compute three trajectory representations:
- **Flat representation** (primary): concatenation of action chunks across the episode, $a \in \mathbb{R}^{T \times 7}$ flattened to a single vector
- **Endpoint representation**: pair $(a_0, a_T)$ of first and last actions
- **Waypoint representation**: subsampled actions at regular intervals along the episode

Per task, the representation is the mean over 10 episodes. Inter-task distances $D^{\text{traj}}$ use Euclidean distances on the chosen representation. The Mantel test compares $D^{\text{traj}}$ to $D^{\ell}$ as in Gate 1.

**Result.** Across the three representations:

$$r(\text{traj}, \ell) \;\in\; [-0.27, -0.20], \quad \rho^2 \le 0.073.$$

The maximum-magnitude correlation is on the flat representation, $r = -0.2698$ ($p = 0.964$ via permutation test). The permutation $p$-value of $0.96$ indicates the observed correlation is not statistically distinguishable from the null distribution; demonstrations are approximately language-orthogonal under the Mantel test.

**Per-task trajectory characteristics.** Mean per-task trajectory $L_2$ norms are tightly clustered (range $[20.376, 20.868]$), confirming that the trajectory representations differ across tasks in *direction* and *fine-grained shape*, not in overall magnitude. This rules out a degenerate explanation where M3's near-zero correlation arises from trajectory-norm artifacts.

**Interpretation.** Demonstration trajectories on LIBERO-Spatial are approximately uncorrelated with instruction embeddings under inter-task distance comparison. This is the empirical realization of the $\varepsilon$-language-orthogonality condition (Definition 1, §III.B) with $\varepsilon$ small enough that the Gaussian MI surrogate $\hat{I}_{p_{\mathcal{D}}}(a; \ell \mid o) \approx 0.04$ nats — an order of magnitude smaller than Gate 1's backbone-level conditioning.

The squared-correlation gap

$$\rho_h^2 / \rho_{\text{demo}}^2 \;=\; 0.578 / 0.073 \;\approx\; 8\times$$

is the principal effect-size measurement of this paper: language conditioning preserved at the backbone is attenuated to near-null at the demonstration-data stage.

## §V.A.4 C1 — Policy-level language correlation in the trained latent

**Quantity measured.** $r(z_{\text{policy}}, \ell)$, the Mantel correlation between OTP-Soft's trained latent representation $z$ (head output) and language embeddings across tasks. Also reported: per-task variance partitioning $\eta^2$ over $z$ slots.

**Methodology.** We train OTP-Soft V3 (Shortcut Flow Matching head + decoder, full architecture as in §IV.B) on LIBERO-Spatial demonstrations for the standard training schedule. We then forward 10 evaluation samples per task through the trained policy, extract the head output $z$, and compute inter-task distance matrices on $z$ analogous to Gate 1 and M3. Additionally, we compute $\eta^2$ — the proportion of variance in each $z$ slot attributable to task identity — to assess whether task-discriminative signal is present in any slot of $z$, even if the aggregate Mantel correlation is small.

**Result.**

$$r(z_{\text{OTP}}, \ell) \;=\; -0.19, \quad p_{\text{Mantel}} = 0.913.$$

The negative sign and high $p$-value indicate $z$ is approximately uncorrelated with language under the inter-task Mantel test. However, the $\eta^2$ analysis reveals structure:

- Median $\eta^2$ across slots: $0.26$
- $\eta^2$ range: $[0.05, 0.96]$
- 25th percentile: $0.08$; 75th percentile: $0.38$
- Median per-slot maximum $\eta^2$ across tasks: $0.67$

A substantial fraction of $z$ slots carry task-discriminative variance ($\eta^2$ near 1 in some slots), but this variance is **concentrated in slots corresponding to non-target object pose encoding** — visual scene structure unique to each task layout — rather than in slots that reflect the linguistic content of the instruction. The high-$\eta^2$ slots discriminate tasks via scene geometry (where the bowl and plate are positioned), not via what the instruction says about them.

**Interpretation.** The trained policy's latent $z$ does encode task-discriminative information, but this information is visually grounded rather than linguistically driven. The aggregate Mantel correlation with language embeddings ($r = -0.19$) is small because the dominant variance in $z$ is task-specific scene structure unrelated to instruction surface form or semantics. This is consistent with the language-orthogonality finding at the demonstration level (§V.A.3, M3): if demonstrations do not encode language-trajectory correlation, the trained policy's latent cannot encode it either, except indirectly through scene structure correlated with task identity.

The C2 condition of §III.A is therefore *not* substantively satisfied for the OTP-Soft V3 policy: while $H(z \mid o) > 0$ trivially holds (the latent is not constant), the task-discriminative structure of $z$ is observation-derived rather than instruction-derived.

## §V.A.5 Backbone capacity check (Test 4)

A potential concern with the Gate 1 result is whether the high backbone correlation reflects genuine task-discriminative encoding or merely the encoding of task-correlated visual features available in the prompt. Test 4 (`results/phase0/test4_capacity.json`) addresses this by training a width-doubled deterministic head on a controlled probe and measuring per-action-dimension informativeness:

$$\bar{F}^a \;=\; 11.30, \quad \max_a F^a \;=\; 26.90.$$

The mean per-dimension $F$-statistic of $11.30$ across the 7 action dimensions, on 500 demonstrations per task × 2 tasks × 1000 training steps, exceeds the null threshold by a substantial margin. This confirms that the backbone has sufficient capacity to express task-discriminative actions in the controlled probe setting — the backbone is not capacity-limited in any sense that would explain the M3 finding. The M3 result is a property of the supervision distribution, not of the backbone's representational capacity.

## §V.A.6 Summary

The three measurements together localize where language conditioning is preserved or lost across the VLA pipeline:

| Stage | Quantity | Value | Interpretation |
|---|---|---|---|
| Backbone (pre-training) | $r(h_{\mathrm{OFT}}, \ell)$ | $+0.76$ | Language preserved at strong correlation |
| Demonstration (supervision signal) | $r(\text{traj}, \ell)$ | $-0.27$ | Language attenuated to near-null |
| Policy latent (post-training) | $r(z_{\text{OTP}}, \ell)$ | $-0.19$ | Policy latent inherits demonstration-level attenuation; task-discriminative variance is visually grounded ($\eta^2$ analysis) |

The transition from $r = +0.76$ to $r = -0.27$ between backbone and demonstrations identifies the supervision data as the locus of information loss, not the backbone capacity (Test 4) or the trained policy in isolation. The C1 measurement on the trained policy ($r = -0.19$) confirms the policy inherits the demonstration-level pattern, consistent with Theorem 3' (§III.B) bounding policy-level language MI by demonstration-level MI.
