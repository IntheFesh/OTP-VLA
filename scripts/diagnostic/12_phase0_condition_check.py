"""
Phase 0 Condition Check: C1 (visual pass-through) + C2 (η² magnitude).

This script does NOT belong to the V6 frozen pre-registration battery. It addresses
two follow-up conditions raised in V6.1 deep-dive review:

C1 (Visual pass-through gate):
    Test whether head z is encoded primarily by visual features (visual pass-through)
    or by language signal (genuine task identity).
    - Extract raw visual_embeds: vision_backbone(image) → projector → mean-pool
    - Extract raw text_embeds: llm.embed_tokens(tokenize(instruction)) → mean-pool
    - Mantel test: D_z vs D_visual, D_z vs D_language
    - Pass: Mantel_language_r > Mantel_visual_r × 1.5

C2 (Effect size of Test 1 significant slices):
    Compute η² for the 18 BH-significant slices from Test 1 to determine if
    head is effectively deterministic (η² > 0.95), medium effect (0.3 ≤ η² ≤ 0.9),
    or noise-dominated (η² < 0.3).

Output: results/phase0/condition_check.json
"""

import sys
import json
import numpy as np
import torch
from pathlib import Path
from scipy import stats
from scipy.spatial.distance import pdist, squareform
from hydra import compose, initialize_config_dir

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

from otp.eval.predictor import OTPSoftPredictor
from otp.data.libero_loader import LIBEROOTPDataset
from otp.train.utils import collate_fn, assemble_batch

PAPER_CKPT_FILE = REPO_ROOT / ".paper_ready_ckpt"
CONFIG_PATH = REPO_ROOT / "configs" / "otp_soft_30e.yaml"
OUT_DIR = REPO_ROOT / "results" / "phase0"
OUT_DIR.mkdir(parents=True, exist_ok=True)

PROBE_TASK_NAMES = [
    "pick_up_the_black_bowl_between_the_plate_and_the_ramekin_and_place_it_on_the_plate",
    "pick_up_the_black_bowl_from_table_center_and_place_it_on_the_plate",
    "pick_up_the_black_bowl_in_the_top_drawer_of_the_wooden_cabinet_and_place_it_on_the_plate",
    "pick_up_the_black_bowl_next_to_the_cookie_box_and_place_it_on_the_plate",
    "pick_up_the_black_bowl_next_to_the_plate_and_place_it_on_the_plate",
    "pick_up_the_black_bowl_next_to_the_ramekin_and_place_it_on_the_plate",
    "pick_up_the_black_bowl_on_the_cookie_box_and_place_it_on_the_plate",
    "pick_up_the_black_bowl_on_the_ramekin_and_place_it_on_the_plate",
    "pick_up_the_black_bowl_on_the_stove_and_place_it_on_the_plate",
    "pick_up_the_black_bowl_on_the_wooden_cabinet_and_place_it_on_the_plate",
]
N_PERM = 1000
LANGUAGE_TO_VISUAL_THRESHOLD = 1.5  # for C1: language_r must exceed visual_r * 1.5


# ============================================================================
# Helpers
# ============================================================================

def load_config():
    with initialize_config_dir(config_dir=str(REPO_ROOT / "configs"), version_base=None):
        return compose(config_name="otp_soft_30e")


def build_dataset(cfg):
    norm_path = getattr(cfg.data, "normalizer_path", None)
    return LIBEROOTPDataset(
        root=Path(cfg.data.root),
        suite=cfg.data.suite,
        grasp_affordance_dir=Path(cfg.data.grasp_affordance_dir),
        normalizer_path=Path(norm_path) if norm_path else None,
        num_grasps_per_object=cfg.model.decoder.num_grasps_per_object,
        num_points=cfg.model.geometry_encoder.num_points,
        horizon=cfg.model.otp_head.horizon,
        train_end_demo=45,
    )


def get_first_sample_for_task(dataset, task_name):
    for i in range(len(dataset)):
        demo_idx, frame_idx = dataset._index[i]
        record = dataset.demo_records[demo_idx]
        if record["npz_path"].parent.name == task_name and frame_idx == 0:
            return i
    for i in range(len(dataset)):
        demo_idx, _ = dataset._index[i]
        record = dataset.demo_records[demo_idx]
        if record["npz_path"].parent.name == task_name:
            return i
    raise RuntimeError(f"No sample for task {task_name}")


def mantel_test(D1, D2, n_perm=N_PERM, seed=20260511):
    """Mantel test: correlation of upper-triangle entries of two distance matrices,
    null via row+column permutation of one matrix.
    """
    K = D1.shape[0]
    iu = np.triu_indices(K, k=1)
    r_obs, _ = stats.pearsonr(D1[iu], D2[iu])
    rng = np.random.default_rng(seed)
    null_r = np.zeros(n_perm)
    for i in range(n_perm):
        perm = rng.permutation(K)
        D2_perm = D2[perm][:, perm]
        null_r[i], _ = stats.pearsonr(D1[iu], D2_perm[iu])
    pval = float((null_r >= r_obs).mean())
    return float(r_obs), pval, null_r


# ============================================================================
# C1: Extract visual_CLS and language_CLS per task
# ============================================================================

@torch.no_grad()
def extract_visual_and_language_embeddings(predictor, dataset, task_names, cfg):
    """For each task, extract:
       - visual_CLS: vision_backbone(image) → projector → mean-pool over T_visual tokens
       - language_CLS: llm.embed_tokens(tokenize(instruction)) → mean-pool over T_text tokens

    These are RAW pre-LLM embeddings (no cross-modal fusion).
    """
    K = len(task_names)
    backbone = predictor.model.backbone

    # ---- Step 1: identify the LLM input embedding layer ----
    try:
        if hasattr(backbone.llm_backbone, "get_input_embeddings"):
            embed_tokens = backbone.llm_backbone.get_input_embeddings()
        elif hasattr(backbone.llm_backbone, "embed_tokens"):
            embed_tokens = backbone.llm_backbone.embed_tokens
        else:
            # For PEFT-wrapped llm_backbone
            base = backbone.llm_backbone.base_model.model if hasattr(backbone.llm_backbone, "base_model") else backbone.llm_backbone
            embed_tokens = base.get_input_embeddings()
        print(f"  LLM input embedding layer: {type(embed_tokens).__name__}")
    except Exception as e:
        raise RuntimeError(f"Cannot locate LLM input embedding: {e}")

    # ---- Step 2: extract per-task embeddings ----
    visual_CLS = np.zeros((K, backbone.hidden_dim), dtype=np.float64)
    language_CLS = np.zeros((K, backbone.hidden_dim), dtype=np.float64)

    for ti, task_name in enumerate(task_names):
        idx = get_first_sample_for_task(dataset, task_name)
        sample = dataset[idx]
        raw_batch = collate_fn([sample])
        batch = assemble_batch(raw_batch, predictor.device, predictor.amp_dtype,
                              cfg.model.otp_head.num_objects)

        # ---- Visual path: image → vision_backbone → projector ----
        image = batch["image"]  # (1, 3, H, W) uint8
        # Apply processor's image transform (the wrapper does this internally;
        # we replicate). Many OpenVLA processors take PIL or numpy; use the
        # wrapper's stored processor.
        from PIL import Image
        img_np = image[0].permute(1, 2, 0).cpu().numpy()  # (H, W, 3) uint8
        img_pil = Image.fromarray(img_np)
        with torch.autocast(device_type="cuda", dtype=predictor.amp_dtype):
            proc_out = backbone.processor(
                images=img_pil, text=batch["instruction"][0],
                return_tensors="pt",
            ).to(predictor.device)
            pixel_values = proc_out["pixel_values"]
            # vision_backbone output → projector
            vision_features = backbone.vision_backbone(pixel_values)
            # vision_features may be (B, T_v, D_v) or a dict; handle both
            if isinstance(vision_features, dict):
                vision_features = vision_features.get("last_hidden_state",
                                                     vision_features.get("hidden_states"))
            elif hasattr(vision_features, "last_hidden_state"):
                vision_features = vision_features.last_hidden_state
            # Project to LLM hidden dim
            visual_embeds = backbone.projector(vision_features)  # (B, T_v, hidden_dim)
            # Mean-pool over T_v tokens
            v_cls = visual_embeds.mean(dim=1)[0]  # (hidden_dim,)

            # ---- Language path: tokenize → embed ----
            text_ids = proc_out["input_ids"]
            text_embeds = embed_tokens(text_ids)  # (B, T_t, hidden_dim)
            l_cls = text_embeds.mean(dim=1)[0]

        visual_CLS[ti] = v_cls.float().cpu().numpy()
        language_CLS[ti] = l_cls.float().cpu().numpy()
        print(f"  task {ti}: visual_CLS L2={np.linalg.norm(visual_CLS[ti]):.3f}, "
              f"language_CLS L2={np.linalg.norm(language_CLS[ti]):.3f}")

    return visual_CLS, language_CLS


# ============================================================================
# C2: η² for Test 1's 18 significant slices
# ============================================================================

def compute_C2_eta_squared(Z, sig_slice_indices):
    """For each significant slice (obj, t), compute multivariate η² and per-dim η².

    Z: (K, N, N_obj, H, D)
    sig_slice_indices: list of (obj, t) tuples

    Multivariate η² for a slice = trace(B) / (trace(B) + trace(W))
    where B and W are between/within-group SSP matrices over 6 dims.

    Per-dim η² for a slice = list of 6 univariate η²
    """
    K, N, N_OBJ, H, D = Z.shape
    N_total = K * N
    results = []

    for (o, t) in sig_slice_indices:
        Z_slice = Z[:, :, o, t, :]  # (K, N, D)
        mu_group = Z_slice.mean(axis=1)  # (K, D)
        mu_grand = Z_slice.reshape(-1, D).mean(axis=0)  # (D,)

        # Multivariate: trace-based η²
        B = np.zeros((D, D))
        W = np.zeros((D, D))
        for k in range(K):
            diff = (mu_group[k] - mu_grand).reshape(-1, 1)
            B += N * (diff @ diff.T)
            for n in range(N):
                diff_n = (Z_slice[k, n] - mu_group[k]).reshape(-1, 1)
                W += diff_n @ diff_n.T
        trace_B = float(np.trace(B))
        trace_W = float(np.trace(W))
        eta2_multi = trace_B / (trace_B + trace_W + 1e-12)

        # Per-dim η²
        per_dim_eta2 = []
        for d in range(D):
            vals = Z_slice[:, :, d]
            mug = vals.mean(axis=1)
            mugd = vals.mean()
            SSB = N * ((mug - mugd) ** 2).sum()
            SSW = ((vals - mug.reshape(-1, 1)) ** 2).sum()
            eta2 = SSB / (SSB + SSW + 1e-12)
            per_dim_eta2.append(float(eta2))

        results.append({
            "obj": int(o), "t": int(t),
            "eta2_multivariate_trace": eta2_multi,
            "trace_B": trace_B, "trace_W": trace_W,
            "per_dim_eta2": per_dim_eta2,
        })

    return results


# ============================================================================
# Main
# ============================================================================

def main():
    print("=" * 70)
    print("Phase 0 Condition Check: C1 (visual pass-through) + C2 (η²)")
    print("=" * 70)

    cfg = load_config()
    dataset = build_dataset(cfg)
    predictor = OTPSoftPredictor(
        ckpt_path=PAPER_CKPT_FILE.read_text().strip(),
        config_path=str(CONFIG_PATH),
        grasp_affordance_dir=str(Path(cfg.data.grasp_affordance_dir)),
        device=torch.device("cuda"),
        log_diagnostics=False,
    )
    predictor.reset(episode_seed=0)

    # ====================================================================
    # C1: visual_CLS vs language_CLS Mantel tests
    # ====================================================================
    print("\n" + "=" * 70)
    print("C1: Visual pass-through gate (Mantel: D_z vs D_visual, D_z vs D_language)")
    print("=" * 70)

    print("\nExtracting visual_CLS and language_CLS for 10 tasks...")
    visual_CLS, language_CLS = extract_visual_and_language_embeddings(
        predictor, dataset, PROBE_TASK_NAMES, cfg
    )

    # Load z_per_task from deep-dive (saved during Q2)
    # Reconstruct: collect head z[obj=0, t=0..4].mean over 20 seeds per task
    print("\nCollecting head z (obj=0, t=0..4) for 10 tasks × 20 seeds...")
    K = len(PROBE_TASK_NAMES)
    z_per_task = np.zeros((K, 30), dtype=np.float64)
    for ti, task_name in enumerate(PROBE_TASK_NAMES):
        idx = get_first_sample_for_task(dataset, task_name)
        sample = dataset[idx]
        raw_batch = collate_fn([sample])
        batch = assemble_batch(raw_batch, predictor.device, predictor.amp_dtype,
                              cfg.model.otp_head.num_objects)
        z_samples = []
        for si in range(20):
            traj = predictor.head_forward_only(batch, seed=si)
            z_samples.append(traj[0, 0, :5, :].cpu().numpy().flatten())
        z_per_task[ti] = np.mean(z_samples, axis=0)
        print(f"  task {ti}: z slice shape={z_per_task[ti].shape}")

    # Pairwise distance matrices
    D_z = squareform(pdist(z_per_task, metric="euclidean"))
    D_visual = squareform(pdist(visual_CLS, metric="euclidean"))
    D_language = squareform(pdist(language_CLS, metric="euclidean"))

    print("\nPairwise distance summary (10 tasks):")
    iu = np.triu_indices(K, k=1)
    print(f"  D_z:        min={D_z[iu].min():.3f}, max={D_z[iu].max():.3f}, mean={D_z[iu].mean():.3f}")
    print(f"  D_visual:   min={D_visual[iu].min():.3f}, max={D_visual[iu].max():.3f}, mean={D_visual[iu].mean():.3f}")
    print(f"  D_language: min={D_language[iu].min():.3f}, max={D_language[iu].max():.3f}, mean={D_language[iu].mean():.3f}")

    print("\nMantel tests (1000-shuffle):")
    r_zv, p_zv, _ = mantel_test(D_z, D_visual)
    r_zl, p_zl, _ = mantel_test(D_z, D_language)
    print(f"  D_z vs D_visual:    Mantel r = {r_zv:+.4f}, p = {p_zv:.4f}")
    print(f"  D_z vs D_language:  Mantel r = {r_zl:+.4f}, p = {p_zl:.4f}")
    print(f"  language/visual r ratio: {r_zl / (r_zv + 1e-9):.2f}")
    print(f"  Threshold for 'language drives z': r_l > r_v × {LANGUAGE_TO_VISUAL_THRESHOLD}")

    # C1 verdict
    if r_zl > LANGUAGE_TO_VISUAL_THRESHOLD * r_zv and r_zl > 0.3:
        c1_verdict = "language_driven_genuine_task_identity"
    elif r_zv > r_zl * 1.5:
        c1_verdict = "visual_pass_through_no_task_conditioning"
    else:
        c1_verdict = "mixed_or_inconclusive_borderline"

    print(f"\n  C1 verdict: {c1_verdict}")

    # ====================================================================
    # C2: η² for Test 1's 18 significant slices
    # ====================================================================
    print("\n" + "=" * 70)
    print("C2: η² for Test 1's significant slices")
    print("=" * 70)

    # Load Test 1 results
    test1_json = OUT_DIR / "test1_head_identifiability.json"
    if not test1_json.exists():
        print(f"  [SKIP] {test1_json} not found")
        c2_results = None
        c2_verdict = "skipped_no_test1_data"
    else:
        with open(test1_json) as f:
            t1 = json.load(f)
        sig_slices = [(s["obj"], s["t"]) for s in t1["slices"] if s.get("bh_significant")]
        print(f"  Found {len(sig_slices)} BH-significant slices from Test 1\n")

        Z = np.load(OUT_DIR / "test1_Z.npy")
        print(f"  Z shape: {Z.shape}")

        c2_records = compute_C2_eta_squared(Z, sig_slices)
        eta2_multi_values = [r["eta2_multivariate_trace"] for r in c2_records]
        eta2_pd_max_values = [max(r["per_dim_eta2"]) for r in c2_records]

        print(f"\n  Multivariate η² (trace-based) summary across 18 sig slices:")
        print(f"    median: {np.median(eta2_multi_values):.4f}")
        print(f"    min:    {np.min(eta2_multi_values):.4f}")
        print(f"    max:    {np.max(eta2_multi_values):.4f}")
        print(f"    p25/p75: {np.percentile(eta2_multi_values, 25):.4f} / "
              f"{np.percentile(eta2_multi_values, 75):.4f}")
        print(f"\n  Per-dim max η² (within each slice) summary:")
        print(f"    median: {np.median(eta2_pd_max_values):.4f}")
        print(f"    min:    {np.min(eta2_pd_max_values):.4f}")
        print(f"    max:    {np.max(eta2_pd_max_values):.4f}")

        # Show top 5 slices by η²
        sorted_records = sorted(c2_records, key=lambda r: -r["eta2_multivariate_trace"])
        print(f"\n  Top 5 slices by η²:")
        for r in sorted_records[:5]:
            per_dim_str = ", ".join([f"{e:.3f}" for e in r["per_dim_eta2"]])
            print(f"    (obj={r['obj']}, t={r['t']}): η²_multi = {r['eta2_multivariate_trace']:.4f}, "
                  f"per_dim η² = [{per_dim_str}]")

        median_eta2 = float(np.median(eta2_multi_values))
        if median_eta2 > 0.95:
            c2_verdict = "head_effectively_deterministic_vestigial_stochasticity"
        elif median_eta2 >= 0.3:
            c2_verdict = "medium_effect_clustered_collapse_framing_holds"
        else:
            c2_verdict = "noise_dominated_clustered_structure_unreliable"
        c2_results = {
            "median_eta2_multivariate": median_eta2,
            "min": float(np.min(eta2_multi_values)),
            "max": float(np.max(eta2_multi_values)),
            "p25": float(np.percentile(eta2_multi_values, 25)),
            "p75": float(np.percentile(eta2_multi_values, 75)),
            "median_per_dim_max": float(np.median(eta2_pd_max_values)),
            "per_slice_records": c2_records,
        }
        print(f"\n  C2 verdict: {c2_verdict}")

    # ====================================================================
    # Save combined results
    # ====================================================================
    summary = {
        "version": "condition_check_v1",
        "C1_visual_vs_language": {
            "Mantel_r_z_vs_visual": r_zv, "Mantel_p_z_vs_visual": p_zv,
            "Mantel_r_z_vs_language": r_zl, "Mantel_p_z_vs_language": p_zl,
            "language_over_visual_ratio": float(r_zl / (r_zv + 1e-9)),
            "threshold": LANGUAGE_TO_VISUAL_THRESHOLD,
            "verdict": c1_verdict,
        },
        "C2_eta_squared": c2_results,
        "C2_verdict": c2_verdict,
    }
    out_path = OUT_DIR / "condition_check.json"
    with open(out_path, "w") as f:
        json.dump(summary, f, indent=2)

    print("\n" + "=" * 70)
    print(f"Combined condition check written to {out_path}")
    print("=" * 70)
    print(f"  C1 (visual pass-through): {c1_verdict}")
    print(f"  C2 (η² magnitude):         {c2_verdict}")


if __name__ == "__main__":
    main()
