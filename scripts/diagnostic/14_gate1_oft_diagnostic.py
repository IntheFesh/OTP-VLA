"""
Gate 1: OFT-backbone-level diagnostic — does the shared OpenVLA-OFT backbone
representation encode task identity via language, visual, or proprio pathway?

V6.1 deep-dive context:
- OFT outer forward is bypassed in our V3 setup. We measure the LLM last-text-token
  hidden state (the representation the OFT action head would consume).
- This is the SHARED backbone representation that BOTH OTP-Soft head and a
  hypothetical OFT-on-this-backbone action head would receive.

Protocol (per V6.1 corrected Gate 1):
- 10 LIBERO-Spatial tasks × 5 demos per task = 50 forward passes
- Extract h_OFT = hidden_states[:, last_text_position, :]
  (T_tok = N_visual + N_text; last_text_position = T_tok - 1)
- Per-task mean h_OFT: (K=10, D=4096)
- Mantel tests:
    D(h_OFT) vs D(visual_CLS)
    D(h_OFT) vs D(language_CLS)
    D(h_OFT) vs D(proprio)  ← fallback pathway test
- Permutation p (1000 shuffles)

Decision matrix (per V6.1 reviewer recommendation):
  r(h_OFT, language) >> r(h_OFT, visual): language-driven → strong framing
  r(h_OFT, visual) >> r(h_OFT, language): visual-driven → strong framing
  r(h_OFT, proprio) >> both: proprio-driven → strong framing variant
  All r ≈ 0: open-question framing (paper still publishable, sell point weaker)
  r(h_OFT, language) > r(h_OFT, visual) by 1.5×: Tier 3 fallback (original framing)

Output: results/phase0/gate1_oft_diagnostic.json
"""

import sys
import json
import numpy as np
import torch
from pathlib import Path
from scipy.spatial.distance import pdist, squareform
from scipy import stats
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
N_DEMOS_PER_TASK = 5
N_PERM = 1000


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


def get_demos_for_task(dataset, task_name, n_demos):
    """Return sample indices (frame=0 preferred) for given task."""
    matched = []
    for i in range(len(dataset)):
        demo_idx, frame_idx = dataset._index[i]
        record = dataset.demo_records[demo_idx]
        if record["npz_path"].parent.name == task_name and frame_idx == 0:
            matched.append(i)
            if len(matched) >= n_demos:
                break
    return matched


def mantel_test(D1, D2, n_perm=N_PERM, seed=20260511):
    """Mantel test on two distance matrices."""
    K = D1.shape[0]
    iu = np.triu_indices(K, k=1)
    r_obs, _ = stats.pearsonr(D1[iu], D2[iu])
    rng = np.random.default_rng(seed)
    null_r = np.zeros(n_perm)
    for i in range(n_perm):
        perm = rng.permutation(K)
        D2_perm = D2[perm][:, perm]
        null_r[i], _ = stats.pearsonr(D1[iu], D2_perm[iu])
    pval = float((null_r >= abs(r_obs)).mean()) if r_obs > 0 else float((null_r <= r_obs).mean())
    return float(r_obs), pval


@torch.no_grad()
def extract_oft_hidden_and_modalities(predictor, dataset, task_names, n_demos, cfg):
    """For each task, extract:
       - h_OFT: last-text-token hidden state from backbone forward (mean over n_demos)
       - visual_CLS: mean-pool of visual_embeds (one demo, frame 0)
       - language_CLS: mean-pool of text_embeds (one demo, frame 0)
       - proprio: 8-dim proprio vector (mean over n_demos)
    """
    backbone = predictor.model.backbone

    if hasattr(backbone.llm_backbone, "get_input_embeddings"):
        embed_tokens = backbone.llm_backbone.get_input_embeddings()
    else:
        embed_tokens = backbone.llm_backbone.embed_tokens

    K = len(task_names)
    D_h = backbone.hidden_dim
    h_OFT = np.zeros((K, D_h), dtype=np.float64)
    visual_CLS = np.zeros((K, D_h), dtype=np.float64)
    language_CLS = np.zeros((K, D_h), dtype=np.float64)
    proprio_vecs = np.zeros((K, 8), dtype=np.float64)

    from PIL import Image
    for ti, task_name in enumerate(task_names):
        demo_indices = get_demos_for_task(dataset, task_name, n_demos)
        if len(demo_indices) < 1:
            raise RuntimeError(f"No demos for {task_name}")
        print(f"  task {ti}: {len(demo_indices)} demos")

        # For h_OFT: average over n_demos
        h_OFT_samples = []
        proprio_samples = []

        for idx in demo_indices:
            sample = dataset[idx]
            raw_batch = collate_fn([sample])
            batch = assemble_batch(raw_batch, predictor.device, predictor.amp_dtype,
                                  cfg.model.otp_head.num_objects)

            with torch.autocast(device_type="cuda", dtype=predictor.amp_dtype):
                out = backbone(image=batch["image"], instruction=batch["instruction"])
            hidden_states = out["hidden_states"]  # (1, T_tok, D_h)
            # Last-text-token = T_tok - 1 (no action tokens since we bypassed OFT outer)
            h_last = hidden_states[0, -1, :].float().cpu().numpy()
            h_OFT_samples.append(h_last)

            proprio_samples.append(batch["proprioception"][0].float().cpu().numpy())

        h_OFT[ti] = np.mean(h_OFT_samples, axis=0)
        proprio_vecs[ti] = np.mean(proprio_samples, axis=0)

        # For visual_CLS / language_CLS: one forward (first demo, frame 0)
        first_sample = dataset[demo_indices[0]]
        raw_b = collate_fn([first_sample])
        batch1 = assemble_batch(raw_b, predictor.device, predictor.amp_dtype,
                               cfg.model.otp_head.num_objects)
        img_np = batch1["image"][0].permute(1, 2, 0).cpu().numpy()
        img_pil = Image.fromarray(img_np)
        prompt = backbone._format_prompt(batch1["instruction"][0])

        with torch.autocast(device_type="cuda", dtype=predictor.amp_dtype):
            proc_out = backbone.processor(
                images=img_pil, text=prompt, return_tensors="pt"
            ).to(predictor.device)
            vision_features = backbone.vision_backbone(proc_out["pixel_values"])
            if isinstance(vision_features, dict):
                vision_features = vision_features.get("last_hidden_state",
                                                     vision_features.get("hidden_states"))
            elif hasattr(vision_features, "last_hidden_state"):
                vision_features = vision_features.last_hidden_state
            visual_embeds = backbone.projector(vision_features)
            text_embeds = embed_tokens(proc_out["input_ids"])
        visual_CLS[ti] = visual_embeds.mean(dim=1)[0].float().cpu().numpy()
        language_CLS[ti] = text_embeds.mean(dim=1)[0].float().cpu().numpy()

    return h_OFT, visual_CLS, language_CLS, proprio_vecs


def main():
    print("=" * 70)
    print("Gate 1: OFT-backbone-level diagnostic (h_OFT vs modalities)")
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

    print(f"\nExtracting h_OFT + visual/language/proprio embeddings for "
          f"{len(PROBE_TASK_NAMES)} tasks × {N_DEMOS_PER_TASK} demos...")
    h_OFT, visual_CLS, language_CLS, proprio_vecs = extract_oft_hidden_and_modalities(
        predictor, dataset, PROBE_TASK_NAMES, N_DEMOS_PER_TASK, cfg
    )
    print(f"  h_OFT shape:        {h_OFT.shape}, L2 mean: {np.linalg.norm(h_OFT, axis=1).mean():.3f}")
    print(f"  visual_CLS shape:   {visual_CLS.shape}, L2 mean: {np.linalg.norm(visual_CLS, axis=1).mean():.3f}")
    print(f"  language_CLS shape: {language_CLS.shape}, L2 mean: {np.linalg.norm(language_CLS, axis=1).mean():.3f}")
    print(f"  proprio shape:      {proprio_vecs.shape}, L2 mean: {np.linalg.norm(proprio_vecs, axis=1).mean():.3f}")

    # Save raw embeddings
    np.save(OUT_DIR / "gate1_h_OFT.npy", h_OFT)
    np.save(OUT_DIR / "gate1_visual_CLS.npy", visual_CLS)
    np.save(OUT_DIR / "gate1_language_CLS.npy", language_CLS)
    np.save(OUT_DIR / "gate1_proprio.npy", proprio_vecs)

    # Pairwise distance matrices
    D_h = squareform(pdist(h_OFT, metric="euclidean"))
    D_vis = squareform(pdist(visual_CLS, metric="euclidean"))
    D_lang = squareform(pdist(language_CLS, metric="euclidean"))
    D_prop = squareform(pdist(proprio_vecs, metric="euclidean"))

    iu = np.triu_indices(D_h.shape[0], k=1)
    print(f"\nPairwise distance summary (10 tasks):")
    print(f"  D(h_OFT):    mean={D_h[iu].mean():.3f}, std={D_h[iu].std():.3f}")
    print(f"  D(visual):   mean={D_vis[iu].mean():.3f}, std={D_vis[iu].std():.3f}")
    print(f"  D(language): mean={D_lang[iu].mean():.3f}, std={D_lang[iu].std():.3f}")
    print(f"  D(proprio):  mean={D_prop[iu].mean():.3f}, std={D_prop[iu].std():.3f}")

    # Mantel tests: h_OFT against each modality
    print(f"\nMantel tests ({N_PERM}-shuffle), h_OFT against modalities:")
    r_h_vis, p_h_vis = mantel_test(D_h, D_vis)
    r_h_lang, p_h_lang = mantel_test(D_h, D_lang)
    r_h_prop, p_h_prop = mantel_test(D_h, D_prop)
    print(f"  h_OFT vs visual:    r = {r_h_vis:+.4f}, p = {p_h_vis:.4f}")
    print(f"  h_OFT vs language:  r = {r_h_lang:+.4f}, p = {p_h_lang:.4f}")
    print(f"  h_OFT vs proprio:   r = {r_h_prop:+.4f}, p = {p_h_prop:.4f}")

    # Verdict
    r_vals = {"language": r_h_lang, "visual": r_h_vis, "proprio": r_h_prop}
    p_vals = {"language": p_h_lang, "visual": p_h_vis, "proprio": p_h_prop}
    dominant = max(r_vals, key=r_vals.get)
    r_dom = r_vals[dominant]
    p_dom = p_vals[dominant]

    # Decision matrix per V6.1 corrected Gate 1
    if r_dom > 0.4 and p_dom < 0.05:
        if dominant == "language":
            verdict = "language_driven_OFT_genuinely_conditions_on_instruction"
            framing = "Path B + Tier 3 (genuine language conditioning, our diagnostic finds OTP-Soft head failed)"
        elif dominant == "visual":
            verdict = "visual_driven_OFT_relies_on_scene_layout"
            framing = "New framing strong: OFT 97% SR is visual-grounded, not language-conditioned"
        else:
            verdict = "proprio_driven_OFT_relies_on_init_state"
            framing = "New framing variant: OFT 97% SR is proprio-grounded"
    elif r_dom > 0.2:
        verdict = "weak_signal_dominant_modality_" + dominant
        framing = "New framing qualified: " + dominant + " has weak but non-zero contribution"
    else:
        verdict = "no_modality_dominant_open_question"
        framing = "Open-question framing: backbone representation not driven by any obvious modality"

    print(f"\nDominant modality: {dominant} (r = {r_dom:+.4f}, p = {p_dom:.4f})")
    print(f"Verdict: {verdict}")
    print(f"Paper framing implication: {framing}")

    summary = {
        "version": "gate1_v1",
        "n_tasks": len(PROBE_TASK_NAMES),
        "n_demos_per_task": N_DEMOS_PER_TASK,
        "n_perm": N_PERM,
        "L2_norms": {
            "h_OFT": float(np.linalg.norm(h_OFT, axis=1).mean()),
            "visual_CLS": float(np.linalg.norm(visual_CLS, axis=1).mean()),
            "language_CLS": float(np.linalg.norm(language_CLS, axis=1).mean()),
            "proprio": float(np.linalg.norm(proprio_vecs, axis=1).mean()),
        },
        "pairwise_distances": {
            "h_OFT_mean": float(D_h[iu].mean()),
            "visual_mean": float(D_vis[iu].mean()),
            "language_mean": float(D_lang[iu].mean()),
            "proprio_mean": float(D_prop[iu].mean()),
        },
        "mantel_h_vs_visual":   {"r": r_h_vis,  "p": p_h_vis},
        "mantel_h_vs_language": {"r": r_h_lang, "p": p_h_lang},
        "mantel_h_vs_proprio":  {"r": r_h_prop, "p": p_h_prop},
        "dominant_modality": dominant,
        "r_dominant": r_dom,
        "p_dominant": p_dom,
        "verdict": verdict,
        "framing_implication": framing,
    }
    out_path = OUT_DIR / "gate1_oft_diagnostic.json"
    with open(out_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\nResults written to {out_path}")


if __name__ == "__main__":
    main()
