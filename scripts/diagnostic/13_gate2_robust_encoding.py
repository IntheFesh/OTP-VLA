"""
Gate 2: Robust encoding check for D_language vs D_visual claim.

V6.1 deep-dive raised concerns:
1. Mean-pool may be a poor proxy for language task discriminability
2. Raw embedding L2 norms differ massively (visual ~14.5 vs language ~0.31, 47× ratio)
   so D_language < D_visual could be a norm artifact, not a task-signal artifact

This script computes D_language and D_visual under:
  4 encoding methods: mean-pool, CLS (first token), last-token, full-seq (mean pairwise)
  2 normalizations:   raw, L2-normalized
= 16 (D_language, D_visual) pairs total.

If all 8 ratios (D_language / D_visual) are << 1, the benchmark-level claim
"language signal is dilute on LIBERO-Spatial" is ROBUST.
If any ratio approaches 1, claim must be narrowed.

Output: results/phase0/gate2_robust_encoding.json
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


def get_first_sample(dataset, task_name):
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
    raise RuntimeError(f"no sample for {task_name}")


@torch.no_grad()
def extract_token_sequences(predictor, dataset, task_names, cfg):
    """For each task: extract full visual_embeds (B, T_v, D) and text_embeds (B, T_t, D).

    Returns:
        visual_seqs: list of (T_v_i, D) tensors per task
        text_seqs:   list of (T_t_i, D) tensors per task
    """
    backbone = predictor.model.backbone

    # Get LLM input embedding layer
    if hasattr(backbone.llm_backbone, "get_input_embeddings"):
        embed_tokens = backbone.llm_backbone.get_input_embeddings()
    else:
        embed_tokens = backbone.llm_backbone.embed_tokens

    visual_seqs = []
    text_seqs = []
    from PIL import Image

    for task_name in task_names:
        idx = get_first_sample(dataset, task_name)
        sample = dataset[idx]
        raw_batch = collate_fn([sample])
        batch = assemble_batch(raw_batch, predictor.device, predictor.amp_dtype,
                              cfg.model.otp_head.num_objects)

        image = batch["image"]
        img_np = image[0].permute(1, 2, 0).cpu().numpy()
        img_pil = Image.fromarray(img_np)
        instruction = batch["instruction"][0]
        # Apply OFT prompt template (LIBERO convention)
        prompt = backbone._format_prompt(instruction)

        with torch.autocast(device_type="cuda", dtype=predictor.amp_dtype):
            proc_out = backbone.processor(
                images=img_pil, text=prompt, return_tensors="pt",
            ).to(predictor.device)
            pixel_values = proc_out["pixel_values"]
            vision_features = backbone.vision_backbone(pixel_values)
            if isinstance(vision_features, dict):
                vision_features = vision_features.get("last_hidden_state",
                                                     vision_features.get("hidden_states"))
            elif hasattr(vision_features, "last_hidden_state"):
                vision_features = vision_features.last_hidden_state
            visual_embeds = backbone.projector(vision_features)  # (1, T_v, D)
            text_ids = proc_out["input_ids"]
            text_embeds = embed_tokens(text_ids)  # (1, T_t, D)

        visual_seqs.append(visual_embeds[0].float().cpu().numpy())
        text_seqs.append(text_embeds[0].float().cpu().numpy())

    return visual_seqs, text_seqs


def encode_meanpool(seqs):
    """Each seq (T, D) → (D,) mean over T."""
    return np.stack([s.mean(axis=0) for s in seqs])


def encode_cls(seqs):
    """First token of each seq."""
    return np.stack([s[0] for s in seqs])


def encode_last(seqs):
    """Last token of each seq (most informative for autoregressive LLM)."""
    return np.stack([s[-1] for s in seqs])


def encode_fullseq_distance_matrix(seqs):
    """For full-seq encoding: pairwise distance = avg distance over all token pairs.

    For each pair (i, j): D_ij = mean_t1 mean_t2 ||seq_i[t1] - seq_j[t2]||_2

    Returns (K, K) distance matrix directly (skip the per-task vector representation).
    """
    K = len(seqs)
    D = np.zeros((K, K))
    for i in range(K):
        for j in range(i + 1, K):
            s_i = seqs[i]  # (T_i, D)
            s_j = seqs[j]  # (T_j, D)
            # Pairwise token distance
            diff = s_i[:, None, :] - s_j[None, :, :]  # (T_i, T_j, D)
            d = np.linalg.norm(diff, axis=-1)  # (T_i, T_j)
            D[i, j] = D[j, i] = d.mean()
    return D


def pairwise_summary(vecs_or_D, label="", normalize=False):
    """If vecs_or_D is (K, D) vectors → optionally L2-normalize → return pairwise D matrix.
    If it's already a (K, K) distance matrix → just return it."""
    if vecs_or_D.ndim == 2 and vecs_or_D.shape[0] == vecs_or_D.shape[1]:
        # It's already a distance matrix (full-seq case)
        D = vecs_or_D
    else:
        # It's vectors (K, D)
        V = vecs_or_D.copy()
        if normalize:
            norms = np.linalg.norm(V, axis=1, keepdims=True) + 1e-12
            V = V / norms
        D = squareform(pdist(V, metric="euclidean"))
    K = D.shape[0]
    iu = np.triu_indices(K, k=1)
    return {
        "mean": float(D[iu].mean()),
        "min": float(D[iu].min()),
        "max": float(D[iu].max()),
        "std": float(D[iu].std()),
        "D_matrix": D.tolist(),
    }


def main():
    print("=" * 70)
    print("Gate 2: Robust Encoding Check (4 encodings × 2 normalizations)")
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

    print(f"\nExtracting visual_embeds and text_embeds for {len(PROBE_TASK_NAMES)} tasks...")
    visual_seqs, text_seqs = extract_token_sequences(
        predictor, dataset, PROBE_TASK_NAMES, cfg
    )
    print(f"  visual_seqs T range: {[v.shape[0] for v in visual_seqs]}")
    print(f"  text_seqs T range:   {[t.shape[0] for t in text_seqs]}")
    print(f"  visual_seqs mean L2 norm per token: "
          f"{np.mean([np.linalg.norm(v, axis=1).mean() for v in visual_seqs]):.3f}")
    print(f"  text_seqs   mean L2 norm per token: "
          f"{np.mean([np.linalg.norm(t, axis=1).mean() for t in text_seqs]):.3f}")

    # Encoders
    encoders = {
        "mean_pool": encode_meanpool,
        "cls": encode_cls,
        "last": encode_last,
    }

    results = {"per_encoding": {}}

    print("\n" + "=" * 70)
    print(f"{'Encoding':<15}{'Norm':<8}{'D_lang':>10}{'D_vis':>10}{'Ratio':>10}")
    print("=" * 70)

    for enc_name, enc_fn in encoders.items():
        visual_vecs = enc_fn(visual_seqs)
        text_vecs = enc_fn(text_seqs)

        for norm_mode in [False, True]:
            norm_label = "L2" if norm_mode else "raw"
            d_vis = pairwise_summary(visual_vecs, normalize=norm_mode)
            d_lang = pairwise_summary(text_vecs, normalize=norm_mode)
            ratio = d_lang["mean"] / (d_vis["mean"] + 1e-12)

            results["per_encoding"][f"{enc_name}_{norm_label}"] = {
                "D_visual": d_vis, "D_language": d_lang,
                "ratio_lang_over_vis": float(ratio),
            }
            print(f"{enc_name:<15}{norm_label:<8}{d_lang['mean']:>10.4f}"
                  f"{d_vis['mean']:>10.4f}{ratio:>10.4f}")

    # Full-seq is a special case — only meaningful in raw form
    print(f"\nFull-seq (pairwise token-token mean distance, raw only)...")
    D_vis_full = encode_fullseq_distance_matrix(visual_seqs)
    D_lang_full = encode_fullseq_distance_matrix(text_seqs)
    d_vis_full = pairwise_summary(D_vis_full)
    d_lang_full = pairwise_summary(D_lang_full)
    ratio_full = d_lang_full["mean"] / (d_vis_full["mean"] + 1e-12)
    results["per_encoding"]["fullseq_raw"] = {
        "D_visual": d_vis_full, "D_language": d_lang_full,
        "ratio_lang_over_vis": float(ratio_full),
    }
    print(f"{'fullseq':<15}{'raw':<8}{d_lang_full['mean']:>10.4f}"
          f"{d_vis_full['mean']:>10.4f}{ratio_full:>10.4f}")

    print("\n" + "=" * 70)
    print("Robustness verdict:")
    print("=" * 70)
    all_ratios = [r["ratio_lang_over_vis"] for r in results["per_encoding"].values()]
    max_ratio = max(all_ratios)
    print(f"  All ratios: {[f'{r:.4f}' for r in all_ratios]}")
    print(f"  Max ratio:  {max_ratio:.4f}")

    if max_ratio < 0.3:
        verdict = "robust_strong: D_lang << D_vis under all 7 encoding/norm combos"
    elif max_ratio < 1.0:
        verdict = "robust_qualified: D_lang < D_vis under all combos but some close"
    else:
        verdict = "fails: some encoding shows D_lang ≈ D_vis or larger — claim narrowed"

    print(f"  Verdict: {verdict}")
    results["max_ratio"] = max_ratio
    results["verdict"] = verdict

    out_path = OUT_DIR / "gate2_robust_encoding.json"
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults written to {out_path}")


if __name__ == "__main__":
    main()
