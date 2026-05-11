"""
Phase 0a: OpenVLA-OFT Backbone Phrasing-Robustness Test.

Gateway test before full OFT SR ablation (A). Determines whether the backbone
h_OFT representation is robust to instruction paraphrasing.

Hypothesis being tested:
  H_A (OFT visual-grounded): Backbone produces ≈identical h_OFT for paraphrased
       instructions of the same task → cosine similarity > 0.95 within-task.
  H_B (OFT language-fragile): Backbone changes h_OFT substantially under
       paraphrase → cosine similarity < 0.70 within-task.
  H_C (intermediate): Partial robustness → cosine similarity 0.70-0.95.

Protocol:
  - 10 LIBERO-Spatial tasks × 6 phrasings = 60 forward passes
  - Each pass: same image (frame 0 of demo 0), different instruction
  - Extract h_OFT[:, -1, :] (last text token after LLM forward, per Gate 1 setup)
  - Within-task: cosine similarity matrix between 6 phrasings
  - Cross-task: baseline cosine similarity between different tasks (same phrasing)
  - Compare within-task vs cross-task to calibrate

Output: results/phase0/oft_phrasing_robustness.json
"""

import sys
import json
import numpy as np
import torch
from pathlib import Path
from hydra import compose, initialize_config_dir
from PIL import Image

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

from otp.eval.predictor import OTPSoftPredictor
from otp.data.libero_loader import LIBEROOTPDataset
from otp.train.utils import collate_fn, assemble_batch

PAPER_CKPT_FILE = REPO_ROOT / ".paper_ready_ckpt"
CONFIG_PATH = REPO_ROOT / "configs" / "otp_soft_30e.yaml"
OUT_DIR = REPO_ROOT / "results" / "phase0"
OUT_DIR.mkdir(parents=True, exist_ok=True)


# ----------------------------------------------------------------------
# Task-specific spatial phrases (taken from LIBERO-Spatial benchmark)
# ----------------------------------------------------------------------
TASK_NAMES_AND_SPATIAL = [
    ("pick_up_the_black_bowl_between_the_plate_and_the_ramekin_and_place_it_on_the_plate",
     "between the plate and the ramekin"),
    ("pick_up_the_black_bowl_from_table_center_and_place_it_on_the_plate",
     "from table center"),
    ("pick_up_the_black_bowl_in_the_top_drawer_of_the_wooden_cabinet_and_place_it_on_the_plate",
     "in the top drawer of the wooden cabinet"),
    ("pick_up_the_black_bowl_next_to_the_cookie_box_and_place_it_on_the_plate",
     "next to the cookie box"),
    ("pick_up_the_black_bowl_next_to_the_plate_and_place_it_on_the_plate",
     "next to the plate"),
    ("pick_up_the_black_bowl_next_to_the_ramekin_and_place_it_on_the_plate",
     "next to the ramekin"),
    ("pick_up_the_black_bowl_on_the_cookie_box_and_place_it_on_the_plate",
     "on the cookie box"),
    ("pick_up_the_black_bowl_on_the_ramekin_and_place_it_on_the_plate",
     "on the ramekin"),
    ("pick_up_the_black_bowl_on_the_stove_and_place_it_on_the_plate",
     "on the stove"),
    ("pick_up_the_black_bowl_on_the_wooden_cabinet_and_place_it_on_the_plate",
     "on the wooden cabinet"),
]

# ----------------------------------------------------------------------
# 6 paraphrasings (1 identity + 5 paraphrased), all semantic-preserving
# Each template uses {spatial} placeholder
# ----------------------------------------------------------------------
PARAPHRASINGS = {
    "P0_identity":    "pick up the black bowl {spatial} and place it on the plate",
    "P1_word_order":  "place on the plate the black bowl that is {spatial}",
    "P2_synonym":     "grab the dark bowl {spatial} and put it on the plate",
    "P3_passive":     "the black bowl {spatial} should be picked up and placed on the plate",
    "P4_verb_change": "move the black bowl {spatial} onto the plate",
    "P5_compact":     "transfer the black bowl {spatial} to the plate",
}
PARAPHRASE_IDS = list(PARAPHRASINGS.keys())


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
    raise RuntimeError(f"No frame-0 sample for {task_name}")


@torch.no_grad()
def extract_h_OFT(predictor, image_pil, instruction_text):
    """Run backbone forward with custom instruction text; return h_OFT (4096,)."""
    backbone = predictor.model.backbone
    # Wrap the instruction through backbone.forward path
    # The backbone takes (image, instruction) — we mimic Gate 1's call
    # but with custom instruction string instead of dataset-supplied one
    # We bypass dataset-level batching and call backbone directly with the
    # internal preprocessing logic.

    # image as torch tensor for backbone (B, 3, H, W) uint8
    img_np = np.array(image_pil)
    img_tensor = torch.from_numpy(img_np).permute(2, 0, 1).unsqueeze(0).to(predictor.device)

    with torch.autocast(device_type="cuda", dtype=predictor.amp_dtype):
        out = backbone(image=img_tensor, instruction=[instruction_text])
    h = out["hidden_states"][0, -1, :].float().cpu().numpy()
    return h


def cosine(a, b):
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-12))


def main():
    print("=" * 70)
    print("Phase 0a: OFT Backbone Phrasing-Robustness Test")
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

    K = len(TASK_NAMES_AND_SPATIAL)
    P = len(PARAPHRASE_IDS)
    D = predictor.model.backbone.hidden_dim

    print(f"\n10 tasks × {P} phrasings = {K * P} forward passes\n")

    # Generate all 60 instructions
    instructions = {}  # (task_idx, phrase_id) -> instruction string
    for ti, (task_name, spatial) in enumerate(TASK_NAMES_AND_SPATIAL):
        for pid, template in PARAPHRASINGS.items():
            instructions[(ti, pid)] = template.format(spatial=spatial)

    # Verify a few examples
    print("Sample instructions for task 0 (spatial = 'between the plate and the ramekin'):")
    for pid in PARAPHRASE_IDS:
        print(f"  {pid}: {instructions[(0, pid)]!r}")
    print()

    # Extract h_OFT for all (task, phrasing) combinations
    h_matrix = np.zeros((K, P, D), dtype=np.float64)
    images = {}  # cache PIL images per task
    print("Running backbone forwards...")
    for ti, (task_name, spatial) in enumerate(TASK_NAMES_AND_SPATIAL):
        idx = get_first_sample(dataset, task_name)
        sample = dataset[idx]
        raw_batch = collate_fn([sample])
        batch = assemble_batch(raw_batch, predictor.device, predictor.amp_dtype,
                              cfg.model.otp_head.num_objects)
        img_np = batch["image"][0].permute(1, 2, 0).cpu().numpy()
        img_pil = Image.fromarray(img_np)
        images[ti] = img_pil

        for pj, pid in enumerate(PARAPHRASE_IDS):
            inst = instructions[(ti, pid)]
            h = extract_h_OFT(predictor, img_pil, inst)
            h_matrix[ti, pj] = h
        print(f"  task {ti} done (h L2: {[f'{np.linalg.norm(h_matrix[ti,p]):.1f}' for p in range(P)]})")

    # Save raw matrix
    np.save(OUT_DIR / "oft_phrasing_h_matrix.npy", h_matrix)
    print(f"\nh_matrix shape = {h_matrix.shape}, saved.")

    # -------------------------------------------------------------
    # Within-task analysis: cosine similarity across phrasings
    # -------------------------------------------------------------
    print("\n" + "=" * 70)
    print("WITHIN-TASK: cosine similarity across 6 phrasings (per task)")
    print("=" * 70)
    within_task_cosines = np.zeros((K, P, P), dtype=np.float64)
    within_task_L2 = np.zeros((K, P, P), dtype=np.float64)
    for ti in range(K):
        for i in range(P):
            for j in range(P):
                within_task_cosines[ti, i, j] = cosine(h_matrix[ti, i], h_matrix[ti, j])
                within_task_L2[ti, i, j] = float(np.linalg.norm(h_matrix[ti, i] - h_matrix[ti, j]))

    # Within-task off-diagonal statistics
    iu = np.triu_indices(P, k=1)
    within_off_diag_cosines_per_task = []
    within_off_diag_L2_per_task = []
    for ti in range(K):
        within_off_diag_cosines_per_task.append(within_task_cosines[ti][iu].tolist())
        within_off_diag_L2_per_task.append(within_task_L2[ti][iu].tolist())

    all_within_cosines = np.array([within_task_cosines[ti][iu] for ti in range(K)]).flatten()
    all_within_L2 = np.array([within_task_L2[ti][iu] for ti in range(K)]).flatten()
    print(f"  Within-task cosine: median={np.median(all_within_cosines):.4f}, "
          f"min={np.min(all_within_cosines):.4f}, max={np.max(all_within_cosines):.4f}, "
          f"std={np.std(all_within_cosines):.4f}")
    print(f"  Within-task L2:     median={np.median(all_within_L2):.2f}, "
          f"min={np.min(all_within_L2):.2f}, max={np.max(all_within_L2):.2f}")

    # -------------------------------------------------------------
    # Cross-task baseline: cosine across tasks, same phrasing
    # -------------------------------------------------------------
    print("\n" + "=" * 70)
    print("CROSS-TASK BASELINE: cosine across 10 tasks (per phrasing)")
    print("=" * 70)
    cross_task_cosines = np.zeros((P, K, K), dtype=np.float64)
    cross_task_L2 = np.zeros((P, K, K), dtype=np.float64)
    for pj in range(P):
        for i in range(K):
            for j in range(K):
                cross_task_cosines[pj, i, j] = cosine(h_matrix[i, pj], h_matrix[j, pj])
                cross_task_L2[pj, i, j] = float(np.linalg.norm(h_matrix[i, pj] - h_matrix[j, pj]))

    iu_k = np.triu_indices(K, k=1)
    all_cross_cosines = np.array([cross_task_cosines[pj][iu_k] for pj in range(P)]).flatten()
    all_cross_L2 = np.array([cross_task_L2[pj][iu_k] for pj in range(P)]).flatten()
    print(f"  Cross-task cosine: median={np.median(all_cross_cosines):.4f}, "
          f"min={np.min(all_cross_cosines):.4f}, max={np.max(all_cross_cosines):.4f}")
    print(f"  Cross-task L2:     median={np.median(all_cross_L2):.2f}, "
          f"min={np.min(all_cross_L2):.2f}, max={np.max(all_cross_L2):.2f}")

    # -------------------------------------------------------------
    # Decision matrix
    # -------------------------------------------------------------
    median_within = np.median(all_within_cosines)
    median_cross = np.median(all_cross_cosines)
    median_within_L2 = np.median(all_within_L2)
    median_cross_L2 = np.median(all_cross_L2)
    ratio_L2 = median_within_L2 / median_cross_L2

    print("\n" + "=" * 70)
    print("Verdict:")
    print("=" * 70)
    print(f"  Median within-task cosine:  {median_within:.4f}")
    print(f"  Median cross-task cosine:   {median_cross:.4f}")
    print(f"  Cosine gap (within - cross): {median_within - median_cross:+.4f}")
    print(f"  L2 ratio (within / cross):  {ratio_L2:.4f}")
    print()

    if median_within > 0.95 and ratio_L2 < 0.30:
        verdict = "robust_strong_supports_explanation_A"
        framing = ("Backbone is highly paraphrase-robust. h_OFT for same-task paraphrasings "
                  "are ≈identical. This strongly suggests OFT inference at action level is "
                  "also paraphrase-robust, supporting explanation A (OFT is visual-grounded "
                  "and 97% SR does not depend on language conditioning).")
    elif median_within > 0.85 and ratio_L2 < 0.50:
        verdict = "robust_moderate_likely_supports_A"
        framing = ("Backbone is moderately paraphrase-robust. h_OFT shifts modestly under "
                  "paraphrase but remains much more similar within-task than cross-task. "
                  "Likely supports explanation A but full OFT SR measurement (A) is needed "
                  "for confirmation.")
    elif median_within > 0.70:
        verdict = "intermediate_ambiguous"
        framing = ("Backbone shows partial paraphrase-robustness. Within-task representations "
                  "shift substantially. Cannot distinguish explanations A vs B vs C without "
                  "direct OFT SR measurement.")
    else:
        verdict = "fragile_likely_supports_B_or_C"
        framing = ("Backbone is paraphrase-fragile: h_OFT changes substantially under "
                  "paraphrase. This suggests OFT inference depends on surface language form, "
                  "supporting explanation B or C (OFT has architectural language path).")

    print(f"  Verdict: {verdict}")
    print(f"  Framing implication: {framing}")

    summary = {
        "version": "oft_phrasing_robustness_v1",
        "n_tasks": K,
        "n_phrasings": P,
        "phrasing_ids": PARAPHRASE_IDS,
        "phrasing_templates": PARAPHRASINGS,
        "median_within_task_cosine": float(median_within),
        "median_cross_task_cosine": float(median_cross),
        "median_within_task_L2": float(median_within_L2),
        "median_cross_task_L2": float(median_cross_L2),
        "cosine_gap": float(median_within - median_cross),
        "L2_ratio_within_over_cross": float(ratio_L2),
        "within_task_cosines_per_task": within_off_diag_cosines_per_task,
        "within_task_L2_per_task": within_off_diag_L2_per_task,
        "verdict": verdict,
        "framing_implication": framing,
    }
    out_path = OUT_DIR / "oft_phrasing_robustness.json"
    with open(out_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\nResults written to {out_path}")
    print(f"Raw h_matrix saved to {OUT_DIR / 'oft_phrasing_h_matrix.npy'}")


if __name__ == "__main__":
    main()
