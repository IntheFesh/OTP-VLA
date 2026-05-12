"""
Step 6.0 Inference Diagnostic - H1 (CFM noise floor) vs H2 (moving target).

Methodology:
  1. Load 5a final ckpt (175 trainable keys: decoder/head/geometry_encoder)
  2. Build training-subset dataset (LIBEROOTPDataset train_end_demo=5)
  3. Collate batch + assemble_batch to model schema
  4. Forward A (NORMAL): full model forward
  5. Forward B (ORACLE): bypass head, gt_trajectory → decoder
  6. Forward C (8 trials oracle): CFM noise floor estimation
"""
import sys
from pathlib import Path
import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

_orig_load = torch.load
def _patched_load(*a, **k):
    k.setdefault("weights_only", False)
    return _orig_load(*a, **k)
torch.load = _patched_load


def main():
    from omegaconf import OmegaConf
    from otp.models.otp_soft_model import OTPSoftModel
    from otp.data.libero_loader import LIBEROOTPDataset
    from otp.train.utils import assemble_batch

    device = torch.device("cuda")
    amp_dtype = torch.bfloat16

    print("=" * 70)
    print("Step 6.0 Inference Diagnostic: H1 vs H2")
    print("=" * 70)

    base = OmegaConf.load(REPO_ROOT / "configs/otp_soft_frozen.yaml")
    override = OmegaConf.load(REPO_ROOT / "configs/otp_soft_pathB_overfit.yaml")
    cfg = OmegaConf.merge(base, override)
    num_objects = cfg.model.otp_head.num_objects
    print(f"\ndeterministic={cfg.model.otp_head.deterministic}, "
          f"use_cocos_source={cfg.model.decoder.use_cocos_source}, "
          f"num_objects={num_objects}")

    print("\nBuilding model...")
    model = OTPSoftModel(cfg.model).to(device)

    ckpt_path = REPO_ROOT / "results/pathB_step5_overfit_20260511_173519/ckpt_step0000500.pt"
    print(f"Loading ckpt: {ckpt_path.name}")
    ckpt = torch.load(ckpt_path, map_location="cpu")
    missing, unexpected = model.load_state_dict(ckpt["model_state"], strict=False)
    print(f"  Step: {ckpt['step']}")
    print(f"  Loaded {len(ckpt['model_state'])} keys; missing={len(missing)} (backbone, expected); unexpected={len(unexpected)}")
    
    # Cast model to amp_dtype (training was in bf16)
    model = model.to(amp_dtype)
    model.eval()
    print(f"  Cast model to {amp_dtype}")

    print("\nBuilding dataset (train_end_demo=5)...")
    ds = LIBEROOTPDataset(
        root=str(REPO_ROOT / cfg.data.root),
        grasp_affordance_dir=str(REPO_ROOT / cfg.data.grasp_affordance_dir),
        suite=cfg.data.suite,
        view_keys=list(cfg.data.view_keys),
        normalizer_path=str(REPO_ROOT / cfg.data.normalizer_path),
        train_end_demo=5,
    )
    print(f"  Size: {len(ds)} samples")

    np.random.seed(42)
    sample_indices = np.random.choice(len(ds), size=16, replace=False)
    print(f"  Sampled indices: {sample_indices[:5].tolist()}...")

    def collate(samples):
        batch = {}
        for k in samples[0].keys():
            vals = [s[k] for s in samples]
            if isinstance(vals[0], torch.Tensor):
                batch[k] = torch.stack(vals)
            elif isinstance(vals[0], np.ndarray):
                batch[k] = torch.stack([torch.from_numpy(v) for v in vals])
            elif isinstance(vals[0], str):
                batch[k] = vals
            else:
                batch[k] = vals
        return batch

    raw = [ds[i] for i in sample_indices]
    raw_batch = collate(raw)
    print(f"  Raw batch keys: {list(raw_batch.keys())}")

    # assemble_batch transforms to model schema
    batch_dev = assemble_batch(raw_batch, device, amp_dtype, num_objects=num_objects)
    print(f"  Assembled batch keys: {list(batch_dev.keys())}")
    print(f"  gt_trajectory: {tuple(batch_dev['gt_trajectory'].shape)}")
    print(f"  gt_action:     {tuple(batch_dev['gt_action'].shape)}")
    print(f"  proprioception:{tuple(batch_dev['proprioception'].shape)}")

    # =================================================================
    # Forward A: NORMAL
    # =================================================================
    print("\n" + "=" * 70)
    print("Forward A (NORMAL): full model — head trajectory → decoder")
    print("=" * 70)
    with torch.no_grad():
        out_A = model(batch_dev)
    A_head = out_A.get("otp_head_loss").item() if out_A.get("otp_head_loss") is not None else None
    A_dec  = out_A.get("decoder_loss").item()  if out_A.get("decoder_loss")  is not None else None
    print(f"  head_loss:    {A_head:.4f}" if A_head else "  head_loss:    None")
    print(f"  decoder_loss: {A_dec:.4f}" if A_dec else "  decoder_loss: None")

    # =================================================================
    # Forward B: ORACLE
    # =================================================================
    print("\n" + "=" * 70)
    print("Forward B (ORACLE): gt trajectory → decoder")
    print("=" * 70)

    proprio = batch_dev["proprioception"]
    affordance = batch_dev["grasp_affordance"]
    gt_traj = batch_dev["gt_trajectory"]
    gt_action = batch_dev["gt_action"]
    point_clouds = batch_dev["object_point_clouds"]

    with torch.no_grad():
        B_n, N_obj = point_clouds.shape[:2]
        pcd_flat = point_clouds.reshape(B_n * N_obj, *point_clouds.shape[2:])
        geom_flat = model.geometry_encoder(pcd_flat)
        geom = geom_flat.reshape(B_n, N_obj, -1)

        dec_out = model.decoder(
            trajectory=gt_traj,
            proprioception=proprio,
            grasp_affordance=affordance,
            object_geometry=geom,
            gt_action=gt_action,
        )
    B_dec = dec_out["loss_output"].total.item()
    print(f"  decoder_loss (oracle traj): {B_dec:.4f}")

    # =================================================================
    # Forward C: noise floor over 8 trials
    # =================================================================
    print("\n" + "=" * 70)
    print("Forward C: noise floor over 8 trials (oracle traj, fresh noise each)")
    print("=" * 70)
    losses_C = []
    with torch.no_grad():
        for trial in range(8):
            dec_out = model.decoder(
                trajectory=gt_traj,
                proprioception=proprio,
                grasp_affordance=affordance,
                object_geometry=geom,
                gt_action=gt_action,
            )
            losses_C.append(dec_out["loss_output"].total.item())
    print(f"  Trials: {[f'{l:.3f}' for l in losses_C]}")
    print(f"  Mean: {np.mean(losses_C):.4f}, Std: {np.std(losses_C):.4f}")
    print(f"  Min: {np.min(losses_C):.4f}, Max: {np.max(losses_C):.4f}")

    # =================================================================
    # Verdict
    # =================================================================
    print("\n" + "=" * 70)
    print("VERDICT")
    print("=" * 70)
    print(f"  A (normal):       decoder_loss = {A_dec:.4f}")
    print(f"  B (oracle):       decoder_loss = {B_dec:.4f}")
    print(f"  C (oracle×8):     mean = {np.mean(losses_C):.4f}, std = {np.std(losses_C):.4f}")
    print(f"  Training final:   ~1.19 (5a last 50 step)")
    
    pct_drop = (A_dec - B_dec) / A_dec * 100
    print(f"\n  A → B drop: {A_dec - B_dec:+.4f} ({pct_drop:+.1f}%)")
    
    if B_dec < A_dec * 0.5:
        print("\n→ H2 CONFIRMED: oracle traj drops decoder_loss > 50%")
        print("  Head output is the bottleneck. FIX: oracle warm-start phase.")
    elif abs(A_dec - B_dec) < 0.15 and np.mean(losses_C) > 1.0:
        print("\n→ H1 CONFIRMED: CFM noise floor dominates (A ≈ B both > 1.0)")
        print("  Even oracle input cannot drop loss. FIX: deterministic decoder.")
    else:
        print(f"\n→ MIXED / unclear. Investigate further.")


if __name__ == "__main__":
    main()
