"""
Pretest: One full train step (forward + backward + optimizer.step) on real
LIBERO data with frozen OpenVLA backbone.

Validates:
  1. assemble_batch + model(batch) produces a finite total_loss.
  2. loss.backward() populates .grad on trainable params (head + decoder).
  3. backbone params have NO .grad (frozen contract holds).
  4. optimizer.step() actually MUTATES trainable param values
     (snapshot pre/post and compare — strongest "is learning" evidence).
  5. Loss components are reported (head_loss, action_loss, etc) so we
     can sanity-check magnitude before a longer run.
"""
import time
from pathlib import Path

import torch
from torch.utils.data import DataLoader
from omegaconf import OmegaConf

from otp.models.otp_soft_model import OTPSoftModel
from otp.data.libero_loader import LIBEROOTPDataset
from otp.train.utils import collate_fn, assemble_batch


# ------------------------------------------------------------------ #
# 1. Config + dataset
# ------------------------------------------------------------------ #
print("\n[1/6] Loading config + dataset ...")
cfg = OmegaConf.load("configs/otp_soft_frozen.yaml")
device = torch.device("cuda")
use_bf16 = cfg.train.precision == "bf16"
amp_dtype = torch.bfloat16 if use_bf16 else torch.float32

# Dims (mirror train_otp_soft.py)
num_objects  = cfg.model.otp_head.get("num_objects", 5)
horizon      = cfg.model.otp_head.get("horizon", 8)
num_grasps   = cfg.model.decoder.get("num_grasps_per_object", 8)
num_points   = cfg.model.geometry_encoder.get("num_points", 256)
print(f"      num_objects={num_objects}  horizon={horizon}  "
      f"num_grasps={num_grasps}  num_points={num_points}")

ds = LIBEROOTPDataset(
    root=Path("data/object_poses"),
    suite="libero_spatial",
    grasp_affordance_dir=Path("data/grasp_affordances"),
    normalizer_path=Path("data/object_poses/meta/stats_qpace.json"),
    num_grasps_per_object=num_grasps,
    num_points=num_points,
    horizon=horizon,
)
loader = DataLoader(ds, batch_size=2, shuffle=True, num_workers=0,
                    collate_fn=collate_fn)
raw_batch = next(iter(loader))
print(f"      batch keys: {sorted(raw_batch.keys())}")

# ------------------------------------------------------------------ #
# 2. Build model (frozen)
# ------------------------------------------------------------------ #
print("\n[2/6] Building OTPSoftModel (frozen, ~30s) ...")
t0 = time.time()
model_cfg = OmegaConf.to_container(cfg.model, resolve=True)
model = OTPSoftModel(model_cfg).to(device)
# Mirror train script's explicit freeze (defense in depth)
for p in model.backbone.parameters():
    p.requires_grad_(False)
print(f"      build wall-clock = {time.time() - t0:.1f}s")

trainable = [p for p in model.parameters() if p.requires_grad]
n_train = sum(p.numel() for p in trainable)
print(f"      trainable params = {n_train:,}")

# ------------------------------------------------------------------ #
# 3. Snapshot trainable params (for post-step mutation check)
# ------------------------------------------------------------------ #
print("\n[3/6] Snapshotting trainable params (pre-step) ...")
# Pick a few representative params from head + decoder for a focused check
named_train = {n: p for n, p in model.named_parameters() if p.requires_grad}
sample_names = list(named_train.keys())[:3] + list(named_train.keys())[-3:]
pre_snapshot = {n: named_train[n].detach().clone() for n in sample_names}
print(f"      snapshotted {len(sample_names)} params: "
      f"head={sample_names[0]!r} ... decoder={sample_names[-1]!r}")

# ------------------------------------------------------------------ #
# 4. Forward + loss
# ------------------------------------------------------------------ #
print("\n[4/6] Forward pass ...")
batch = assemble_batch(raw_batch, device, amp_dtype, num_objects)
print(f"      assembled batch keys: {sorted(batch.keys())}")
print(f"      batch.image:  {tuple(batch['image'].shape)} {batch['image'].dtype}")

model.train()
torch.cuda.reset_peak_memory_stats()
t0 = time.time()
with torch.autocast(device_type="cuda", dtype=amp_dtype, enabled=use_bf16):
    out = model(batch)
torch.cuda.synchronize()
print(f"      forward wall-clock = {(time.time()-t0)*1000:.0f} ms")

# Inspect output
print(f"      out keys: {sorted(out.keys())}")
loss = out["total_loss"]
assert loss is not None, "FAIL: total_loss is None (NaN guard fired on first step!)"
assert torch.isfinite(loss), f"FAIL: loss = {loss.item()} not finite"
print(f"      total_loss = {loss.item():.4f}  (finite: ✓)")
for k, v in out.items():
    if k != "total_loss" and isinstance(v, torch.Tensor) and v.ndim == 0:
        print(f"        {k:30s} = {v.item():.4f}")

# ------------------------------------------------------------------ #
# 5. Backward + grad audit
# ------------------------------------------------------------------ #
print("\n[5/6] Backward + grad audit ...")
optimizer = torch.optim.AdamW(trainable, lr=1e-4)
optimizer.zero_grad(set_to_none=True)
t0 = time.time()
loss.backward()
torch.cuda.synchronize()
print(f"      backward wall-clock = {(time.time()-t0)*1000:.0f} ms")

# Trainable params: should ALL have non-None .grad
n_with_grad = sum(1 for p in trainable if p.grad is not None)
n_grad_finite = sum(1 for p in trainable
                    if p.grad is not None and torch.isfinite(p.grad).all())
print(f"      trainable params with non-None .grad: {n_with_grad}/{len(trainable)}")
print(f"      trainable params with finite .grad:   {n_grad_finite}/{len(trainable)}")
assert n_with_grad == len(trainable), \
    "FAIL: some trainable params got no grad (computation graph broken)"
assert n_grad_finite == len(trainable), \
    "FAIL: some grads non-finite (bf16 overflow? check loss components)"

# Backbone params: should ALL have None .grad (frozen contract)
backbone_params = list(model.backbone.parameters())
n_backbone_with_grad = sum(1 for p in backbone_params if p.grad is not None)
print(f"      backbone params with .grad: {n_backbone_with_grad}/{len(backbone_params)}  (expect 0)")
assert n_backbone_with_grad == 0, \
    f"FAIL: {n_backbone_with_grad} backbone params got grad — frozen contract violated"

# Grad magnitude (for sanity — should not be 0, should not be huge)
grad_norms = [p.grad.norm().item() for p in trainable if p.grad is not None]
import statistics
print(f"      grad norm stats: "
      f"min={min(grad_norms):.2e} "
      f"median={statistics.median(grad_norms):.2e} "
      f"max={max(grad_norms):.2e}")

# ------------------------------------------------------------------ #
# 6. Optimizer step + mutation check
# ------------------------------------------------------------------ #
print("\n[6/6] Optimizer step + param mutation check ...")
optimizer.step()
n_changed = 0
for n in sample_names:
    delta = (named_train[n] - pre_snapshot[n]).abs().max().item()
    changed = delta > 0
    if changed:
        n_changed += 1
    print(f"      {n[:50]:50s}  Δmax={delta:.2e}  {'✓' if changed else '✗'}")
assert n_changed == len(sample_names), \
    f"FAIL: only {n_changed}/{len(sample_names)} sampled params changed"

print(f"\n      peak GPU memory = {torch.cuda.max_memory_allocated()/1e9:.2f} GB")
print("\n=== Stage 4c passed: one train step works end-to-end ===")
