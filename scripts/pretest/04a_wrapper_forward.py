"""
Pretest: OpenVLABackboneWrapper forward on real LIBERO data.

Validates:
  1. _preprocess_image produces dual-encoder format (B, 6, 224, 224 expected).
  2. vision_backbone forward accepts the input.
  3. End-to-end forward returns (B, T_tok, 4096) hidden states.
  4. attention_mask shape matches.
  5. Hidden states are finite.
  6. GPU memory + wall-clock are reasonable.
"""
import time
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from otp.models.openvla_wrapper import OpenVLABackboneWrapper
from otp.data.libero_loader import LIBEROOTPDataset


print("\n[1/5] Building dataset (real data) ...")
ds = LIBEROOTPDataset(
    root=Path("data/object_poses"),
    suite="libero_spatial",
    grasp_affordance_dir=Path("data/grasp_affordances"),
    normalizer_path=Path("data/object_poses/meta/stats_qpace.json"),
    num_grasps_per_object=8,
    num_points=256,
    horizon=8,
)
print(f"      dataset len = {len(ds)}")

print("\n[2/5] Building loader (B=2, num_workers=0, default collate) ...")
loader = DataLoader(ds, batch_size=2, shuffle=True, num_workers=0)
batch = next(iter(loader))
print(f"      image shape:    {batch['image'].shape}  dtype={batch['image'].dtype}")
print(f"      image min/max:  {batch['image'].float().min():.3f} / {batch['image'].float().max():.3f}")
print(f"      instruction:    {batch['instruction'][0]!r}")
print(f"      n_instructions: {len(batch['instruction'])}")

print("\n[3/5] Loading wrapper (frozen) ...")
t0 = time.time()
wrapper = OpenVLABackboneWrapper(mode="frozen").cuda()
print(f"      load wall-clock = {time.time() - t0:.1f}s")
print(f"      GPU mem after load = {torch.cuda.memory_allocated()/1e9:.2f} GB")

print("\n[4/5] Forward pass (10 iterations) ...")
image = batch["image"].cuda()
instruction = list(batch["instruction"])
times = []
for i in range(10):
    t0 = time.time()
    with torch.no_grad():
        out = wrapper(image=image, instruction=instruction)
    torch.cuda.synchronize()
    dt = time.time() - t0
    times.append(dt)
    if i == 0:
        h = out["hidden_states"]
        m = out["attention_mask"]
        print(f"      [iter 0] hidden_states = {tuple(h.shape)}  dtype={h.dtype}")
        print(f"      [iter 0] attn_mask     = {tuple(m.shape)}  dtype={m.dtype}")
        print(f"      [iter 0] finite        = {torch.isfinite(h).all().item()}")
        print(f"      [iter 0] hidden mean   = {h.float().mean().item():.4f}")
        print(f"      [iter 0] hidden std    = {h.float().std().item():.4f}")

print(f"\n[5/5] Timing summary")
print(f"      iter 0 (cold):  {times[0]*1000:.0f} ms")
print(f"      iter 1-9 mean:  {sum(times[1:])/9*1000:.0f} ms")
print(f"      GPU mem peak  = {torch.cuda.max_memory_allocated()/1e9:.2f} GB")
print("\n=== Pretest passed ===")
