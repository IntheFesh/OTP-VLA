"""
Probe: verify online conditioning extraction in LIBERO sim end-to-end.
Final go/no-go for Stage 5.
"""
import os
import sys
from pathlib import Path
import numpy as np
import torch
import re

os.environ.setdefault("MUJOCO_GL", "egl")

# Workaround: LIBERO stores init_states as numpy pickles which trip
# PyTorch 2.12's weights_only=True default. Patch torch.load globally.
_orig_torch_load = torch.load
def _patched_load(*a, **kw):
    kw.setdefault("weights_only", False)
    return _orig_torch_load(*a, **kw)
torch.load = _patched_load


# 1. Build the env
from libero.libero.benchmark import get_benchmark_dict
print("[1/6] Loading LIBERO benchmark ...")
benchmark = get_benchmark_dict()["libero_spatial"]()
task_id = 1
task = benchmark.get_task(task_id)
print(f"      task name: {task.name}")

from libero.libero.envs import OffScreenRenderEnv
bddl_path = benchmark.get_task_bddl_file_path(task_id)
env_args = {
    "bddl_file_name": bddl_path,
    "camera_heights": 128,
    "camera_widths": 128,
}
env = OffScreenRenderEnv(**env_args)
init_states = benchmark.get_task_init_states(task_id)
print(f"      init_states type: {type(init_states).__name__}, len: {len(init_states) if hasattr(init_states, '__len__') else 'n/a'}")
env.set_init_state(init_states[0])
obs = env.reset()
print(f"      obs keys (first 12): {sorted(obs.keys())[:12]}")

# 2. Object names (canonical for LIBERO-Spatial)
print("\n[2/6] Object names ...")
OBJECT_NAMES = [
    "akita_black_bowl_1",
    "akita_black_bowl_2",
    "cookies_1",
    "glazed_rim_porcelain_ramekin_1",
    "plate_1",
]
print(f"      {OBJECT_NAMES}")

# 3. Proprioception
print("\n[3/6] Proprioception ...")
ee_pos = obs["robot0_eef_pos"]
ee_quat = obs["robot0_eef_quat"]
gripper_qpos = obs["robot0_gripper_qpos"]
print(f"      ee_pos    : shape={ee_pos.shape}  values={ee_pos}")
print(f"      ee_quat   : shape={ee_quat.shape}  values={ee_quat}")
print(f"      gripper   : shape={gripper_qpos.shape}  values={gripper_qpos}")

from scipy.spatial.transform import Rotation as R
ee_axis_angle = R.from_quat(ee_quat[[1, 2, 3, 0]]).as_rotvec()
proprio = np.concatenate([ee_pos, ee_axis_angle, gripper_qpos[:2]]).astype(np.float32)
print(f"      proprio (8): {proprio}")

# 4. Cached affordance
print("\n[4/6] Cached affordances ...")
N_OBJ, N_POINTS, N_GRASPS = 5, 256, 8
GRASP_DIR = Path("data/grasp_affordances")
_re_strip = re.compile(r"_\d+$")

object_point_clouds = np.zeros((N_OBJ, N_POINTS, 3), dtype=np.float32)
grasp_affordance = np.zeros((N_OBJ, N_GRASPS, 7), dtype=np.float32)
grasp_affordance_mask = np.zeros((N_OBJ, N_GRASPS), dtype=bool)

for i, name in enumerate(OBJECT_NAMES):
    base_name = _re_strip.sub("", name)
    npz_path = GRASP_DIR / f"{base_name}.npz"
    if not npz_path.exists():
        print(f"      [WARN] {npz_path} missing")
        continue
    d = np.load(npz_path)
    mv, g, vm = d["mesh_vertices"], d["grasps"], d["valid_mask"]
    object_point_clouds[i, :min(N_POINTS, mv.shape[0])] = mv[:N_POINTS]
    grasp_affordance[i, :min(N_GRASPS, g.shape[0])] = g[:N_GRASPS]
    grasp_affordance_mask[i, :min(N_GRASPS, vm.shape[0])] = vm[:N_GRASPS]
    print(f"      {name:35s} -> {base_name}.npz  ({mv.shape[0]} pts, {g.shape[0]} grasps)")

# 5. Object indices
print("\n[5/6] Object indices ...")
object_indices = np.arange(N_OBJ)
print(f"      {object_indices}")

# 6. OTP-Soft forward
print("\n[6/6] OTPSoftModel forward ...")
from omegaconf import OmegaConf
from otp.models.otp_soft_model import OTPSoftModel

cfg = OmegaConf.load("configs/otp_soft_frozen.yaml")
model_cfg = OmegaConf.to_container(cfg.model, resolve=True)
print(f"      Building model (~30s) ...")
model = OTPSoftModel(model_cfg).cuda()
model.eval()

ckpt_path = "results/otp_soft_4e/ckpt_step0003000.pt"
print(f"      Loading {ckpt_path} ...")
state = torch.load(ckpt_path, map_location="cpu")
trainable_only = state.get("trainable_only", False)
missing, unexpected = model.load_state_dict(state["model_state"], strict=False)
print(f"      ckpt: trainable_only={trainable_only}  missing={len(missing)}  unexpected={len(unexpected)}")

img = obs["agentview_image"]
print(f"      agentview_image: shape={img.shape}  dtype={img.dtype}")
img_t = torch.from_numpy(img).permute(2, 0, 1).unsqueeze(0).cuda()

batch = {
    "image":               img_t,
    "instruction":         [obs.get("task_description", task.name)],
    "object_indices":      torch.from_numpy(object_indices).long().unsqueeze(0).cuda(),
    "object_point_clouds": torch.from_numpy(object_point_clouds).unsqueeze(0).cuda(),
    "proprioception":      torch.from_numpy(proprio).unsqueeze(0).cuda(),
    "grasp_affordance":    torch.from_numpy(grasp_affordance).unsqueeze(0).cuda(),
}

print("\n      Batch shapes:")
for k, v in batch.items():
    if isinstance(v, torch.Tensor):
        print(f"        {k:25s} {tuple(v.shape)}  {v.dtype}")
    else:
        print(f"        {k:25s} {v}")

print("\n      Running forward ...")
with torch.no_grad(), torch.amp.autocast(device_type="cuda", dtype=torch.bfloat16):
    out = model(batch)

if "pred_action" in out and out["pred_action"] is not None:
    pa = out["pred_action"]
    print(f"\n      pred_action shape: {tuple(pa.shape)}  dtype: {pa.dtype}")
    print(f"      finite: {torch.isfinite(pa).all().item()}")
    print(f"      mean: {pa.float().mean().item():.4f}")
    print(f"      std:  {pa.float().std().item():.4f}")
    print(f"      first action: {pa[0, 0].cpu().numpy()}")
    print("\n=== PASSED ===")
else:
    print(f"\n      [FAIL] out keys: {list(out.keys())}")
    print(f"      pred_action is None")
    sys.exit(1)

env.close()
