"""
Verify LIBERO sim env.reset() obs schema for predictor implementation.

Critical questions answered by this probe:
  Q1. Image key — is 'agentview_rgb' the right key? What's the shape?
  Q2. EE pose — quat or axis-angle? What's the key name?
  Q3. Object pos/quat keys — do _discover_object_names() preconditions hold?
  Q4. dtype — uint8 image vs float?

Run from repo root:
  python scripts/pretest/05b_sim_obs_schema.py
"""
import os
os.environ["MUJOCO_GL"] = "egl"

import numpy as np
import torch

# Monkey-patch torch.load for LIBERO init_states pickles
_orig_load = torch.load
def _patched_load(*args, **kwargs):
    kwargs.setdefault("weights_only", False)
    return _orig_load(*args, **kwargs)
torch.load = _patched_load

from libero.libero import benchmark
from libero.libero.envs import OffScreenRenderEnv

# Pick first task in libero_spatial
benchmark_dict = benchmark.get_benchmark_dict()
task_suite = benchmark_dict["libero_spatial"]()
task = task_suite.get_task(0)
print(f"Task: {task.name}")
print(f"Language: {task.language}")
print()

env_args = {
    "bddl_file_name": os.path.join(
        task_suite.get_task_bddl_file_path(0)
    ) if hasattr(task_suite, "get_task_bddl_file_path")
       else task.bddl_file,
    "camera_heights": 128,
    "camera_widths": 128,
}

env = OffScreenRenderEnv(**env_args)
obs = env.reset()

print("=== obs dict keys ===")
print(sorted(obs.keys()))
print()

print("=== Per-key shape + dtype + sample ===")
for k in sorted(obs.keys()):
    v = obs[k]
    if hasattr(v, 'shape'):
        if v.size <= 8:
            print(f"  {k:50s} shape={v.shape} dtype={v.dtype}  value={v.tolist() if hasattr(v, 'tolist') else v}")
        else:
            print(f"  {k:50s} shape={v.shape} dtype={v.dtype}  range=[{v.min():.3f}, {v.max():.3f}]")
    else:
        print(f"  {k:50s} type={type(v).__name__} value={v}")

print()
print("=== Object name discovery test ===")
from otp.data.object_pose_extractor import ObjectPoseExtractor
names = ObjectPoseExtractor._discover_object_names(obs)
print(f"_discover_object_names returned {len(names)} names:")
for n in names:
    print(f"  {n}")

env.close()
print("\n=== DONE ===")
