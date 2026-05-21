"""
Diagnostic 2 (PATCHED v2): C1 Robustness Check
===============================================

Patch: Use _build_batch (correct method name) instead of _assemble_batch.

Goal: measure r(z, visual) vs r(z, language) to rule out trivial-null hypothesis.

Constraint: BF16 inference, <10GB GPU, ~30-45 min.
"""

from __future__ import annotations

import sys
import os
import time
from pathlib import Path

sys.path.insert(0, '/root/autodl-tmp/OTP-VLA')
os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'
os.environ['HF_HOME'] = '/root/autodl-tmp/hf_cache'

import torch
_orig_torch_load = torch.load
def _patched_load(*args, **kwargs):
    kwargs.setdefault("weights_only", False)
    return _orig_torch_load(*args, **kwargs)
torch.load = _patched_load

import numpy as np


def header(s):
    print("\n" + "=" * 70)
    print(s)
    print("=" * 70)


repo_root = Path("/root/autodl-tmp/OTP-VLA")

# -----------------------------------------------------------------------
header("STEP 1: Load model")
# -----------------------------------------------------------------------
ckpt_dir = repo_root / "results" / "pathB_step6_seed42_20260513_165856"
ckpt_path = ckpt_dir / "ckpt_step0165000.pt"
config_path = repo_root / "configs" / "otp_soft_pathB_seed42.yaml"

from otp.eval.predictor import OTPSoftPredictor

predictor = OTPSoftPredictor(
    ckpt_path=ckpt_path,
    config_path=config_path,
    grasp_affordance_dir=repo_root / "data" / "grasp_affordances",
    device=torch.device("cuda"),
    log_diagnostics=False,
)
print(f"Predictor loaded. GPU mem: {torch.cuda.memory_allocated()/1e9:.2f} GB")

model = predictor.model
device = torch.device("cuda")


# -----------------------------------------------------------------------
header("STEP 2: Per-task forward pass collection")
# -----------------------------------------------------------------------
from libero.libero import benchmark
from libero.libero.envs import OffScreenRenderEnv

bm = benchmark.get_benchmark_dict()['libero_spatial']()
K_TASKS = 10
N_SAMPLES = 5

per_task_z = []
per_task_visual = []
per_task_lang = []


def extract_language_embedding(instruction: str) -> np.ndarray:
    """Mean-pooled token embeddings (same protocol as Phase 0 Gate 1)."""
    proc = predictor.model.backbone.processor
    tokens = proc.tokenizer(instruction, return_tensors="pt").input_ids.to(device)
    with torch.no_grad():
        # Try common paths to embed_tokens
        embed_layer = None
        try:
            embed_layer = model.backbone.openvla.language_model.model.embed_tokens
        except AttributeError:
            try:
                embed_layer = model.backbone.openvla.language_model.embed_tokens
            except AttributeError:
                # Walk module tree for first Embedding layer
                for name, mod in model.backbone.named_modules():
                    if isinstance(mod, torch.nn.Embedding):
                        embed_layer = mod
                        print(f"  Fallback embed_layer found at: {name}")
                        break
        if embed_layer is None:
            raise RuntimeError("Cannot find embed_tokens in model")
        token_embs = embed_layer(tokens)
        pooled = token_embs.mean(dim=1)
    return pooled.float().cpu().numpy()[0]


t_start = time.time()

for task_id in range(K_TASKS):
    task = bm.get_task(task_id)
    task_bddl = bm.get_task_bddl_file_path(task_id)
    init_states = bm.get_task_init_states(task_id)

    print(f"\n[Task {task_id}] {task.language[:60]}...")

    env = OffScreenRenderEnv(
        bddl_file_name=str(task_bddl),
        camera_heights=128, camera_widths=128,
    )

    task_z = []
    task_visual = []

    for sample_idx in range(N_SAMPLES):
        env.seed(sample_idx)
        env.reset()
        init_state = init_states[sample_idx % len(init_states)]
        env.set_init_state(init_state)
        obs, _, _, _ = env.step(np.zeros(7, dtype=np.float32))

        predictor.reset(episode_seed=task_id * 100 + sample_idx)

        # PATCH: use _build_batch (correct method name)
        try:
            batch, object_names = predictor._build_batch(obs, task.language)
        except Exception as e:
            print(f"  Sample {sample_idx} _build_batch failed: {e}")
            continue

        try:
            with torch.no_grad():
                with torch.cuda.amp.autocast(dtype=torch.bfloat16):
                    # Extract backbone visual feature
                    hidden, attn_mask = model._run_backbone(batch)
                    visual_emb = hidden.float().mean(dim=1).cpu().numpy()[0]

                    # Forward through head to get trajectory (z)
                    head_out = model.otp_head(
                        backbone_hidden=hidden,
                        backbone_attention_mask=attn_mask,
                        object_indices=batch["object_indices"],
                        gt_trajectory=None,  # inference mode
                    )
                    # Extract trajectory — try common keys
                    z_tensor = None
                    for key in ["trajectories", "trajectory", "pred_trajectory", "sample"]:
                        if isinstance(head_out, dict) and key in head_out:
                            z_tensor = head_out[key]
                            break
                    if z_tensor is None:
                        # head_out may be tensor directly
                        if torch.is_tensor(head_out):
                            z_tensor = head_out
                        else:
                            print(f"  Sample {sample_idx} head_out keys: "
                                  f"{list(head_out.keys()) if isinstance(head_out, dict) else type(head_out)}")
                            continue
                    z_flat = z_tensor.float().cpu().numpy().flatten()

            task_z.append(z_flat)
            task_visual.append(visual_emb)
        except Exception as e:
            print(f"  Sample {sample_idx} forward failed: {e}")
            import traceback
            traceback.print_exc()
            continue

    env.close()

    if not task_z:
        print(f"  ALL samples failed for task {task_id}")
        continue

    task_z_mean = np.stack(task_z).mean(axis=0)
    task_visual_mean = np.stack(task_visual).mean(axis=0)
    lang_emb = extract_language_embedding(task.language)

    per_task_z.append(task_z_mean)
    per_task_visual.append(task_visual_mean)
    per_task_lang.append(lang_emb)

    elapsed = time.time() - t_start
    print(f"  ✓ z={task_z_mean.shape} v={task_visual_mean.shape} "
          f"l={lang_emb.shape} | elapsed {elapsed:.1f}s")

if len(per_task_z) < 3:
    print(f"\nERROR: Only {len(per_task_z)} tasks succeeded. Cannot compute Mantel.")
    sys.exit(1)

Z = np.stack(per_task_z)
V = np.stack(per_task_visual)
L = np.stack(per_task_lang)
print(f"\nFinal matrices: Z {Z.shape}, V {V.shape}, L {L.shape}")


# -----------------------------------------------------------------------
header("STEP 3: Mantel correlations")
# -----------------------------------------------------------------------

def dist_matrix(X):
    n = X.shape[0]
    D = np.zeros((n, n))
    for i in range(n):
        for j in range(n):
            D[i, j] = np.linalg.norm(X[i] - X[j])
    return D


def mantel(DA, DB, n_perm=10000, seed=42):
    n = DA.shape[0]
    iu = np.triu_indices(n, k=1)
    a = DA[iu]
    b = DB[iu]
    r = np.corrcoef(a, b)[0, 1]
    rng = np.random.RandomState(seed)
    null = np.zeros(n_perm)
    for k in range(n_perm):
        perm = rng.permutation(n)
        Dperm = DB[perm][:, perm]
        bp = Dperm[iu]
        null[k] = np.corrcoef(a, bp)[0, 1]
    p = (np.abs(null) >= np.abs(r)).mean()
    return r, p


def bootstrap_ci(DA, DB, B=2000, seed=42):
    n = DA.shape[0]
    iu = np.triu_indices(n, k=1)
    rng = np.random.RandomState(seed)
    rs = np.zeros(B)
    for k in range(B):
        idx = rng.choice(n, n, replace=True)
        DAs = DA[idx][:, idx]
        DBs = DB[idx][:, idx]
        a = DAs[iu]
        b = DBs[iu]
        if np.std(a) < 1e-8 or np.std(b) < 1e-8:
            rs[k] = np.nan
        else:
            rs[k] = np.corrcoef(a, b)[0, 1]
    rs = rs[~np.isnan(rs)]
    if len(rs) < 100:
        return float('nan'), float('nan')
    lo, hi = np.percentile(rs, [2.5, 97.5])
    return lo, hi


DZ = dist_matrix(Z)
DV = dist_matrix(V)
DL = dist_matrix(L)

print("\nMantel r(Z, L) — sanity check vs published -0.19...")
r_zl, p_zl = mantel(DZ, DL, n_perm=10000)
ci_zl = bootstrap_ci(DZ, DL, B=2000)
print(f"  r(z, language) = {r_zl:+.4f}")
print(f"  95% bootstrap CI: [{ci_zl[0]:+.4f}, {ci_zl[1]:+.4f}]")
print(f"  Mantel p (10000 perms): {p_zl:.4f}")

diff = abs(r_zl - (-0.19))
print(f"  Diff from published -0.19: {diff:.4f}  "
      f"{'✓ within ±0.05' if diff < 0.05 else '⚠️  protocol drift'}")

print("\nMantel r(Z, V) — NEW measurement...")
r_zv, p_zv = mantel(DZ, DV, n_perm=10000)
ci_zv = bootstrap_ci(DZ, DV, B=2000)
print(f"  r(z, visual) = {r_zv:+.4f}")
print(f"  95% bootstrap CI: [{ci_zv[0]:+.4f}, {ci_zv[1]:+.4f}]")
print(f"  Mantel p (10000 perms): {p_zv:.4f}")


# -----------------------------------------------------------------------
header("STEP 4: VERDICT")
# -----------------------------------------------------------------------
abs_r_zl = abs(r_zl)
abs_r_zv = abs(r_zv)
print(f"\n|r(z, language)| = {abs_r_zl:.4f}")
print(f"|r(z, visual)|   = {abs_r_zv:.4f}")
print(f"Ratio |r_zv|/|r_zl| = {abs_r_zv / max(abs_r_zl, 1e-6):.2f}")

if abs_r_zv > 2 * abs_r_zl:
    verdict = "(i) z learned visual signal, not language. SUPPORTS framing."
elif abs_r_zv > abs_r_zl + 0.1:
    verdict = "(intermediate) z has moderate visual signal, marginally > language."
else:
    verdict = "(ii) z is trivially near-null. CANNOT support framing."

print(f"\n{verdict}")

# Save
out_dir = repo_root / "results" / "phase0_c1_robustness"
out_dir.mkdir(parents=True, exist_ok=True)

import json
with open(out_dir / "c1_robustness.json", "w") as f:
    json.dump({
        "r_z_lang": float(r_zl),
        "ci_z_lang": [float(ci_zl[0]), float(ci_zl[1])],
        "p_z_lang": float(p_zl),
        "r_z_visual": float(r_zv),
        "ci_z_visual": [float(ci_zv[0]), float(ci_zv[1])],
        "p_z_visual": float(p_zv),
        "n_tasks": int(K_TASKS),
        "n_samples_per_task": int(N_SAMPLES),
        "tasks_succeeded": len(per_task_z),
        "z_dim": int(Z.shape[1]),
        "visual_dim": int(V.shape[1]),
        "language_dim": int(L.shape[1]),
        "verdict": verdict,
    }, f, indent=2)

np.save(out_dir / "Z_per_task.npy", Z)
np.save(out_dir / "V_per_task.npy", V)
np.save(out_dir / "L_per_task.npy", L)
print(f"\nResults: {out_dir}")
print("=" * 70)
