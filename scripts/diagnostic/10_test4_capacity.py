"""
Phase 0 Test 4: Capacity test via action-supervised regression (V6.1).

V6 frozen items (per pre-registration §2.6):
- Width-doubled head body (no flow matching wrapper, deterministic forward)
- Linear z→action projection
- L1 loss on demo first-step actions, 1000 steps, 2-task subset {0, 5}, batch=16, lr=1e-4
- Action-level F̄ᵃ measurement (7 action dims)

V6 frozen verdict (M3):
- F̄ᵃ ≥ 5.128: backbone signal sufficient → Path B candidate
- F̄ᵃ ∈ [2.469, 5.128): marginal, document rationale
- F̄ᵃ < 2.469: backbone insufficient, reassess hidden state extraction

V6.1 implementation note:
- OTPHead's __init__ forces construction of a flow_matcher; cannot bypass cleanly via
  width_mult parameter. Instead, build a "CapacityHead" that reuses OTPHead's
  cross-attention architecture (width-doubled hidden_dim=2048) but outputs
  trajectory directly without flow matching — purely deterministic.
- Backbone is frozen (no grad), reused from V3 checkpoint via predictor.model.backbone.

Output: results/phase0/test4_capacity.json
"""

import sys
import json
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from pathlib import Path
from scipy import stats
from hydra import compose, initialize_config_dir

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

from otp.eval.predictor import OTPSoftPredictor
from otp.data.libero_loader import LIBEROOTPDataset
from otp.train.utils import collate_fn, assemble_batch
from otp.models.otp_head import OTPCrossAttentionLayer

PAPER_CKPT_FILE = REPO_ROOT / ".paper_ready_ckpt"
CONFIG_PATH = REPO_ROOT / "configs" / "otp_soft_30e.yaml"
OUT_DIR = REPO_ROOT / "results" / "phase0"
OUT_DIR.mkdir(parents=True, exist_ok=True)

# FROZEN per V6 pre-registration §2.6
PROBE_TASK_NAMES = [
    "pick_up_the_black_bowl_between_the_plate_and_the_ramekin_and_place_it_on_the_plate",  # task 0
    "pick_up_the_black_bowl_next_to_the_ramekin_and_place_it_on_the_plate",                # task 5
]
N_DEMOS_PER_TASK = 500
N_TRAIN_STEPS = 1000
LR = 1e-4
BATCH = 16
N_PROBES_PER_TASK = 20
F_CRIT_005 = 2.469
F_CRIT_0001 = 5.128
ACTION_DIM = 7
WIDTH_MULT = 2  # width-doubled per V6 §2.6


# ============================================================================
# Capacity head: deterministic cross-attention + linear z→action projection
# ============================================================================

class CapacityHead(nn.Module):
    """Width-doubled deterministic head + z→action projection.

    Architecture mirrors OTPHead's cross-attention structure but skips flow matching:
    - Object queries (N_obj learnable embeddings)
    - Project backbone hidden to capacity hidden_dim
    - K layers of cross-attention
    - Direct linear projection: queries → trajectory (N_obj, H, 6)
    - Then flatten and project trajectory → action (7,)

    Training target: L1 loss against demo first-step action.
    """

    def __init__(self, backbone_dim=4096, base_hidden_dim=1024, width_mult=WIDTH_MULT,
                 num_objects=5, horizon=8, num_heads=8, num_layers=4):
        super().__init__()
        self.num_objects = num_objects
        self.horizon = horizon
        hidden_dim = base_hidden_dim * width_mult

        # Object queries (width-doubled)
        self.obj_queries = nn.Parameter(torch.randn(1, num_objects, hidden_dim) * 0.02)
        # Backbone projection (width-doubled)
        self.backbone_proj = nn.Linear(backbone_dim, hidden_dim, bias=False)
        # Cross-attention layers (width-doubled, same depth)
        self.cross_attn_layers = nn.ModuleList([
            OTPCrossAttentionLayer(
                query_dim=hidden_dim,
                memory_dim=hidden_dim,
                num_heads=num_heads,
            )
            for _ in range(num_layers)
        ])
        # Direct trajectory projection: per-object queries → (H, 6)
        self.traj_proj = nn.Linear(hidden_dim, horizon * 6)
        # z (240-dim flattened trajectory) → action (7-dim)
        traj_dim = num_objects * horizon * 6
        self.z_to_action = nn.Linear(traj_dim, ACTION_DIM)

    def forward(self, backbone_hidden, backbone_attn_mask):
        """
        Args:
            backbone_hidden: (B, S, backbone_dim)
            backbone_attn_mask: (B, S) or None
        Returns:
            z: (B, N_obj, H, 6) trajectory
            a: (B, 7) action
        """
        B = backbone_hidden.shape[0]
        memory = self.backbone_proj(backbone_hidden)  # (B, S, hidden_dim)
        # Compute padding mask for cross-attention
        memory_key_padding_mask = None
        if backbone_attn_mask is not None:
            memory_key_padding_mask = (backbone_attn_mask == 0)
        # Object queries expanded to batch
        queries = self.obj_queries.expand(B, -1, -1)  # (B, N_obj, hidden_dim)
        # K layers of cross-attention
        for layer in self.cross_attn_layers:
            queries = layer(queries, memory, memory_key_padding_mask)
        # Project to trajectory: (B, N_obj, H*6) → (B, N_obj, H, 6)
        traj_flat = self.traj_proj(queries)  # (B, N_obj, H*6)
        z = traj_flat.reshape(B, self.num_objects, self.horizon, 6)
        # z → action
        z_flat = z.reshape(B, -1)  # (B, 240)
        a = self.z_to_action(z_flat)  # (B, 7)
        return z, a


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


def get_task_sample_indices(dataset, task_name, max_samples):
    """Return sample indices (frame=0 preferred) for one task, up to max_samples."""
    matched = []
    for i in range(len(dataset)):
        demo_idx, frame_idx = dataset._index[i]
        record = dataset.demo_records[demo_idx]
        if record["npz_path"].parent.name == task_name and frame_idx == 0:
            matched.append(i)
            if len(matched) >= max_samples:
                break
    if not matched:
        # fallback to any frame
        for i in range(len(dataset)):
            demo_idx, _ = dataset._index[i]
            record = dataset.demo_records[demo_idx]
            if record["npz_path"].parent.name == task_name:
                matched.append(i)
                if len(matched) >= max_samples:
                    break
    return matched


def build_subset_indices(dataset, task_names, n_per_task):
    """Collect sample indices for given tasks, capped at n_per_task each."""
    all_indices = []
    for name in task_names:
        ids = get_task_sample_indices(dataset, name, n_per_task)
        all_indices.extend(ids)
        print(f"  task '{name[:50]}...': {len(ids)} samples")
    return all_indices


class SubsetWrapper(torch.utils.data.Dataset):
    def __init__(self, dataset, indices):
        self.dataset = dataset
        self.indices = indices

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
        return self.dataset[self.indices[idx]]


def train_capacity_head(predictor, cfg, dataset):
    """Train width-doubled deterministic capacity head on 2-task subset."""
    print("\n--- Building capacity head ---")
    base_hidden_dim = cfg.model.otp_head.hidden_dim  # 1024
    head = CapacityHead(
        backbone_dim=cfg.model.backbone_dim,
        base_hidden_dim=base_hidden_dim,
        width_mult=WIDTH_MULT,
        num_objects=cfg.model.otp_head.num_objects,
        horizon=cfg.model.otp_head.horizon,
        num_heads=cfg.model.otp_head.num_heads,
        num_layers=cfg.model.otp_head.num_layers,
    ).to(predictor.device)
    n_params = sum(p.numel() for p in head.parameters())
    print(f"  CapacityHead params: {n_params/1e6:.1f}M "
          f"(hidden_dim {base_hidden_dim} × {WIDTH_MULT} = {base_hidden_dim * WIDTH_MULT})")

    # Freeze backbone
    backbone = predictor.model.backbone
    for p in backbone.parameters():
        p.requires_grad = False
    backbone.eval()

    print("\n--- Building subset loader ---")
    subset_indices = build_subset_indices(dataset, PROBE_TASK_NAMES, N_DEMOS_PER_TASK)
    print(f"  Total subset samples: {len(subset_indices)}")

    subset = SubsetWrapper(dataset, subset_indices)
    loader = torch.utils.data.DataLoader(
        subset, batch_size=BATCH, shuffle=True,
        collate_fn=collate_fn, num_workers=0,
        generator=torch.Generator().manual_seed(42),
    )

    print("\n--- Training capacity head (1000 steps, L1 on first-step action) ---")
    optim = torch.optim.AdamW(head.parameters(), lr=LR)
    head.train()

    losses = []
    step = 0
    loader_iter = iter(loader)
    num_objects = cfg.model.otp_head.num_objects

    while step < N_TRAIN_STEPS:
        try:
            raw_batch = next(loader_iter)
        except StopIteration:
            loader_iter = iter(loader)
            raw_batch = next(loader_iter)

        batch = assemble_batch(raw_batch, predictor.device, predictor.amp_dtype, num_objects)
        # First-step action (B, 7)
        action_gt_chunk = batch["gt_action"]  # (B, H, 7)
        a_gt = action_gt_chunk[:, 0, :].float()  # (B, 7), use first horizon step

        # Forward (backbone frozen, head trainable)
        with torch.autocast(device_type="cuda", dtype=predictor.amp_dtype, enabled=predictor._use_bf16):
            with torch.no_grad():
                hidden, attn_mask = predictor.model._run_backbone(batch)
            _, a_pred = head(hidden, attn_mask)
            a_pred = a_pred.float()

        loss = F.l1_loss(a_pred, a_gt)

        optim.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(head.parameters(), 1.0)
        optim.step()

        if step % 100 == 0:
            losses.append({"step": step, "l1": float(loss.item())})
            print(f"  step {step}: L1 = {loss.item():.4f}")
        step += 1

    print(f"  step {step-1} (final): L1 = {loss.item():.4f}")
    return head, losses


def measure_action_F_for_capacity(predictor, head, dataset, cfg):
    """Run trained head on N_PROBES_PER_TASK demos per task, compute action-level F̄ᵃ.

    Per V6 §2.6 M3: F-test over 7 action dims using K=2 tasks × N=20 demos.
    """
    head.eval()
    K = len(PROBE_TASK_NAMES)
    A = np.zeros((K, N_PROBES_PER_TASK, ACTION_DIM), dtype=np.float64)
    num_objects = cfg.model.otp_head.num_objects

    for ti, name in enumerate(PROBE_TASK_NAMES):
        sample_indices = get_task_sample_indices(dataset, name, N_PROBES_PER_TASK)
        if len(sample_indices) < N_PROBES_PER_TASK:
            print(f"  WARNING: task {ti} has only {len(sample_indices)} samples")
        for si, idx in enumerate(sample_indices[:N_PROBES_PER_TASK]):
            sample = dataset[idx]
            raw_batch = collate_fn([sample])
            batch = assemble_batch(raw_batch, predictor.device, predictor.amp_dtype, num_objects)
            with torch.autocast(device_type="cuda", dtype=predictor.amp_dtype, enabled=predictor._use_bf16):
                with torch.no_grad():
                    hidden, attn_mask = predictor.model._run_backbone(batch)
                    _, a_pred = head(hidden, attn_mask)
            A[ti, si] = a_pred[0].float().cpu().numpy()

    # F-test per action dim
    K = A.shape[0]
    N = A.shape[1]
    N_total = K * N
    per_dim_F = []
    per_dim_pval = []
    per_dim_eta2 = []
    per_dim_means = []
    for d in range(ACTION_DIM):
        vals = A[:, :, d]
        mu_group = vals.mean(axis=1)
        mu_grand = vals.mean()
        SSB = N * ((mu_group - mu_grand) ** 2).sum()
        SSW = ((vals - mu_group.reshape(-1, 1)) ** 2).sum()
        MSB = SSB / (K - 1)
        MSW = SSW / (N_total - K)
        F_val = MSB / (MSW + 1e-12)
        pval = 1.0 - stats.f.cdf(F_val, K - 1, N_total - K)
        eta2 = SSB / (SSB + SSW + 1e-12)
        per_dim_F.append(float(F_val))
        per_dim_pval.append(float(pval))
        per_dim_eta2.append(float(eta2))
        per_dim_means.append([float(m) for m in mu_group])

    F_mean = float(np.mean(per_dim_F))
    F_max = float(np.max(per_dim_F))
    return {
        "F_action_mean": F_mean,
        "F_action_max": F_max,
        "per_dim_F": per_dim_F,
        "per_dim_pval": per_dim_pval,
        "per_dim_eta2": per_dim_eta2,
        "per_dim_task_means": per_dim_means,
    }


def main():
    print("=" * 70)
    print("Phase 0 Test 4: Capacity Test (Action-Supervised, V6.1)")
    print("=" * 70)
    ckpt = PAPER_CKPT_FILE.read_text().strip()
    print(f"Checkpoint: {ckpt}\n")

    cfg = load_config()
    dataset = build_dataset(cfg)
    print(f"Dataset: {len(dataset)} samples\n")

    predictor = OTPSoftPredictor(
        ckpt_path=ckpt,
        config_path=str(CONFIG_PATH),
        grasp_affordance_dir=str(Path(cfg.data.grasp_affordance_dir)),
        device=torch.device("cuda"),
        log_diagnostics=False,
    )
    predictor.reset(episode_seed=0)
    print(f"Predictor: device={predictor.device}, amp_dtype={predictor.amp_dtype}\n")

    head, losses = train_capacity_head(predictor, cfg, dataset)

    print("\n--- Measuring action-level F̄ᵃ ---")
    metrics = measure_action_F_for_capacity(predictor, head, dataset, cfg)

    print(f"\n  Per-dim F (7 action dims): {[f'{f:.2f}' for f in metrics['per_dim_F']]}")
    print(f"  Per-dim p:                 {[f'{p:.3e}' for p in metrics['per_dim_pval']]}")
    print(f"  Per-dim η²:                {[f'{e:.3f}' for e in metrics['per_dim_eta2']]}")
    print(f"\n  F̄ᵃ mean = {metrics['F_action_mean']:.3f}")
    print(f"  F̄ᵃ max  = {metrics['F_action_max']:.3f}")

    # V6 §2.6 M3 frozen verdict
    F_mean = metrics["F_action_mean"]
    if F_mean >= F_CRIT_0001:
        verdict = "backbone_sufficient_path_B_correct"
        decision = "Path B is the correct route. Proceed to deterministic head + decoder Cocos retrain."
    elif F_mean >= F_CRIT_005:
        verdict = "backbone_marginal_document_rationale"
        decision = ("Marginal. Document rationale. Path B is risky but viable; "
                    "consider width scaling or Path A as alternative.")
    else:
        verdict = "backbone_insufficient_reassess_hidden_state"
        decision = ("Backbone projection signal insufficient. Reassess hidden state extraction "
                    "(layer choice, task-token attention pooling) BEFORE retrain. "
                    "Consider Path α (workshop) if reassessment fails.")

    summary = {
        "version": "V6.1",
        "test": "M3 action-level F̄ᵃ via width-doubled deterministic head",
        "task_subset": PROBE_TASK_NAMES,
        "n_demos_per_task": N_DEMOS_PER_TASK,
        "n_train_steps": N_TRAIN_STEPS,
        "n_probes_per_task": N_PROBES_PER_TASK,
        "width_mult": WIDTH_MULT,
        "F_action_mean": F_mean,
        "F_action_max": metrics["F_action_max"],
        "per_dim_F": metrics["per_dim_F"],
        "per_dim_pval": metrics["per_dim_pval"],
        "per_dim_eta2": metrics["per_dim_eta2"],
        "per_dim_task_means": metrics["per_dim_task_means"],
        "F_critical_005": F_CRIT_005,
        "F_critical_0001": F_CRIT_0001,
        "training_log": losses,
        "final_train_loss": losses[-1] if losses else None,
        "verdict": verdict,
        "decision_implication": decision,
    }
    with open(OUT_DIR / "test4_capacity.json", "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\nResults written to {OUT_DIR / 'test4_capacity.json'}")
    print(f"\nVerdict: {verdict}")
    print(f"Decision: {decision}")


if __name__ == "__main__":
    main()
