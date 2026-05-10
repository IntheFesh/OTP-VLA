"""
Phase 0 Test 4: Capacity test via action-supervised regression.

V6 revisions:
- M3: Target is action-level F̄ᵃ (7-dim a), NOT z-level F̄ᶻ (240-dim). Rationale: in
      z→a supervised regression, a wide z→a projection can compensate for arbitrarily
      task-agnostic z, making F̄ᶻ uninformative about backbone capacity. F̄ᵃ directly
      tests whether backbone hidden state carries task-discriminative signal that can
      reach action space.

Triggered only when Test 3 fails to push (median ≥ 5× baseline AND max ≥ 5.128).

Spec (V6 design §2.6):
- Width-doubled OTPHead (same depth, no flow matching wrapper, deterministic forward)
- Linear projection z (240-dim) → action (7-dim)
- L1 loss on demo first-step actions, 1000 steps, 2-task subset, batch=16, lr=1e-4
- Measure F̄ᵃ on the 7 action dims (20 demos per task as "samples", deterministic)

FROZEN verdict [M3]:
- F̄ᵃ ≥ 5.128:  backbone signal sufficient → Path B (deterministic head + decoder Cocos)
- F̄ᵃ ∈ [2.469, 5.128):  marginal, document rationale
- F̄ᵃ < 2.469:  backbone projection insufficient; reassess hidden state extraction

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

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from otp.models.otp_head import OTPHead
from otp.data.libero_loader import build_loader

REPO_ROOT = Path(__file__).resolve().parents[2]
OUT_DIR = REPO_ROOT / "results" / "phase0"
OUT_DIR.mkdir(parents=True, exist_ok=True)

SUBSET_TASKS = [0, 5]
N_DEMOS_PER_TASK = 500
N_TRAIN_STEPS = 1000
LR = 1e-4
BATCH = 16
N_PROBES_PER_TASK = 20
F_CRIT_005 = 2.469
F_CRIT_0001 = 5.128
ACTION_DIM = 7


class CapacityHead(nn.Module):
    """[M3] Width-doubled OTPHead body + linear z→a projection.
    Deterministic forward, no CFM wrapper. Trained with L1 on demo actions; tests
    whether backbone projection carries task-discriminative signal to action space.
    """
    def __init__(self, base_head_cfg, width_mult=2):
        super().__init__()
        cfg = dict(base_head_cfg)
        cfg["hidden_dim"] = cfg["hidden_dim"] * width_mult
        self.head_body = OTPHead.build_deterministic(cfg)
        self.z_to_action = nn.Linear(240, ACTION_DIM)

    def forward(self, c):
        z = self.head_body(c)  # (B, 5, 8, 6)
        z_flat = z.reshape(z.shape[0], -1)
        a = self.z_to_action(z_flat)  # (B, 7)
        return z, a

    def conditioning_fn(self, batch):
        return self.head_body.compute_conditioning(batch)


def train_capacity():
    import yaml
    with open(REPO_ROOT / "configs" / "otp_soft_30e.yaml") as f:
        cfg = yaml.safe_load(f)
    head = CapacityHead(cfg["head"], width_mult=2).cuda()
    optim = torch.optim.AdamW(head.parameters(), lr=LR)

    loader = build_loader(
        REPO_ROOT / "configs" / "otp_soft_30e.yaml",
        batch_size=BATCH, split="train",
        task_ids=SUBSET_TASKS,
        max_demos_per_task=N_DEMOS_PER_TASK,
    )
    it = iter(loader)
    losses = []
    for step in range(N_TRAIN_STEPS):
        try:
            batch = next(it)
        except StopIteration:
            it = iter(loader)
            batch = next(it)
        c = head.conditioning_fn(batch)
        a_gt = batch["action_gt"].cuda()  # (B, 7) first-step action
        _, a_pred = head(c)
        loss = F.l1_loss(a_pred, a_gt)
        optim.zero_grad()
        loss.backward()
        optim.step()
        if step % 100 == 0:
            losses.append({"step": step, "l1": float(loss.item())})
            print(f"  step {step}: L1 {loss.item():.4f}")
    return head, losses


def measure_action_F(head):
    """[M3] Action-level F̄ᵃ. Run head on N_PROBES_PER_TASK demos per task,
    measure ANOVA F across action dims.
    """
    K = len(SUBSET_TASKS)
    A = np.zeros((K, N_PROBES_PER_TASK, ACTION_DIM), dtype=np.float64)
    for ti, tid in enumerate(SUBSET_TASKS):
        loader = build_loader(
            REPO_ROOT / "configs" / "otp_soft_30e.yaml",
            batch_size=1, split="train", task_ids=[tid],
        )
        it = iter(loader)
        for si in range(N_PROBES_PER_TASK):
            batch = next(it)
            c = head.conditioning_fn(batch)
            with torch.no_grad():
                _, a_pred = head(c)
            A[ti, si] = a_pred[0].cpu().numpy()

    N_total = K * N_PROBES_PER_TASK
    per_dim_F = []
    per_dim_eta2 = []
    for d in range(ACTION_DIM):
        vals = A[:, :, d]
        mu_group = vals.mean(axis=1)
        mu_grand = vals.mean()
        SSB = N_PROBES_PER_TASK * ((mu_group - mu_grand) ** 2).sum()
        SSW = ((vals - mu_group.reshape(-1, 1)) ** 2).sum()
        MSB = SSB / (K - 1)
        MSW = SSW / (N_total - K)
        F_val = MSB / (MSW + 1e-12)
        eta2 = SSB / (SSB + SSW + 1e-12)
        per_dim_F.append(float(F_val))
        per_dim_eta2.append(float(eta2))

    F_mean = float(np.mean(per_dim_F))
    F_max = float(np.max(per_dim_F))
    return {
        "F_action_mean": F_mean,
        "F_action_max": F_max,
        "per_dim_F": per_dim_F,
        "per_dim_eta2": per_dim_eta2,
    }


def main():
    print("=== Phase 0 Test 4: Capacity Test (Action-Supervised, V6) ===\n")
    head, losses = train_capacity()
    metrics = measure_action_F(head)

    F_mean = metrics["F_action_mean"]
    if F_mean >= F_CRIT_0001:
        verdict = "backbone_sufficient_path_B_correct"
    elif F_mean >= F_CRIT_005:
        verdict = "backbone_marginal_document_rationale"
    else:
        verdict = "backbone_insufficient_reassess_hidden_state"

    summary = {
        "version": "V6",
        "test": "M3 action-level F̄ᵃ",
        "F_action_mean": F_mean,
        "F_action_max": metrics["F_action_max"],
        "per_dim_F": metrics["per_dim_F"],
        "per_dim_eta2": metrics["per_dim_eta2"],
        "F_critical_005": F_CRIT_005,
        "F_critical_0001": F_CRIT_0001,
        "training_log": losses,
        "verdict": verdict,
    }
    with open(OUT_DIR / "test4_capacity.json", "w") as f:
        json.dump(summary, f, indent=2)
    print(f"F̄ᵃ mean = {F_mean:.3f}")
    print(f"F̄ᵃ max  = {metrics['F_action_max']:.3f}")
    print(f"per-dim F: {[f'{f:.2f}' for f in metrics['per_dim_F']]}")
    print(f"per-dim η²: {[f'{e:.3f}' for e in metrics['per_dim_eta2']]}")
    print(f"Verdict: {verdict}")


if __name__ == "__main__":
    main()
