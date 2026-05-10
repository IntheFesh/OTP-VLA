"""
Phase 0 Test 3: Cocos × Contrastive 2×2 ablation on head, disentangling
condition-source modification from task-discriminative auxiliary signal.

V6 revisions:
- S1.a: 2-task subset pre-registered as {0, 5} (cross-group pair in LIBERO-Spatial)
- S1.b: Warm-start from V3 checkpoint, NOT random init (test recovery of collapsed head)
- S1.c: F summary as (median, max, 95th-percentile) triple across 240 cells
- S2: Batch-level InfoNCE — negatives from all in-batch non-same-task samples,
      not task-level prototype (~14 negatives at batch 16, vs 1 negative at k=2 prototype)

Spec (V6 design §2.5):
- 4 conditions: Baseline / +Cocos / +Contrastive / Both
- task_ids = {0, 5}, 500 demos each, 1000 train steps, batch=16, lr=1e-4
- Cocos: x_0 ~ N(α F_φ(c), β² I), α=β=1.0, F_φ = MLP(c_dim→512→256→latent_dim)
- Contrastive: batch-level InfoNCE, τ=0.07, λ=1.0
- Post-train: collect z at 20 seeds × 2 tasks, univariate F per (obj, t, dim) cell
- Summary: (median, max, 95-pctile) across 240 cells

FROZEN Tier 1 trigger:
  any non-baseline condition satisfies BOTH
    median F̄ improvement over baseline ≥ 5×
    max F̄ ≥ 5.128

Output: results/phase0/test3_ablation/{baseline,cocos,contrastive,both}/head_finetuned.pt
        results/phase0/test3_summary.json
"""

import sys
import json
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from pathlib import Path
from dataclasses import dataclass

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from otp.models.otp_head import OTPHead
from otp.data.libero_loader import build_loader

REPO_ROOT = Path(__file__).resolve().parents[2]
OUT_DIR = REPO_ROOT / "results" / "phase0" / "test3_ablation"
OUT_DIR.mkdir(parents=True, exist_ok=True)

# FROZEN config
SUBSET_TASKS = [0, 5]
N_DEMOS_PER_TASK = 500
N_TRAIN_STEPS = 1000
LR = 1e-4
BATCH = 16
N_SEEDS_FOR_FTEST = 20
ALPHA = 1.0
BETA = 1.0
TAU = 0.07
LAMBDA_CONTRAST = 1.0
F_CRIT_0001 = 5.128
TIER1_MEDIAN_RATIO = 5.0
TIER1_MAX_ABSOLUTE = F_CRIT_0001


@dataclass
class CondConfig:
    name: str
    use_cocos: bool
    use_contrastive: bool


CONDITIONS = [
    CondConfig("baseline", False, False),
    CondConfig("cocos", True, False),
    CondConfig("contrastive", False, True),
    CondConfig("both", True, True),
]


class CocosSource(nn.Module):
    """F_φ: condition embedding c → latent source offset; x_0 ~ N(α F_φ(c), β² I)."""
    def __init__(self, c_dim, latent_dim, hidden=256):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(c_dim, 512), nn.GELU(),
            nn.Linear(512, hidden), nn.GELU(),
            nn.Linear(hidden, latent_dim),
        )

    def forward(self, c, alpha=ALPHA, beta=BETA):
        mu = alpha * self.net(c)
        noise = beta * torch.randn_like(mu)
        return mu + noise


class ContrastiveProjector(nn.Module):
    """Projects flattened z (240-dim) to language embedding dim, L2 normalized."""
    def __init__(self, z_dim, lang_dim):
        super().__init__()
        self.proj = nn.Sequential(
            nn.Linear(z_dim, 512), nn.GELU(),
            nn.Linear(512, lang_dim),
        )

    def forward(self, z_flat):
        return F.normalize(self.proj(z_flat), dim=-1)


def batch_infonce_loss(z_proj, e_per_sample, task_ids_in_batch, tau=TAU):
    """[S2] Batch-level InfoNCE.

    z_proj: (B, lang_dim), L2 normalized
    e_per_sample: (B, lang_dim), L2 normalized language embedding for the task of each sample
    task_ids_in_batch: (B,) tensor of task indices (used to identify same-task vs different-task)

    For each anchor z_i, positive is e_{t(i)}, negatives are e_{t(j)} for all j with t(j) ≠ t(i).
    """
    B = z_proj.shape[0]
    # Pairwise similarity: (B, B)
    sim = z_proj @ e_per_sample.T / tau

    # Mask: 1 where sample j has same task as anchor i (excluding self)
    # We use the e of each sample's task; the positive for anchor i is e_per_sample[i] itself.
    # Build target: anchor i's positive is index i.
    targets = torch.arange(B, device=z_proj.device)

    # But we want to exclude other same-task samples from negatives.
    # Standard treatment: only mask out self (i==j) and treat all other in-batch e as negatives,
    # since different-task samples have different e and same-task samples have same e (still
    # constitute correct positive signal). With batch-level negatives, multiple same-task
    # samples in a batch contribute multiple positives — we handle via supervised contrastive
    # (SupCon) variant: positive set = {j : t(j) = t(i), j ≠ i}.

    # SupCon formulation (Khosla et al. 2020):
    # L_i = -1/|P(i)| Σ_{p ∈ P(i)} log [ exp(sim_ip) / Σ_{a ≠ i} exp(sim_ia) ]
    task_match = task_ids_in_batch.unsqueeze(0) == task_ids_in_batch.unsqueeze(1)  # (B, B)
    eye = torch.eye(B, dtype=torch.bool, device=z_proj.device)
    pos_mask = task_match & ~eye  # positives: same task, not self
    valid_mask = ~eye  # all non-self as candidates for denominator

    # Stability: subtract max
    sim_max, _ = sim.max(dim=1, keepdim=True)
    sim_stable = sim - sim_max.detach()
    exp_sim = torch.exp(sim_stable) * valid_mask.float()
    denom = exp_sim.sum(dim=1, keepdim=True) + 1e-12

    log_prob = sim_stable - torch.log(denom)
    # Average over positive set per anchor
    n_pos = pos_mask.float().sum(dim=1)  # (B,)
    # Avoid division by zero when no positives (single-task batch fallback)
    valid_anchors = n_pos > 0
    if not valid_anchors.any():
        return torch.tensor(0.0, device=z_proj.device, requires_grad=True)

    pos_log_prob = (log_prob * pos_mask.float()).sum(dim=1) / n_pos.clamp(min=1)
    loss = -pos_log_prob[valid_anchors].mean()
    return loss


def train_one_condition(cond, base_head_state, lang_emb_per_task, lang_dim, c_dim, latent_dim):
    """Run 1000 steps of toy training under given condition; return finetuned head + aux modules.

    [S1.b] Warm-start from V3 head state.
    """
    print(f"\n--- Training condition: {cond.name} ---")
    out_path = OUT_DIR / cond.name
    out_path.mkdir(exist_ok=True)

    # Warm-start
    head = OTPHead.from_state(base_head_state).cuda()
    head.train()
    optim_params = list(head.parameters())

    cocos = None
    if cond.use_cocos:
        cocos = CocosSource(c_dim, latent_dim).cuda()
        optim_params += list(cocos.parameters())

    proj = None
    if cond.use_contrastive:
        proj = ContrastiveProjector(z_dim=240, lang_dim=lang_dim).cuda()
        optim_params += list(proj.parameters())

    optim = torch.optim.AdamW(optim_params, lr=LR)

    loader = build_loader(
        REPO_ROOT / "configs" / "otp_soft_30e.yaml",
        batch_size=BATCH, split="train",
        task_ids=SUBSET_TASKS,
        max_demos_per_task=N_DEMOS_PER_TASK,
    )
    loader_iter = iter(loader)
    log = []

    for step in range(N_TRAIN_STEPS):
        try:
            batch = next(loader_iter)
        except StopIteration:
            loader_iter = iter(loader)
            batch = next(loader_iter)

        c = head.compute_conditioning(batch)  # (B, c_dim)
        z_gt = batch["z_gt"].cuda()  # (B, 240)
        task_ids = batch["task_id"].cuda()  # (B,)

        # CFM loss
        t = torch.rand(z_gt.shape[0], device="cuda")
        if cond.use_cocos and cocos is not None:
            x0 = cocos(c).reshape(z_gt.shape)
        else:
            x0 = torch.randn_like(z_gt)
        x_t = (1 - t.view(-1, 1)) * x0 + t.view(-1, 1) * z_gt
        v_target = z_gt - x0
        v_pred = head.flow_forward(x_t, t, c)
        loss_cfm = F.mse_loss(v_pred, v_target)

        # Contrastive loss [S2]: batch-level SupCon InfoNCE
        loss_contrast = torch.tensor(0.0, device="cuda")
        if cond.use_contrastive and proj is not None:
            with torch.no_grad():
                z_sample = head.sample(c, n_steps=8)  # (B, 240)
            z_proj = proj(z_sample)
            # e_per_sample: look up each sample's task language embedding
            e_per_sample = lang_emb_per_task[task_ids.cpu()].cuda()  # (B, lang_dim)
            e_per_sample = F.normalize(e_per_sample, dim=-1)
            loss_contrast = batch_infonce_loss(z_proj, e_per_sample, task_ids, tau=TAU)

        loss = loss_cfm + LAMBDA_CONTRAST * loss_contrast
        optim.zero_grad()
        loss.backward()
        optim.step()

        if step % 100 == 0:
            log.append({
                "step": step,
                "loss_cfm": float(loss_cfm.item()),
                "loss_contrast": float(loss_contrast.item()),
            })
            print(f"  step {step}: cfm {loss_cfm.item():.4f}, contrast {loss_contrast.item():.4f}")

    torch.save({"head": head.state_dict(),
                "cocos": cocos.state_dict() if cocos else None,
                "proj": proj.state_dict() if proj else None,
                "log": log},
               out_path / "head_finetuned.pt")
    return head, cocos, log


def measure_F_summary(head, cocos, task_ids, n_seeds):
    """[S1.c] Collect z and compute (median, max, 95-pctile) of univariate F across 240 cells."""
    K = len(task_ids)
    Z = np.zeros((K, n_seeds, 5, 8, 6), dtype=np.float64)
    for ti, tid in enumerate(task_ids):
        loader = build_loader(
            REPO_ROOT / "configs" / "otp_soft_30e.yaml",
            batch_size=1, split="train", task_ids=[tid],
        )
        batch = next(iter(loader))
        c = head.compute_conditioning(batch)
        for si in range(n_seeds):
            torch.manual_seed(si)
            with torch.no_grad():
                if cocos is not None:
                    x0 = cocos(c).reshape(-1, 240)
                else:
                    x0 = torch.randn(1, 240, device="cuda")
                z = head.sample_from_x0(x0, c, n_steps=8)
            Z[ti, si] = z.cpu().numpy().reshape(5, 8, 6)

    Fs = []
    N_total = K * n_seeds
    for o in range(5):
        for t in range(8):
            for d in range(6):
                vals = Z[:, :, o, t, d]
                mu_group = vals.mean(axis=1)
                mu_grand = vals.mean()
                MSB = (n_seeds * ((mu_group - mu_grand) ** 2).sum()) / (K - 1)
                MSW = ((vals - mu_group.reshape(-1, 1)) ** 2).sum() / (N_total - K)
                Fs.append(MSB / (MSW + 1e-12))
    Fs = np.array(Fs)
    return {
        "median": float(np.median(Fs)),
        "max": float(np.max(Fs)),
        "p95": float(np.percentile(Fs, 95)),
        "all_F": Fs.tolist(),
    }


def main():
    print("=== Phase 0 Test 3: Cocos × Contrastive 2×2 Ablation (V6) ===\n")
    ckpt_path = (REPO_ROOT / ".paper_ready_ckpt").read_text().strip()
    base_state = torch.load(ckpt_path, map_location="cpu")
    base_head_state = base_state["head"]

    # Pre-compute task language embeddings (one per task in SUBSET_TASKS for the lookup)
    # We compute for ALL LIBERO-Spatial tasks to support flexible task IDs in batches.
    from otp.eval.predictor import compute_task_language_embeddings
    lang_emb_all = compute_task_language_embeddings(list(range(10)))  # (10, lang_dim)
    lang_dim = lang_emb_all.shape[-1]

    # Infer c_dim and latent_dim from the existing head
    probe_head = OTPHead.from_state(base_head_state).cuda()
    c_dim = probe_head.c_dim
    latent_dim = probe_head.latent_dim
    del probe_head
    print(f"Inferred c_dim={c_dim}, latent_dim={latent_dim}, lang_dim={lang_dim}\n")

    results = {}
    for cond in CONDITIONS:
        head, cocos, log = train_one_condition(
            cond, base_head_state, lang_emb_all, lang_dim, c_dim, latent_dim
        )
        F_summary = measure_F_summary(head, cocos, SUBSET_TASKS, N_SEEDS_FOR_FTEST)
        results[cond.name] = {
            "F_median": F_summary["median"],
            "F_max": F_summary["max"],
            "F_p95": F_summary["p95"],
            "training_log_tail": log[-3:] if log else [],
        }
        print(f"\n[{cond.name}] F̄ median={F_summary['median']:.3f}  "
              f"max={F_summary['max']:.3f}  p95={F_summary['p95']:.3f}")

    baseline_median = max(results["baseline"]["F_median"], 1e-6)
    for name in ["cocos", "contrastive", "both"]:
        results[name]["median_ratio_vs_baseline"] = results[name]["F_median"] / baseline_median

    # [S1.c] FROZEN Tier 1: median ratio ≥ 5× AND max F̄ ≥ 5.128
    triggers_tier1 = any(
        results[n]["median_ratio_vs_baseline"] >= TIER1_MEDIAN_RATIO
        and results[n]["F_max"] >= TIER1_MAX_ABSOLUTE
        for n in ["cocos", "contrastive", "both"]
    )
    results["frozen_decision"] = {
        "version": "V6",
        "tier1_thresholds": {
            "median_ratio": TIER1_MEDIAN_RATIO,
            "max_absolute": TIER1_MAX_ABSOLUTE,
        },
        "triggers_tier1": triggers_tier1,
        "next_step": "full_path_A_retrain" if triggers_tier1 else "trigger_test4",
    }

    out_path = OUT_DIR.parent / "test3_summary.json"
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nSummary written to {out_path}")
    print(f"Tier 1 triggered: {triggers_tier1}")
    print(f"Next step: {results['frozen_decision']['next_step']}")


if __name__ == "__main__":
    main()
