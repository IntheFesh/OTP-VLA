# OTP-VLA Repository Layout

Every file change must cite the §X.Y contract it satisfies in its commit message.
Claude Code must read this file before creating any new file or directory.

## Root Structure

```
otp-vla/
├── REPO_LAYOUT.md          ← THIS FILE — read before touching anything
├── README.md               ← User-facing project overview
├── pyproject.toml          ← Package metadata (otp_vla 0.1.0, Python >=3.10)
├── requirements.txt        ← Pinned dependencies
├── .gitignore
├── configs/                ← Hydra YAML configs (§1.2 config hash printed at startup)
├── otp/                    ← Main Python package (import as `otp.*`)
│   ├── utils/              ← Stateless math / helper utilities
│   ├── models/             ← Model architectures
│   ├── controllers/        ← Fixed OSC controller
│   ├── data/               ← Dataset loaders, normalizers
│   ├── train/              ← Training loop, loss, LR scheduler
│   └── eval/               ← Evaluation harness
├── scripts/                ← Shell + Python automation scripts
│   └── env_common.sh       ← (§7.1) Sourced by all .sh scripts; sets MUJOCO_GL etc.
├── tests/                  ← pytest test suite (100% pass required in CI)
├── notebooks/              ← Exploratory Jupyter notebooks (not imported by otp/)
├── docs/                   ← Interface contracts, calibration contract, etc.
└── third_party/            ← Auto-cloned external repos (gitignored)
```

---

## Module Contracts

### `otp/utils/lie_algebra.py`
**Status**: Stage 1  
**Purpose**: SO(3) / SE(3) Lie algebra operations for OTP trajectory parameterisation.  
**Inputs / Outputs**:

| Function | Input | Output | Notes |
|---|---|---|---|
| `so3_log(R)` | `(B,3,3)` float32 | `(B,3)` float32 | Taylor expansion for \|ω\|<1e-6 |
| `so3_exp(ω)` | `(B,3)` float32 | `(B,3,3)` float32 | Rodrigues formula |
| `se3_to_lie(T)` | `(B,4,4)` float32 | `(B,6)` float32 | ω first, t second |
| `lie_to_se3(ξ)` | `(B,6)` float32 | `(B,4,4)` float32 | inverse of above |
| `quat_to_so3(q)` | `(B,4)` wxyz float32 | `(B,3,3)` float32 | — |
| `so3_to_quat(R)` | `(B,3,3)` float32 | `(B,4)` wxyz float32 | canonical positive w |

**Callers**: `otp/models/`, `otp/data/`, `tests/test_lie_algebra.py`  
**Contract invariant**: `so3_exp(so3_log(R)) == R` and `lie_to_se3(se3_to_lie(T)) == T` to within 1e-5.

---

### `otp/utils/flow_matching.py`
**Status**: Stage 1  
**Purpose**: Conditional Flow Matching loss and sampler for OTP action head.  
**Inputs / Outputs**:

| Class / Method | Input | Output | Notes |
|---|---|---|---|
| `FlowMatching.forward(x_1, condition)` | `x_1: (B,D)`, `condition: dict` | `LossOutput` | CFM loss |
| `FlowMatching.sample(condition, num_steps, x_0)` | `condition: dict` | `(B,D)` | Euler ODE integration |
| `ShortcutFlowMatching.forward(x_1, condition)` | same + shortcut `d` | `LossOutput` | self-consistency loss |

**Callers**: `otp/models/otp_head.py` (Stage 2), `tests/test_flow_matching.py`  
**Contract invariant**: `ShortcutFlowMatching` with `d=1` produces same loss as `FlowMatching`.  
**LossOutput contract** (§2.2): all loss keys available via `to_csv_row()` — no manual key access in training loop.

---

### `otp/train/` (Stage 2+)
Entry point scripts must print `[ENV]`/`[CFG]`/`[CKPT]`/`[DATA]` snapshot on startup (§1.2).  
All loss returns must be `LossOutput` dataclass (§2.2).  
CSV written every ≤50 steps (§2.4).

---

### `otp/data/` (Stage 2+)
Must call `_resolve_view_keys()` at init time (§3.3).  
Must load normalizer from `meta/stats_qpace.json`; `raise FileNotFoundError` if missing (§5.1).

---

### `scripts/env_common.sh` (§7.1)
Must be sourced by all `scripts/*.sh`.  
Sets: `MUJOCO_GL=egl`, `PYOPENGL_PLATFORM=egl`, `PYTHONPATH`, `HF_ENDPOINT`.

---

## Change Log

| Stage | Files Added / Modified | Contract |
|---|---|---|
| 0 | `otp/**/__init__.py`, `requirements.txt`, `pyproject.toml`, `.gitignore`, `README.md`, `scripts/01_setup_env.sh` | Initial skeleton |
| 1 | `otp/utils/lie_algebra.py`, `otp/utils/flow_matching.py`, `tests/test_lie_algebra.py`, `tests/test_flow_matching.py`, `REPO_LAYOUT.md` | §3.4 action dims, §2.2 LossOutput, M1 interface contracts |
