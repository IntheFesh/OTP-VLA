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

---

### `otp/models/otp_head.py`
**Status**: Stage 2  
**Purpose**: Cross-attention OTP head predicting (B, N_obj, H, 6) Lie algebra trajectories via Shortcut Flow Matching.

| Class / Method | Input | Output | Notes |
|---|---|---|---|
| `OTPCrossAttentionLayer.forward(queries, memory, mask)` | `queries:(B,N,D)`, `memory:(B,S,M)` | `(B,N,D)` | Pre-norm cross-attn + FFN |
| `OTPVelocityNet.forward(x_t, t, condition, d)` | `x_t:(B,D)`, `t:(B,1)`, `condition:dict`, `d:(B,1)\|None` | `(B,D)` | D=N_obj*H*6 |
| `OTPHead.forward(backbone_hidden, backbone_attention_mask, object_indices, gt_trajectory)` | `(B,S,M)`, `(B,S)`, `(B,N)`, `(B,N,H,6)\|None` | `dict` | NaN guard on backbone |
| `OTPHead.sample(backbone_hidden, backbone_attention_mask, object_indices, num_steps)` | — | `(B,N_obj,H,6)` | Shortcut or Euler |

**Callers**: training loop, eval harness  
**Contract**: `forward(gt_trajectory=None)` returns `{'trajectories': (B,N_obj,H,6)}`; `forward(gt_trajectory=...)` returns `{'loss_output': LossOutput}`.

---

### `otp/controllers/components/`
**Status**: Stage 2

| Module | Class | Key Method |
|---|---|---|
| `trajectory_parser.py` | `TrajectoryParser` | `parse(xi: (N_obj,H,6)) -> (N_obj,H,4,4)` |
| `grasp_pose_estimator.py` | `GraspPoseEstimator` | `estimate_grasp(object_pose:(4,4)) -> dict` |
| `phase_scheduler.py` | `PhaseScheduler` | `step(...) -> str` — ALL signals computed before any branching |
| `osc_target_generator.py` | `OSCTargetGenerator` | `get_target(phase, ...) -> (4,4)` — explicit if-elif per phase (M7) |
| `gripper_controller.py` | `GripperController` | `get_command(phase) -> float ∈ {-1.0, +1.0}` |
| `failure_recovery.py` | `FailureRecoveryRule` | `check(phase, step_count) -> bool` |

**PhaseScheduler invariant**: `phase_trace` entry appended with ALL `transition_signals` keys before any phase transition is evaluated. No `return` or short-circuit evaluation before the log append.

**OSCTargetGenerator invariant**: one explicit `if-elif` branch per phase (APPROACH, PRE_GRASP, GRASP, TRANSPORT, RELEASE, DONE). No unified formula.

---

### `otp/controllers/fixed_manipulation_controller.py`
**Status**: Stage 2

| Method | Input | Output |
|---|---|---|
| `set_trajectory(xi, object_pose, release_target_xy)` | `xi:(N_obj,H,6) torch.Tensor`, `object_pose:(4,4)`, `release_target_xy:(2,)` | `None` |
| `step(ee_pose, object_pose, gripper_state)` | `ee_pose:(4,4)`, `object_pose:(4,4)`, `gripper_state:float` | `action:(7,) ndarray` |
| `reset()` | — | `None` |

**Contract**: action = `[dx, dy, dz, dax, day, daz, gripper]`. Gripper is exactly ±1 (M8). NaN trajectory → zero action. `phase_trace` property proxies `PhaseScheduler.phase_trace`.

---

---

### `otp/data/object_pose_extractor.py`
**Status**: Stage 3
**Purpose**: Extract per-frame object SE(3) poses from a LIBERO demo HDF5 file.

| Method | Input | Output |
|---|---|---|
| `ObjectPoseExtractor(suite, bddl_root=None)` | — | — |
| `extract(demo_path)` | `Path` | `dict{object_names, object_poses(N,T,4,4), actions(T,7), instruction}` |

**Caller**: `scripts/03_extract_object_poses.py`. Lazy-imports `h5py`.

---

### `otp/data/libero_loader.py`
**Status**: Stage 3
**Purpose**: PyTorch-style dataset over LIBERO demo HDF5 files.

| Method | Input | Output |
|---|---|---|
| `LiberoDataset(root, suite, view_keys=None, normalizer_path=None)` | — | — |
| `__getitem__(idx)` | `int` | `dict{image, instruction, action, ee_pose, object_poses, object_names}` |

**Contracts**: §3.3 — `_resolve_view_keys()` runs at init; raises if no camera key.
§5.1 — normalizer JSON `<root>/meta/stats_qpace.json` must exist; else `FileNotFoundError`.

---

### `otp/data/perturbation_protocol.py`
**Status**: Stage 3
**Purpose**: Structured perturbation generation + validation + manifest writer.
Backs paper §V.B "controlled perturbation suite". Five types:
`instruction_rephrase`, `object_referent_swap`, `distractor_inject`,
`spatial_relation_perturb`, `visual_style`.

| Method | Input | Output |
|---|---|---|
| `gen_<type>(...)` | type-specific | candidate string / image (raises `NoPerturbationAvailable`) |
| `validate_<type>(...)` | original + candidate | `(ok: bool, reason: str)` |
| `generate_full_suite(libero_demos, output_dir)` | `list[dict]`, `Path` | `dict{manifest_path, stats}` |

**Vocab table**: `configs/perturbation_vocab.yaml` — public, version-controlled.
**Output layout**: `output_dir/manifest.json`, `output_dir/samples/<task>_<type>_<idx>.json|.npy`.

---

### `scripts/03_extract_object_poses.py` & `scripts/03b_generate_perturbation_suite.py`
**Status**: Stage 3
Both print §1.2 startup snapshot (`[ENV]`/`[CFG]`/`[CKPT]`/`[DATA]`) and exit
non-zero with a clear hint when prerequisite data is missing.

---

### `notebooks/01_motivation_figure.ipynb`
**Status**: Stage 3
Measures OpenVLA-OFT "mismatch ratio" per perturbation type using the public
checkpoint `openvla/openvla-7b-oft`. Skips early with `[SKIP]` line if
prerequisite data, CUDA, or checkpoint is missing — never silently fails.

---

## Change Log

| Stage | Files Added / Modified | Contract |
|---|---|---|
| 0 | `otp/**/__init__.py`, `requirements.txt`, `pyproject.toml`, `.gitignore`, `README.md`, `scripts/01_setup_env.sh` | Initial skeleton |
| 1 | `otp/utils/lie_algebra.py`, `otp/utils/flow_matching.py`, `tests/test_lie_algebra.py`, `tests/test_flow_matching.py`, `REPO_LAYOUT.md` | §3.4 action dims, §2.2 LossOutput, M1 interface contracts |
| 2 | `otp/models/otp_head.py`, `otp/controllers/components/` (6 files), `otp/controllers/fixed_manipulation_controller.py`, `tests/test_otp_head.py`, `tests/test_phase_scheduler.py`, `tests/test_osc_target_generator.py`, `tests/test_fixed_manipulation_controller.py`, `REPO_LAYOUT.md` | §2.2 LossOutput, M1 interface contracts, M7 no short-circuit, M8 discrete gripper |
| 3 | `otp/data/perturbation_protocol.py`, `otp/data/object_pose_extractor.py`, `otp/data/libero_loader.py`, `configs/perturbation_vocab.yaml`, `scripts/03_extract_object_poses.py`, `scripts/03b_generate_perturbation_suite.py`, `tests/test_perturbation_protocol.py`, `notebooks/01_motivation_figure.ipynb`, `requirements.txt`, `REPO_LAYOUT.md` | §V.B perturbation suite, §3.3 view-key resolution, §5.1 normalizer contract, §1.2 startup snapshot |
