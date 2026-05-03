# OTP-VLA: Object-Trajectory Policy for Vision-Language-Action Models

OTP-VLA proposes a **SE(3) object-trajectory prediction** head on top of OpenVLA-OFT that decouples *where to move the object* from *how to move the robot*, enabling interpretable, data-efficient robot manipulation. The predicted object trajectory is consumed by a fixed Operational Space Controller (OSC), removing the need to learn low-level motor commands end-to-end. We benchmark against the OpenVLA-OFT action-head baseline on **LIBERO-Spatial**.

---

## Citation

```bibtex
@article{otp_vla2025,
  title   = {Object-Trajectory Policy: SE(3) Trajectory Prediction with Fixed OSC Controller for Robot Manipulation},
  author  = {TODO},
  journal = {TODO},
  year    = {2025},
}
```

---

## Hardware Requirements

| Component | Requirement |
|-----------|------------|
| GPU | NVIDIA RTX PRO 6000 (96 GB) or equivalent ≥ 40 GB VRAM |
| CPU | 22+ vCPU (tested on Intel Xeon Platinum 8470Q) |
| RAM | ≥ 110 GB |
| Storage | ≥ 500 GB (datasets + checkpoints) |
| CUDA | 12.1+ |
| OS | Ubuntu 20.04 / 22.04 |

---

## Installation

```bash
# 1. Clone this repo
git clone https://github.com/inthefesh/otp-vla.git
cd otp-vla

# 2. Run the automated setup (creates conda env, installs all deps,
#    clones third-party repos, and verifies the installation)
bash scripts/01_setup_env.sh

# 3. Activate the environment
conda activate otp_vla
```

> **Note:** `flash-attn` is installed separately inside the script with
> `--no-build-isolation` because it requires the CUDA toolkit headers at
> build time.

---

## Repository Structure

```
otp-vla/
├── configs/            # Hydra YAML configs (model, data, training, eval)
├── otp/                # Main Python package
│   ├── models/         # OTP model head + OpenVLA-OFT backbone wrappers
│   ├── controllers/    # Fixed OSC controller implementation
│   ├── data/           # LIBERO dataset loaders & augmentation
│   ├── train/          # Training loops, loss functions, schedulers
│   ├── eval/           # Evaluation harness & metrics
│   └── utils/          # Geometry (SE3), logging, visualization helpers
├── scripts/            # Shell & Python automation scripts
│   └── 01_setup_env.sh # Environment setup (← start here)
├── tests/              # Unit & integration tests (pytest)
├── notebooks/          # Exploratory Jupyter notebooks
└── third_party/        # Auto-cloned external repos (gitignored)
    ├── openvla-oft/    # OpenVLA-OFT baseline
    └── LIBERO/         # LIBERO benchmark
```

---

## Running Experiments

> Steps below are placeholders — they will be filled in as each stage is completed.

### Stage 1 — Data Preparation
```bash
# TODO: download LIBERO-Spatial dataset and convert to RLDS format
```

### Stage 2 — Baseline (OpenVLA-OFT)
```bash
# TODO: fine-tune OpenVLA-OFT with LoRA on LIBERO-Spatial
```

### Stage 3 — OTP Head Training
```bash
# TODO: train SE(3) trajectory prediction head
```

### Stage 4 — Evaluation
```bash
# TODO: evaluate both models in LIBERO simulation
```

---

## License

MIT
