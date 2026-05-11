"""
Action predictor adapters for Stage 5 evaluation.

Provides a uniform interface (ActionPredictor ABC) for both:
  - OTPSoftPredictor: wraps OTPSoftModel ckpt for sim eval
  - OpenVLAOFTPredictor: wraps the OpenVLA-OFT baseline (TODO Phase E2+)

The ABC contract is intentionally narrow:
  - reset(episode_seed): episode-scoped state reset
  - predict_chunk(obs, instruction, num_samples, reduction): main inference

Sim eval loop calls these without knowing which model is underneath, so
that paper §V comparison rows (OFT baseline vs OTP-Soft) share one rollout
implementation.

Reproducibility (per V2 design):
  predict_chunk uses counter-based per-call reseeding, with seed scheme
    forward_seed = episode_seed * 100000 + step_counter * 1000 + sample_idx
  This isolates predictor RNG from external consumption (sim env, etc.)
  so that (episode_seed, step_counter) -> output is a pure function.

Schema requirements (per F1-F14 design ledger, V3 verify):
  obs must contain SIM_KEYS_REQUIRED. Missing keys raise KeyError with
  available-keys diagnostic. This guards against LIBERO/robosuite version
  drift silently breaking sim eval.
"""

from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, Dict, List, Literal, Optional, Tuple, Union

import numpy as np
import torch

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
# Required keys in sim env obs dict, verified F1-F2 / F6 / F8 (V3 verify
# scripts/pretest/05b_sim_obs_schema.py). Missing keys -> fail-loud KeyError.
SIM_KEYS_REQUIRED: Tuple[str, ...] = (
    "agentview_image",   # F1: (128, 128, 3) uint8 HWC; sim key, hdf5 key is 'agentview_rgb'
    "robot0_eef_pos",    # F6: (3,) float64
    "robot0_eef_quat",   # F6: (4,) float64, wxyz convention
)

# Action chunk dimensions
ACTION_DIM: int = 7
DEFAULT_HORIZON: int = 8


# ---------------------------------------------------------------------------
# ABC
# ---------------------------------------------------------------------------
class ActionPredictor(ABC):
    """Abstract action predictor used by sim eval loop.

    Concrete implementations: OTPSoftPredictor, OpenVLAOFTPredictor.
    """

    @abstractmethod
    def reset(self, episode_seed: int) -> None:
        """Reset internal state at the start of each episode.

        Sets the episode_seed and zeros the step counter for counter-based
        per-call RNG reseeding (RNG isolation from external consumption).
        """
        ...

    @abstractmethod
    def predict_chunk(
        self,
        obs: dict,
        instruction: str,
        num_samples: int = 1,
        reduction: Literal["none", "mean", "median", "centroid"] = "none",
    ) -> np.ndarray:
        """Predict the next action chunk.

        Args:
            obs:        Sim env observation dict (must contain SIM_KEYS_REQUIRED).
            instruction: Task description string.
            num_samples: Number of CFM samples to draw at this condition (>=1).
                         Default 1 for vanilla single-sample inference.
            reduction:   How to reduce multiple samples to a single chunk.
                         'none' (default): use sample 0; ignores num_samples > 1
                                           reduction structure but still draws
                                           num_samples for diagnostic purposes.
                         'mean':   element-wise mean across samples.
                         'median': element-wise median across samples (robust
                                   to multimodality if histogram diagnostic
                                   shows non-unimodal distributions).
                         'centroid': NotImplementedError (TBD if needed).

        Returns:
            (H, ACTION_DIM) numpy float32 array. Default H=DEFAULT_HORIZON=8.

        Raises:
            KeyError: if obs missing any of SIM_KEYS_REQUIRED.
            NotImplementedError: if reduction='centroid'.
            ValueError: if reduction is unrecognized or num_samples < 1.
        """
        ...


# ---------------------------------------------------------------------------
# OTPSoftPredictor (Phase E1.2-E1.5: not yet implemented — stub)
# ---------------------------------------------------------------------------
class OTPSoftPredictor(ActionPredictor):
    """OTP-Soft policy adapter for sim eval.

    Wraps an OTPSoftModel ckpt; reuses production-side helpers
    (_discover_object_names, load_affordance_for_objects, assemble_batch)
    so train/eval conditioning is byte-identical (no logic duplication).
    """

    def __init__(
        self,
        ckpt_path: Path,
        config_path: Path,
        grasp_affordance_dir: Path,
        device: torch.device,
        amp_dtype: torch.dtype = torch.bfloat16,
        log_diagnostics: bool = False,
    ) -> None:
        """Build OTP-Soft predictor from a trainable_only ckpt.

        Frozen backbone weights load automatically from HF cache when
        OTPSoftModel is constructed; the trainable_only ckpt overlays
        head/decoder/geometry_encoder weights on top.

        Args:
            ckpt_path:           Path to ckpt_step*.pt (saved by train script
                                 with trainable_only=True flag).
            config_path:         Path to model config yaml (e.g. otp_soft_4e.yaml).
                                 Hydra defaults inheritance is resolved here
                                 since we're not going through @hydra.main.
            grasp_affordance_dir: Directory containing per-object .npz files
                                  (e.g. data/grasp_affordances).
            device:              torch device for model + forward.
            amp_dtype:           autocast dtype matching train cfg.train.precision.
                                 Default bf16 (matches production train config).
            log_diagnostics:     If True, log per-call DEBUG info (cache hits,
                                 obs key check, etc).

        Raises:
            FileNotFoundError: if ckpt_path or config_path or grasp_affordance_dir
                               not found.
            RuntimeError: if ckpt has unexpected keys or essential weights are
                          missing (after accounting for frozen backbone +
                          flow_matcher.model duplicate path).

        Note (cfg vs ckpt cross-check, V3 todo):
            Currently we read num_grasps/num_points/num_objects/horizon from
            cfg only. A stricter check would verify these against meta_dims
            saved in the ckpt — but train script does not yet save meta_dims.
            If the cfg used here mismatches the cfg used at training time,
            model.load_state_dict will fail with size-mismatch error (loud
            but in deep stack). Future work: save meta_dims in ckpt + verify
            here. See V3 session todo.
        """
        # ----- Resolve & validate paths ----- #
        self.ckpt_path = Path(ckpt_path)
        self.config_path = Path(config_path)
        self.grasp_affordance_dir = Path(grasp_affordance_dir)

        for p, name in [
            (self.ckpt_path, "ckpt_path"),
            (self.config_path, "config_path"),
            (self.grasp_affordance_dir, "grasp_affordance_dir"),
        ]:
            if not p.exists():
                raise FileNotFoundError(
                    f"OTPSoftPredictor: {name}={p} does not exist."
                )

        self.device = device
        self.amp_dtype = amp_dtype
        self.log_diagnostics = log_diagnostics
        self._use_bf16 = (device.type == "cuda" and amp_dtype == torch.bfloat16)

        # ----- CPU warning (Conf-2) ----- #
        if device.type != "cuda":
            import warnings
            warnings.warn(
                f"OTPSoftPredictor on device={device}. CFM forward + OpenVLA "
                f"backbone on CPU is ~100x slower than GPU. Sim eval will be "
                f"impractical. Pass torch.device('cuda') unless intentionally "
                f"CPU-debugging.",
                RuntimeWarning, stacklevel=2,
            )

        # ----- Resolve cfg (Hydra defaults inheritance) ----- #
        self.cfg = self._resolve_cfg(self.config_path)

        # ----- Auto-read dim fields from cfg (Conf-1) ----- #
        head_cfg = self.cfg.model.otp_head
        dec_cfg = self.cfg.model.decoder
        geom_cfg = self.cfg.model.geometry_encoder

        self.num_grasps: int = int(dec_cfg.num_grasps_per_object)
        self.num_points: int = int(geom_cfg.num_points)
        self.num_objects: int = int(head_cfg.num_objects)
        self.horizon: int = int(head_cfg.horizon)

        # ----- Build model + load ckpt ----- #
        self.model = self._build_and_load_model(self.cfg, self.ckpt_path)
        self.model = self.model.to(device)
        self.model.eval()

        # ----- Internal state ----- #
        # affordance cache: keyed by object_names tuple (D3 design)
        self._affordance_cache: Dict[
            Tuple[str, ...], Tuple[np.ndarray, np.ndarray, np.ndarray]
        ] = {}
        # episode RNG state (initialized via reset())
        self._episode_seed: Optional[int] = None
        self._step_counter: int = 0

        # ----- Startup snapshot ----- #
        n_trainable = sum(
            p.numel() for p in self.model.parameters() if p.requires_grad
        )
        n_total = sum(p.numel() for p in self.model.parameters())
        logger.info("=" * 60)
        logger.info("OTPSoftPredictor ready")
        logger.info(f"  ckpt:       {self.ckpt_path}")
        logger.info(f"  device:     {device}, amp_dtype={amp_dtype}, use_bf16={self._use_bf16}")
        logger.info(f"  dims:       K={self.num_grasps}, P={self.num_points}, "
                    f"N_obj={self.num_objects}, H={self.horizon}")
        logger.info(f"  params:     {n_trainable:,} trainable / {n_total:,} total")
        logger.info("=" * 60)

    # ------------------------------------------------------------------
    # Helpers (private)
    # ------------------------------------------------------------------
    @staticmethod
    def _resolve_cfg(config_path: Path):
        """Resolve Hydra 'defaults: - <base> - _self_' inheritance manually
        since we're not going through @hydra.main.

        Returns merged OmegaConf DictConfig.
        """
        from omegaconf import OmegaConf
        cfg = OmegaConf.load(config_path)
        if "defaults" not in cfg:
            return cfg

        defaults = cfg.pop("defaults")
        bases: List[str] = []
        for d in defaults:
            if isinstance(d, str) and d != "_self_":
                bases.append(d)
            # _self_ token + dict-style entries: skip silently

        merged = OmegaConf.create({})
        for base_name in bases:
            base_path = config_path.parent / f"{base_name}.yaml"
            if not base_path.exists():
                raise FileNotFoundError(
                    f"Defaults base '{base_name}' not found at {base_path}"
                )
            base_cfg = OmegaConf.load(base_path)
            # Recursive: merge base's own defaults first
            if "defaults" in base_cfg:
                base_defaults = base_cfg.pop("defaults")
                for bd in base_defaults:
                    if isinstance(bd, str) and bd != "_self_":
                        bd_path = config_path.parent / f"{bd}.yaml"
                        if bd_path.exists():
                            merged = OmegaConf.merge(merged, OmegaConf.load(bd_path))
            merged = OmegaConf.merge(merged, base_cfg)
        # Overlay current cfg (the _self_ part)
        return OmegaConf.merge(merged, cfg)

    @staticmethod
    def _build_and_load_model(cfg, ckpt_path: Path):
        """Build OTPSoftModel from cfg, load trainable_only ckpt with
        strict=False (frozen backbone + flow_matcher.model duplicate path
        are expected missing).

        Raises:
            RuntimeError: if ckpt has unexpected keys or essential
                          (non-backbone, non-duplicate-path) weights missing.
        """
        from omegaconf import OmegaConf
        from otp.models.otp_soft_model import OTPSoftModel

        model_cfg = OmegaConf.to_container(cfg.model, resolve=True)
        logger.info("Building OTPSoftModel...")
        model = OTPSoftModel(model_cfg)

        logger.info(f"Loading ckpt: {ckpt_path}")
        ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
        if not isinstance(ckpt, dict) or "model_state" not in ckpt:
            raise RuntimeError(
                f"Ckpt {ckpt_path} does not have expected schema "
                f"(dict with 'model_state' key). Got: {type(ckpt)}"
            )
        model_state = ckpt["model_state"]

        # strict=False because:
        #   1. frozen backbone weights NOT in trainable_only ckpt (loaded from HF)
        #   2. ShortcutFlowMatching internally references velocity_net as
        #      self.model -> PyTorch registers same tensor under two paths
        #      (otp_head.velocity_net.* AND otp_head.flow_matcher.model.*).
        #      Trainable_only ckpt saves only canonical path; duplicate ref
        #      auto-populates when canonical path loads (verified V3 session).
        missing, unexpected = model.load_state_dict(model_state, strict=False)

        backbone_missing = [k for k in missing if k.startswith("backbone.")]
        flow_matcher_dup = [k for k in missing if ".flow_matcher.model." in k]
        real_missing = [
            k for k in missing
            if not k.startswith("backbone.") and ".flow_matcher.model." not in k
        ]
        if real_missing:
            raise RuntimeError(
                f"Real missing keys (not backbone, not flow_matcher dup ref): "
                f"{real_missing[:5]}..."
            )
        if unexpected:
            raise RuntimeError(
                f"Unexpected keys in ckpt: {unexpected[:5]}..."
            )
        logger.info(
            f"Loaded {len(model_state)} trainable params "
            f"(skipped {len(backbone_missing)} frozen backbone, "
            f"{len(flow_matcher_dup)} flow_matcher dup refs)"
        )
        return model

    def reset(self, episode_seed: int) -> None:
        """Reset internal RNG state at start of each episode.

        Counter-based reseeding scheme:
            forward_seed = episode_seed * 100000 + step_counter * 1000 + sample_idx
        After reset(), step_counter starts at 0 and increments by 1 per
        predict_chunk call. This makes (episode_seed, step_counter, sample_idx)
        a deterministic pure function of predictor output.

        Args:
            episode_seed: integer seed for this episode; recommend < 100000
                          to avoid overflow with default 100000-step encoding.
        """
        if not isinstance(episode_seed, (int, np.integer)):
            raise TypeError(f"episode_seed must be int, got {type(episode_seed)}")
        self._episode_seed = int(episode_seed)
        self._step_counter = 0
        if self.log_diagnostics:
            logger.debug(f"reset: episode_seed={episode_seed}, step_counter=0")

    def predict_chunk(
        self,
        obs: dict,
        instruction: str,
        num_samples: int = 1,
        reduction: Literal["none", "mean", "median", "centroid"] = "none",
        *,
        _return_dict: bool = False,
    ) -> Union[np.ndarray, Dict[str, Any]]:
        """Predict next action chunk from sim obs.

        See ABC docstring for arg semantics. Implementation details:

        RNG isolation: Saves external torch / cuda / numpy RNG state on
            entry, switches to predictor's deterministic seed, runs forward,
            then restores external state. Sim env / domain randomization
            consumers see no RNG drift caused by predictor.

        Forward computation:
            1. Build batch dict from sim obs (schema validated, F1-F14)
            2. For sample_idx in range(num_samples):
                 set seed = episode_seed * 100000 + step_counter * 1000 + sample_idx
                 reseed torch/cuda/np
                 with autocast(bf16): out = model(batch)
                 collect out["pred_action"][0] (H, 7) np.float32
            3. Apply reduction
            4. Increment step_counter

        Returns:
            (H, 7) np.float32 array  if _return_dict=False (default)
            dict with all internal data if _return_dict=True (diagnostic use)
        """
        # ----- Validate args ----- #
        if self._episode_seed is None:
            raise RuntimeError(
                "predict_chunk called before reset(). "
                "Call predictor.reset(episode_seed) at start of each episode."
            )
        if num_samples < 1:
            raise ValueError(f"num_samples must be >= 1, got {num_samples}")
        if reduction not in ("none", "mean", "median", "centroid"):
            raise ValueError(
                f"Unknown reduction: {reduction!r}. "
                f"Supported: 'none', 'mean', 'median', 'centroid'."
            )
        if reduction == "centroid":
            raise NotImplementedError(
                "centroid reduction (K-means/KDE) TBD; use 'median' for "
                "multimodal-robust point estimate."
            )

        # ----- Build batch (sim path) ----- #
        batch, object_names = self._build_batch(obs, instruction)

        # ----- RNG isolation: save external state ----- #
        saved_torch_state = torch.get_rng_state()
        saved_cuda_state = (
            torch.cuda.get_rng_state_all() if self.device.type == "cuda" else None
        )
        saved_np_state = np.random.get_state()

        # ----- Sample loop ----- #
        raw_samples = np.zeros((num_samples, self.horizon, ACTION_DIM), dtype=np.float32)
        trajectory = None  # only populated if model returns it
        try:
            base_seed = self._episode_seed * 100000 + self._step_counter * 1000
            for s in range(num_samples):
                forward_seed = base_seed + s
                # Per-sample reseeding: this is what makes CFM noise reproducible
                torch.manual_seed(forward_seed)
                if self.device.type == "cuda":
                    torch.cuda.manual_seed_all(forward_seed)
                np.random.seed(forward_seed % (2**32))  # np seed must fit uint32

                with torch.no_grad():
                    with torch.autocast(
                        device_type=self.device.type,
                        dtype=self.amp_dtype,
                        enabled=self._use_bf16,
                    ):
                        out = self.model(batch)

                pred = out["pred_action"]  # (1, H, 7) bf16 (autocast output)
                # Cast to fp32 for downstream numpy reduction stability
                raw_samples[s] = pred[0].float().cpu().numpy()

                # Capture OTP head trajectory on first sample for _return_dict
                if s == 0 and _return_dict and "trajectory" in out:
                    traj = out["trajectory"]  # may be (1, N_obj, H, 6)
                    trajectory = traj[0].float().cpu().numpy()

        finally:
            # ----- Restore external RNG state (always, even on exception) ----- #
            torch.set_rng_state(saved_torch_state)
            if saved_cuda_state is not None:
                torch.cuda.set_rng_state_all(saved_cuda_state)
            np.random.set_state(saved_np_state)

        # ----- Reduction ----- #
        if reduction == "none":
            reduced = raw_samples[0]                       # (H, 7) — sample 0
        elif reduction == "mean":
            reduced = raw_samples.mean(axis=0)             # (H, 7)
        elif reduction == "median":
            reduced = np.median(raw_samples, axis=0)       # (H, 7)
        else:
            # Already validated above; should not reach
            raise ValueError(f"Unreachable reduction: {reduction}")

        # ----- Increment step counter (for next call) ----- #
        self._step_counter += 1

        if self.log_diagnostics:
            logger.debug(
                f"predict_chunk: ep_seed={self._episode_seed} "
                f"step={self._step_counter - 1} num_samples={num_samples} "
                f"reduction={reduction} pred_action.range=[{reduced.min():.3f}, {reduced.max():.3f}]"
            )

        # ----- Return ----- #
        if _return_dict:
            return {
                "pred_action":   reduced,                  # (H, 7) reduced
                "raw_samples":   raw_samples,              # (num_samples, H, 7) before reduction
                "trajectory":    trajectory,               # (N_obj, H, 6) or None
                "object_names":  object_names,             # List[str]
                "step_counter":  self._step_counter - 1,   # at-call step (post-increment)
                "episode_seed":  self._episode_seed,
            }
        return reduced

    # ------------------------------------------------------------------
    # Sim obs -> forward batch dict (E1.3)
    # ------------------------------------------------------------------
    @staticmethod
    def _check_obs_schema(obs: dict) -> None:
        """Fail-loud check that obs contains all SIM_KEYS_REQUIRED.

        Raises KeyError with available-keys diagnostic if any key missing.
        Guards against LIBERO/robosuite version drift silently breaking
        sim eval.
        """
        missing = [k for k in SIM_KEYS_REQUIRED if k not in obs]
        if missing:
            raise KeyError(
                f"Sim obs missing required keys: {missing}. "
                f"Available keys: {sorted(obs.keys())}. "
                f"This may indicate LIBERO/robosuite version drift; verify "
                f"obs schema matches scripts/pretest/05b_sim_obs_schema.py."
            )

    def _get_affordance(
        self,
        object_names: Tuple[str, ...],
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Return (affordance, mask, point_clouds) for given object set.

        Cached by object_names tuple (D3 design) — different tasks sharing
        the same 5-object set (LIBERO-Spatial does) auto-reuse the cache.

        Returns:
            affordance:      (N_obj, K, 7) float32
            affordance_mask: (N_obj, K)    bool
            point_clouds:    (N_obj, P, 3) float32
        """
        if object_names not in self._affordance_cache:
            # Cache miss: load via single-source-of-truth free function
            from otp.data.libero_loader import load_affordance_for_objects
            aff, mask, pc = load_affordance_for_objects(
                grasp_affordance_dir=self.grasp_affordance_dir,
                object_names=list(object_names),
                num_grasps=self.num_grasps,
                num_points=self.num_points,
            )
            self._affordance_cache[object_names] = (aff, mask, pc)
            if self.log_diagnostics:
                logger.debug(
                    f"affordance cache miss for {object_names}; "
                    f"cache size now {len(self._affordance_cache)}"
                )
        return self._affordance_cache[object_names]
        
    @torch.no_grad()
    def head_forward_only(
        self,
        batch: Dict[str, Any],
        seed: int,
    ) -> torch.Tensor:
        """Run backbone + OTP head sample only, returning trajectory.

        Args:
            batch: forward-ready batch dict (already collated, tensors on device).
                Must contain: image, instruction, object_indices, object_point_clouds,
                proprioception, grasp_affordance.
                gt_trajectory is NOT required (we sample, not train).
            seed: deterministic seed for CFM noise. Same seed → identical output.

        Returns:
            trajectory: (B, N_obj, H, 6) float tensor on self.device.

        Note: This bypasses the decoder entirely for Phase 0 head probes.
        RNG state is saved before and restored after to avoid drift.
        """
        # ----- RNG isolation: save external state ----- #
        saved_torch_state = torch.get_rng_state()
        saved_cuda_state = (
            torch.cuda.get_rng_state_all() if self.device.type == "cuda" else None
        )
        saved_np_state = np.random.get_state()

        try:
            # Per-call reseeding for reproducible CFM noise
            torch.manual_seed(seed)
            if self.device.type == "cuda":
                torch.cuda.manual_seed_all(seed)
            np.random.seed(seed % (2**32))

            with torch.autocast(
                device_type=self.device.type,
                dtype=self.amp_dtype,
                enabled=self._use_bf16,
            ):
                # Run backbone
                hidden, attn_mask = self.model._run_backbone(batch)
                # Sample from head only
                trajectory = self.model.otp_head.sample(
                    backbone_hidden=hidden,
                    backbone_attention_mask=attn_mask,
                    object_indices=batch["object_indices"],
                )
            # Cast to fp32 for downstream numpy stability
            trajectory = trajectory.float()

        finally:
            # Always restore RNG state
            torch.set_rng_state(saved_torch_state)
            if saved_cuda_state is not None:
                torch.cuda.set_rng_state_all(saved_cuda_state)
            np.random.set_state(saved_np_state)

        return trajectory

    def _build_batch(
        self,
        obs: dict,
        instruction: str,
    ) -> Tuple[Dict[str, Any], List[str]]:
        """Convert sim env observation to OTPSoftModel.forward batch dict (B=1).

        Schema follows train-time assemble_batch (otp/train/utils.py line 100):
            image:               (1, 3, H, W) uint8         from obs["agentview_image"]
            instruction:         List[str] len=1
            object_indices:      (1, N_obj)                 = arange(N_obj)
            object_point_clouds: (1, N_obj, P, 3) bf16      from cache
            proprioception:      (1, 8) bf16                = pos(3) + quat(4) + 0(1)
            grasp_affordance:    (1, N_obj, K, 7) bf16      from cache
            (gt_trajectory / gt_action intentionally NOT included
             so model.forward runs in inference mode.)

        Args:
            obs:         sim env step/reset output dict
            instruction: task description string

        Returns:
            batch:        dict mapped to device with appropriate dtypes
            object_names: List[str] of objects in this batch (for diagnostics
                          / _return_dict consumers; NOT a model input)
        """
        # 1. Schema validation (fail-loud)
        self._check_obs_schema(obs)

        # 2. Image: agentview_image (128, 128, 3) uint8 HWC -> (1, 3, H, W) uint8
        agentview = np.asarray(obs["agentview_image"])
        if agentview.dtype != np.uint8:
            raise ValueError(
                f"Expected obs['agentview_image'] dtype uint8, got {agentview.dtype}. "
                f"Sim setup may be misconfigured."
            )
        if agentview.ndim != 3 or agentview.shape[-1] != 3:
            raise ValueError(
                f"Expected obs['agentview_image'] shape (H, W, 3), got {agentview.shape}."
            )
        # HWC -> CHW, add batch dim
        img_chw = agentview.transpose(2, 0, 1)  # (3, H, W)
        image = torch.from_numpy(img_chw).unsqueeze(0)  # (1, 3, H, W) uint8

        # 3. Object names + affordance lookup
        from otp.data.object_pose_extractor import ObjectPoseExtractor
        object_names = ObjectPoseExtractor._discover_object_names(obs)
        if len(object_names) != self.num_objects:
            raise ValueError(
                f"Discovered {len(object_names)} objects {object_names}, "
                f"but model expects num_objects={self.num_objects}."
            )
        aff, mask, pc = self._get_affordance(tuple(object_names))

        grasp_affordance = torch.from_numpy(aff).unsqueeze(0)        # (1, N_obj, K, 7)
        object_point_clouds = torch.from_numpy(pc).unsqueeze(0)      # (1, N_obj, P, 3)

        # 4. Proprioception: pos(3) + quat(4, wxyz) + zeros(1) = 8d
        # (gripper dim structurally zero by design; F7 / paper §IV.B framing)
        ee_pos = np.asarray(obs["robot0_eef_pos"], dtype=np.float32)    # (3,)
        ee_quat = np.asarray(obs["robot0_eef_quat"], dtype=np.float32)  # (4,) wxyz (V2 verified)
        if ee_pos.shape != (3,):
            raise ValueError(f"Expected ee_pos shape (3,), got {ee_pos.shape}")
        if ee_quat.shape != (4,):
            raise ValueError(f"Expected ee_quat shape (4,), got {ee_quat.shape}")
        gripper = np.zeros(1, dtype=np.float32)
        proprio_np = np.concatenate([ee_pos, ee_quat, gripper], axis=0)  # (8,)
        proprioception = torch.from_numpy(proprio_np).unsqueeze(0)        # (1, 8)

        # 5. Object indices: arange(N_obj), batch dim = 1
        object_indices = torch.arange(self.num_objects).unsqueeze(0)  # (1, N_obj)

        # 6. Move tensors to device with appropriate dtypes
        # image stays uint8 (wrapper internally casts; F5)
        # other float tensors -> amp_dtype for backbone bf16 path
        batch = {
            "image":               image.to(self.device),
            "instruction":         [instruction],
            "object_indices":      object_indices.to(self.device),
            "object_point_clouds": object_point_clouds.to(self.device, self.amp_dtype),
            "proprioception":      proprioception.to(self.device, self.amp_dtype),
            "grasp_affordance":    grasp_affordance.to(self.device, self.amp_dtype),
        }
        return batch, object_names


# ---------------------------------------------------------------------------
# OpenVLAOFTPredictor (Phase E2+: separate baseline implementation)
# ---------------------------------------------------------------------------
class OpenVLAOFTPredictor(ActionPredictor):
    """OpenVLA-OFT baseline predictor for paper comparison rows.

    Wraps moojink/openvla-7b-oft-finetuned-libero-spatial via the HF
    OpenVLAForActionPrediction.predict_action interface.
    """

    def __init__(
        self,
        hf_id: str = "moojink/openvla-7b-oft-finetuned-libero-spatial",
        device: Optional[torch.device] = None,
    ) -> None:
        raise NotImplementedError("E2+: OFT baseline implementation deferred")

    def reset(self, episode_seed: int) -> None:
        raise NotImplementedError("E2+ pending")

    def predict_chunk(
        self,
        obs: dict,
        instruction: str,
        num_samples: int = 1,
        reduction: Literal["none", "mean", "median", "centroid"] = "none",
    ) -> np.ndarray:
        raise NotImplementedError("E2+ pending")
