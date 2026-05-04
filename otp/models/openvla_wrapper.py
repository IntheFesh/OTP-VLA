"""
otp/models/openvla_wrapper.py

OpenVLA backbone wrapper for OTP-Soft policy.

Loads moojink/openvla-7b-oft-finetuned-libero-spatial and exposes the same
forward signature as _TinyBackboneStub:

    forward(image, instruction) ->
        dict{'hidden_states': (B, T_tok, 4096), 'attention_mask': (B, T_tok)}

Where:
  image       : (B, 3, H, W)  uint8 *or* float32/bf16  (any resolution)
  instruction : List[str]       raw task description strings

Preprocessing pipeline (image path):
  1. If uint8 → scale to [0, 1] float
  2. Resize to IMAGE_SIZE × IMAGE_SIZE with bilinear interpolation
  3. Normalise with SigLIP mean/std via the loaded AutoProcessor

Preprocessing pipeline (text path):
  Tokenise using the model's own tokenizer with the OpenVLA-OFT prompt:
      "In: What action should the robot take to {instruction}?\nOut:"

Mode:
  frozen  — all backbone parameters frozen (default for §IV ablation)
  lora    — LoRA adapters on q_proj / v_proj, rank=16
  full    — all parameters trainable

Loading:
  Set HF_ENDPOINT=https://hf-mirror.com before calling if the default
  HuggingFace CDN is blocked.  The first call downloads ~14 GB and caches
  in $HF_HOME (default ~/.cache/huggingface).

  Example:
      export HF_ENDPOINT=https://hf-mirror.com
      export HF_HOME=/root/autodl-tmp/hf_cache
      from otp.models.openvla_wrapper import OpenVLABackboneWrapper
      bb = OpenVLABackboneWrapper(mode='frozen')

NaN guard: hidden_states are checked on every forward; ValueError is raised
  (not silently zeroed) if any element is non-finite.
"""

from __future__ import annotations

import logging
import time
from typing import List, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

logger = logging.getLogger(__name__)

# Default checkpoint — can be overridden at construction time.
_DEFAULT_CKPT = "moojink/openvla-7b-oft-finetuned-libero-spatial"

# SigLIP image normalisation constants (used as fallback if processor
# doesn't expose them).
_SIGLIP_MEAN = (0.5, 0.5, 0.5)
_SIGLIP_STD  = (0.5, 0.5, 0.5)

# Target spatial resolution fed to the vision encoder.
IMAGE_SIZE = 224


class OpenVLABackboneWrapper(nn.Module):
    """
    OpenVLA-OFT backbone for OTP-Soft.

    Args:
        checkpoint:    HuggingFace model ID or local path.
        mode:          'frozen' | 'lora' | 'full'
        lora_rank:     LoRA rank (only used when mode='lora').
        lora_targets:  Module names to apply LoRA to.
        max_new_tokens: Unused (wrapper only does forward, not generation).
    """

    def __init__(
        self,
        checkpoint: str = _DEFAULT_CKPT,
        mode: str = "frozen",
        lora_rank: int = 16,
        lora_targets: Optional[List[str]] = None,
    ) -> None:
        super().__init__()
        self.checkpoint = checkpoint
        self.mode = mode

        t0 = time.time()
        logger.info("Loading OpenVLA processor from %s …", checkpoint)
        try:
            from transformers import AutoProcessor
        except ImportError as e:
            raise ImportError(
                "transformers is required for OpenVLABackboneWrapper. "
                "pip install transformers accelerate"
            ) from e

        # AutoProcessor handles both image preprocessing and tokenisation.
        self.processor = AutoProcessor.from_pretrained(
            checkpoint,
            trust_remote_code=True,
        )

        logger.info("Loading OpenVLA model (bf16) — this may take several minutes …")
        from transformers import AutoModelForCausalLM

        self._model = AutoModelForCausalLM.from_pretrained(
            checkpoint,
            torch_dtype=torch.bfloat16,
            trust_remote_code=True,
            low_cpu_mem_usage=True,
        )
        elapsed = time.time() - t0
        n_params = sum(p.numel() for p in self._model.parameters())
        logger.info(
            "OpenVLA loaded in %.1f s  params=%s M",
            elapsed, f"{n_params/1e6:.0f}",
        )

        # Apply training mode.
        if mode == "frozen":
            for p in self._model.parameters():
                p.requires_grad_(False)
            logger.info("Backbone FROZEN (all params grad=False)")

        elif mode == "lora":
            try:
                from peft import LoraConfig, get_peft_model, TaskType
            except ImportError as e:
                raise ImportError("pip install peft") from e

            targets = lora_targets or ["q_proj", "v_proj"]
            lora_cfg = LoraConfig(
                task_type=TaskType.CAUSAL_LM,
                r=lora_rank,
                lora_alpha=lora_rank * 2,
                target_modules=targets,
                lora_dropout=0.05,
                bias="none",
            )
            self._model = get_peft_model(self._model, lora_cfg)
            self._model.print_trainable_parameters()
            logger.info("LoRA adapters attached (rank=%d, targets=%s)", lora_rank, targets)

        elif mode == "full":
            for p in self._model.parameters():
                p.requires_grad_(True)
            logger.info("Backbone FULL (all params trainable)")
        else:
            raise ValueError(f"Unknown backbone mode: {mode!r}. Choose from frozen/lora/full.")

        # Cache normalisation stats from processor if available.
        self._img_mean: Optional[torch.Tensor] = None
        self._img_std: Optional[torch.Tensor] = None
        self._try_cache_norm_stats()

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _try_cache_norm_stats(self) -> None:
        """Cache image normalisation tensors from the processor."""
        try:
            ip = self.processor.image_processor
            mean = getattr(ip, "image_mean", None) or _SIGLIP_MEAN
            std  = getattr(ip, "image_std",  None) or _SIGLIP_STD
            self._img_mean = torch.tensor(mean, dtype=torch.float32).view(1, 3, 1, 1)
            self._img_std  = torch.tensor(std,  dtype=torch.float32).view(1, 3, 1, 1)
        except Exception:
            pass  # will fall back to manual SIGLIP constants in _preprocess_image

    def _preprocess_image(self, image: torch.Tensor) -> torch.Tensor:
        """
        (B, 3, H, W) uint8-or-float → (B, 3, 224, 224) float32 normalised.

        Does not cast to bfloat16 — the model cast handles that via autocast.
        """
        img = image.float()
        if img.max() > 1.5:          # likely uint8 [0, 255]
            img = img / 255.0

        # Resize to IMAGE_SIZE.
        if img.shape[-2] != IMAGE_SIZE or img.shape[-1] != IMAGE_SIZE:
            img = F.interpolate(
                img, size=(IMAGE_SIZE, IMAGE_SIZE),
                mode="bilinear", align_corners=False,
            )

        # Normalise.
        mean = self._img_mean
        std  = self._img_std
        if mean is None:
            mean = torch.tensor(_SIGLIP_MEAN, dtype=torch.float32).view(1, 3, 1, 1)
            std  = torch.tensor(_SIGLIP_STD,  dtype=torch.float32).view(1, 3, 1, 1)
        mean = mean.to(img.device)
        std  = std.to(img.device)
        return (img - mean) / std                             # (B, 3, 224, 224)

    @staticmethod
    def _format_prompt(instruction: str) -> str:
        """OpenVLA-OFT prompt template (LIBERO fine-tune convention)."""
        inst = instruction.strip().rstrip(".").lower()
        return f"In: What action should the robot take to {inst}?\nOut:"

    def _tokenize(
        self,
        instructions: List[str],
        device: torch.device,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Tokenise instructions → (input_ids, attention_mask) on device."""
        prompts = [self._format_prompt(i) for i in instructions]
        enc = self.processor.tokenizer(
            prompts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=256,
        )
        return enc["input_ids"].to(device), enc["attention_mask"].to(device)

    # ------------------------------------------------------------------
    # Forward
    # ------------------------------------------------------------------

    def forward(
        self,
        image: torch.Tensor,
        instruction: List[str],
    ) -> dict:
        """
        Args:
            image:       (B, 3, H, W)  uint8 or float, any resolution.
            instruction: B task descriptions (raw strings).

        Returns:
            dict{
              'hidden_states':  (B, T_tok, 4096)  last-layer LM hidden states,
              'attention_mask': (B, T_tok)         1=valid, 0=padding.
            }

        The returned hidden states cover both vision tokens (prepended by the
        VLM) and text tokens.  OTPHead cross-attends to the full sequence; it
        uses learnable object-query embeddings and gathers positional hints
        via object_indices supplied by the training loop.
        """
        device = image.device

        # Preprocess image for SigLIP vision encoder.
        pixel_values = self._preprocess_image(image).to(
            self._model.dtype
        )                                                          # (B, 3, 224, 224) bf16

        # Tokenise instructions.
        input_ids, attn_mask = self._tokenize(instruction, device)

        # Forward through VLM backbone.
        grad_ctx = torch.enable_grad() if self.mode != "frozen" else torch.no_grad()
        with grad_ctx:
            out = self._model(
                pixel_values=pixel_values,
                input_ids=input_ids,
                attention_mask=attn_mask,
                output_hidden_states=True,
                return_dict=True,
            )

        # Extract last-layer hidden states.
        if hasattr(out, "hidden_states") and out.hidden_states is not None:
            hidden = out.hidden_states[-1]                         # (B, T, 4096)
        elif hasattr(out, "last_hidden_state"):
            hidden = out.last_hidden_state
        else:
            raise RuntimeError(
                "OpenVLA forward did not return hidden_states.  "
                "Ensure output_hidden_states=True is supported."
            )

        # NaN guard — fail loud, never silently zero.
        if not torch.isfinite(hidden).all():
            n_bad = (~torch.isfinite(hidden)).sum().item()
            raise ValueError(
                f"OpenVLA hidden_states contain {n_bad} non-finite values. "
                f"Check bf16 overflow or NaN-propagating inputs."
            )

        # Attention mask for full sequence (vision tokens + text tokens).
        # The model prepends vision tokens; their length = T - input_ids.shape[1].
        T = hidden.shape[1]
        n_txt = input_ids.shape[1]
        n_vis = T - n_txt
        if n_vis > 0:
            vis_mask = torch.ones(
                hidden.shape[0], n_vis,
                dtype=attn_mask.dtype, device=device,
            )
            full_mask = torch.cat([vis_mask, attn_mask], dim=1)   # (B, T)
        else:
            full_mask = attn_mask

        return {
            "hidden_states":  hidden,       # (B, T, 4096) bf16
            "attention_mask": full_mask,    # (B, T)
        }
