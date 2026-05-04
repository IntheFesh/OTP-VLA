"""
otp/models/openvla_wrapper.py

OpenVLA-OFT backbone wrapper for OTP-Soft.

Bypasses the OFT outer forward (which bakes in action-mask logic and parallel
decoding) and instead composes the *internal* modules directly:

    image  →  vision_backbone  →  projector  →  visual_embeds   ┐
                                                                 ├→ cat → llm_backbone
    instruction → tokenizer → embed_tokens → text_embeds        ┘  (inputs_embeds=)

The OFT action decoder, action queries, and parallel-decoding plumbing in the
checkpoint are dropped because they are wholly replaced by our OTP head and
LanguageAgnosticDecoder.

Forward signature (matches `_TinyBackboneStub` for drop-in replacement):
    forward(image, instruction) ->
        dict{'hidden_states': (B, T_tok, D_h),
             'attention_mask': (B, T_tok)}

Where T_tok = N_vision_tokens (~256 for 224×224 fused SigLIP+DINOv2)
            + N_text_tokens (padded prompt length).

Image preprocessing (internal — wrapper does NOT use HF processor for images,
which expects PIL):
    1. uint8 → float / 255 (or pass-through if already in [0, 1])
    2. resize → 224×224 bilinear
    3. ImageNet normalize  mean=[.485,.456,.406]  std=[.229,.224,.225]
    4. cast to model dtype (bf16)

Mode handling:
    frozen — vision_backbone + projector + llm_backbone all requires_grad=False
    lora   — vision_backbone + projector frozen; LoRA on llm_backbone q/v/k/o
    full   — everything trainable

NaN guard:
    Hidden states checked on every forward; ValueError raised on non-finite.

Loading / env:
    HF_ENDPOINT  — mirror URL (e.g. https://hf-mirror.com on AutoDL)
    HF_HOME      — local cache dir
"""

from __future__ import annotations

import logging
import time
from typing import List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

logger = logging.getLogger(__name__)

_DEFAULT_CKPT = "moojink/openvla-7b-oft-finetuned-libero-spatial"

IMAGE_SIZE = 224
_IMAGENET_MEAN = (0.485, 0.456, 0.406)
_IMAGENET_STD  = (0.229, 0.224, 0.225)


class OpenVLABackboneWrapper(nn.Module):
    """
    Wraps the vision_backbone, projector, and llm_backbone modules of an
    OpenVLA-OFT checkpoint, bypassing the OFT outer forward.

    Args:
        checkpoint:  HF model ID or local path.
        mode:        'frozen' | 'lora' | 'full'
        lora_rank:   LoRA rank (mode='lora' only).
        lora_alpha:  LoRA alpha.
    """

    # Candidate (vision, projector, llm) attribute name triples.  Probed in
    # order; the first triple all-present on the loaded model wins.  If none
    # match, we raise with the actual top-level children listed so the user
    # can update this list (see scripts/inspect_openvla_modules.py).
    _MODULE_CANDIDATES: Tuple[Tuple[str, str, str], ...] = (
        ("vision_backbone",  "projector",              "llm_backbone"),
        ("vision_tower",     "multi_modal_projector",  "language_model"),
        ("vision_model",     "projector",              "language_model"),
    )

    def __init__(
        self,
        checkpoint: str = _DEFAULT_CKPT,
        mode: str = "frozen",
        lora_rank: int = 32,
        lora_alpha: int = 16,
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
                "pip install transformers accelerate peft"
            ) from e

        self.processor = AutoProcessor.from_pretrained(
            checkpoint, trust_remote_code=True,
        )

        full = self._load_full_model(checkpoint)
        elapsed = time.time() - t0
        n_params = sum(p.numel() for p in full.parameters())
        logger.info("OpenVLA loaded in %.1fs  params=%.0fM",
                    elapsed, n_params / 1e6)

        # Extract the three modules we keep; discard everything else.
        vision, projector, llm = self._extract_modules(full)
        self.vision_backbone = vision
        self.projector = projector
        self.llm_backbone = llm

        # Drop the parent reference so the OFT action head is freed.
        del full
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        # Apply training mode.
        if mode == "frozen":
            self._freeze(self.vision_backbone)
            self._freeze(self.projector)
            self._freeze(self.llm_backbone)
            logger.info("Backbone FROZEN (vision + projector + llm all grad=False)")
        elif mode == "lora":
            self._freeze(self.vision_backbone)
            self._freeze(self.projector)
            try:
                from peft import LoraConfig, get_peft_model, TaskType
            except ImportError as e:
                raise ImportError("pip install peft") from e
            lora_cfg = LoraConfig(
                task_type=TaskType.CAUSAL_LM,
                r=lora_rank,
                lora_alpha=lora_alpha,
                target_modules=["q_proj", "v_proj", "k_proj", "o_proj"],
                lora_dropout=0.05,
                bias="none",
            )
            self.llm_backbone = get_peft_model(self.llm_backbone, lora_cfg)
            self.llm_backbone.print_trainable_parameters()
            logger.info("LoRA on llm_backbone  rank=%d alpha=%d  vision frozen",
                        lora_rank, lora_alpha)
        elif mode == "full":
            for p in self.parameters():
                p.requires_grad_(True)
            logger.info("Backbone FULL (all params trainable)")
        else:
            raise ValueError(f"Unknown backbone mode: {mode!r}")

        self.hidden_dim = self._infer_hidden_dim()
        logger.info("hidden_dim=%d", self.hidden_dim)

    # ------------------------------------------------------------------
    # Loading / module extraction
    # ------------------------------------------------------------------

    @staticmethod
    def _load_full_model(checkpoint: str):
        """Try AutoModelForVision2Seq, fall back to AutoModelForCausalLM."""
        try:
            from transformers import AutoModelForVision2Seq
            return AutoModelForVision2Seq.from_pretrained(
                checkpoint, trust_remote_code=True,
                torch_dtype=torch.bfloat16, low_cpu_mem_usage=True,
            )
        except (ImportError, AttributeError):
            from transformers import AutoModelForCausalLM
            return AutoModelForCausalLM.from_pretrained(
                checkpoint, trust_remote_code=True,
                torch_dtype=torch.bfloat16, low_cpu_mem_usage=True,
            )

    @classmethod
    def _extract_modules(cls, model) -> Tuple[nn.Module, nn.Module, nn.Module]:
        """Locate (vision_backbone, projector, llm_backbone) by attribute name."""
        for vname, pname, lname in cls._MODULE_CANDIDATES:
            if all(hasattr(model, n) for n in (vname, pname, lname)):
                logger.info("Module triple matched: (%s, %s, %s)", vname, pname, lname)
                return getattr(model, vname), getattr(model, pname), getattr(model, lname)
        children = [n for n, _ in model.named_children()]
        raise RuntimeError(
            f"Could not locate (vision, projector, llm) modules on "
            f"{type(model).__name__}. Top-level children: {children}. "
            f"Run scripts/inspect_openvla_modules.py and update "
            f"OpenVLABackboneWrapper._MODULE_CANDIDATES."
        )

    @staticmethod
    def _freeze(module: nn.Module) -> None:
        for p in module.parameters():
            p.requires_grad_(False)

    def _infer_hidden_dim(self) -> int:
        """Best-effort hidden_size from llm_backbone.config (handles peft wrapper)."""
        m = self.llm_backbone
        for path in ("config", "base_model.config", "base_model.model.config"):
            obj = m
            try:
                for part in path.split("."):
                    obj = getattr(obj, part)
                if hasattr(obj, "hidden_size"):
                    return int(obj.hidden_size)
            except AttributeError:
                continue
        return 4096  # LLaMA-7B default

    # ------------------------------------------------------------------
    # Preprocessing
    # ------------------------------------------------------------------

    def _model_dtype(self) -> torch.dtype:
        """Dtype of the underlying weights (bf16 in production)."""
        for p in self.llm_backbone.parameters():
            return p.dtype
        return torch.bfloat16

    def _preprocess_image(self, image: torch.Tensor) -> torch.Tensor:
        """
        (B, 3, H, W) uint8-or-float → (B, 3, 224, 224) bf16, ImageNet-normalised.
        """
        img = image.float()
        if img.max() > 1.5:
            img = img / 255.0
        if img.shape[-2] != IMAGE_SIZE or img.shape[-1] != IMAGE_SIZE:
            img = F.interpolate(
                img, size=(IMAGE_SIZE, IMAGE_SIZE),
                mode="bilinear", align_corners=False,
            )
        mean = torch.tensor(_IMAGENET_MEAN, device=img.device).view(1, 3, 1, 1)
        std  = torch.tensor(_IMAGENET_STD,  device=img.device).view(1, 3, 1, 1)
        img = (img - mean) / std
        return img.to(self._model_dtype())

    @staticmethod
    def _format_prompt(instruction: str) -> str:
        """OpenVLA-OFT prompt template (LIBERO fine-tune convention)."""
        inst = instruction.strip().rstrip(".").lower()
        return f"In: What action should the robot take to {inst}?\nOut:"

    def _tokenize(
        self,
        instructions: List[str],
        device: torch.device,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        prompts = [self._format_prompt(i) for i in instructions]
        enc = self.processor.tokenizer(
            prompts, return_tensors="pt",
            padding=True, truncation=True, max_length=64,
        )
        return enc["input_ids"].to(device), enc["attention_mask"].to(device)

    # ------------------------------------------------------------------
    # Forward — manual composition (NOT OFT outer forward)
    # ------------------------------------------------------------------

    def _embed_text(self, input_ids: torch.Tensor) -> torch.Tensor:
        """get_input_embeddings() works for both raw LLM and peft-wrapped LLM."""
        return self.llm_backbone.get_input_embeddings()(input_ids)

    def _run_vision(self, pixel_values: torch.Tensor) -> torch.Tensor:
        """
        Vision encoding → (B, N_vis, D_vision).

        OpenVLA's PrismaticVisionBackbone forward returns a Tensor directly.
        Some HF wrappers return a ModelOutput; we unwrap defensively.
        """
        out = self.vision_backbone(pixel_values)
        if isinstance(out, torch.Tensor):
            return out
        for attr in ("last_hidden_state", "pooler_output"):
            if hasattr(out, attr):
                v = getattr(out, attr)
                if v is not None:
                    return v
        if isinstance(out, (tuple, list)):
            return out[0]
        raise RuntimeError(
            f"Vision backbone returned unexpected type: {type(out).__name__}"
        )

    def forward(
        self,
        image: torch.Tensor,
        instruction: List[str],
        **kwargs,
    ) -> dict:
        """
        Args:
            image:       (B, 3, H, W) uint8 or float.
            instruction: List[str] of length B.

        Returns:
            dict{
              'hidden_states':  (B, T_tok, hidden_dim),
              'attention_mask': (B, T_tok),
            }
        """
        device = image.device
        B = image.shape[0]

        # ---- 1. Vision encoding ---- #
        pixel_values = self._preprocess_image(image)
        grad_vision = self.mode == "full"
        vision_ctx = torch.enable_grad() if grad_vision else torch.no_grad()
        with vision_ctx:
            vision_features = self._run_vision(pixel_values)
            visual_embeds = self.projector(vision_features)         # (B, N_vis, D_h)

        n_vis = visual_embeds.shape[1]

        # ---- 2. Text tokenisation + embedding ---- #
        input_ids, attn_mask = self._tokenize(instruction, device)
        text_embeds = self._embed_text(input_ids)                    # (B, S, D_h)

        # ---- 3. Concatenate multimodal sequence ---- #
        visual_embeds = visual_embeds.to(text_embeds.dtype)
        inputs_embeds = torch.cat([visual_embeds, text_embeds], dim=1)
        full_attn_mask = torch.cat(
            [
                torch.ones(B, n_vis, dtype=attn_mask.dtype, device=device),
                attn_mask,
            ],
            dim=1,
        )

        # ---- 4. LLM forward (output_hidden_states=True for last layer) ---- #
        grad_llm = self.mode != "frozen"
        llm_ctx = torch.enable_grad() if grad_llm else torch.no_grad()
        with llm_ctx:
            out = self.llm_backbone(
                inputs_embeds=inputs_embeds,
                attention_mask=full_attn_mask,
                output_hidden_states=True,
                return_dict=True,
            )

        # Extract last-layer hidden states.
        hidden = None
        if hasattr(out, "last_hidden_state") and out.last_hidden_state is not None:
            hidden = out.last_hidden_state
        elif hasattr(out, "hidden_states") and out.hidden_states is not None:
            hidden = out.hidden_states[-1]
        if hidden is None:
            raise RuntimeError(
                "llm_backbone did not return hidden states. "
                "Ensure output_hidden_states=True is honoured."
            )

        # ---- 5. NaN guard ---- #
        if not torch.isfinite(hidden).all():
            n_bad = (~torch.isfinite(hidden)).sum().item()
            raise ValueError(
                f"OpenVLA hidden_states contain {n_bad} non-finite values "
                f"(bf16 overflow or NaN-propagating inputs)."
            )

        return {
            "hidden_states":  hidden,         # (B, T_tok, D_h)
            "attention_mask": full_attn_mask, # (B, T_tok)
        }
