"""
scripts/inspect_openvla_modules.py

MANDATORY first-run inspection: prints the OpenVLA-OFT module tree so the
attribute names assumed by OpenVLABackboneWrapper can be verified.

Run once after `pip install transformers peft accelerate` is in place and
network/HF cache is reachable.  If the printed names differ from the
candidates in OpenVLABackboneWrapper._MODULE_CANDIDATES, update that list.

Usage:
  export HF_ENDPOINT=https://hf-mirror.com
  export HF_HOME=/root/autodl-tmp/hf_cache
  python scripts/inspect_openvla_modules.py
"""

from __future__ import annotations

import os
import sys

import torch

CKPT = os.environ.get(
    "OPENVLA_CKPT",
    "moojink/openvla-7b-oft-finetuned-libero-spatial",
)


def _try_load(ckpt: str):
    """Try AutoModelForVision2Seq, fall back to AutoModelForCausalLM."""
    try:
        from transformers import AutoModelForVision2Seq
        print(f"[load] Trying AutoModelForVision2Seq …")
        return AutoModelForVision2Seq.from_pretrained(
            ckpt, trust_remote_code=True,
            torch_dtype=torch.bfloat16, low_cpu_mem_usage=True,
        )
    except (ImportError, AttributeError, Exception) as e:
        print(f"[load] AutoModelForVision2Seq failed: {type(e).__name__}: {e}")
        from transformers import AutoModelForCausalLM
        print(f"[load] Falling back to AutoModelForCausalLM …")
        return AutoModelForCausalLM.from_pretrained(
            ckpt, trust_remote_code=True,
            torch_dtype=torch.bfloat16, low_cpu_mem_usage=True,
        )


def main() -> None:
    print(f"=== Loading {CKPT} ===")
    M = _try_load(CKPT)

    print(f"\n=== Top-level model class: {type(M).__name__} ===")
    print(f"=== Top-level children ===")
    for name, child in M.named_children():
        n = sum(p.numel() for p in child.parameters())
        print(f"  {name:32s} {type(child).__name__:32s} params={n/1e6:8.1f}M")

    print(f"\n=== Module names matching vision/llm/projector/siglip/dino ===")
    for name, _ in M.named_modules():
        low = name.lower()
        if any(kw in low for kw in
               ["vision", "llm", "language", "projector", "siglip", "dino"]):
            depth = name.count(".")
            if depth <= 2:
                print(f"  {name}")

    print(f"\n=== Config ===")
    cfg = getattr(M, "config", None)
    if cfg is not None:
        for attr in ("hidden_size", "text_config", "llm_backbone_id",
                     "vision_backbone_id", "image_size"):
            if hasattr(cfg, attr):
                v = getattr(cfg, attr)
                if hasattr(v, "hidden_size"):
                    print(f"  config.{attr}.hidden_size = {v.hidden_size}")
                else:
                    print(f"  config.{attr} = {v}")

    print(f"\n=== Probing common attribute name candidates ===")
    candidates = [
        ("vision_backbone", "projector", "llm_backbone"),
        ("vision_tower",    "multi_modal_projector", "language_model"),
        ("vision_model",    "projector", "language_model"),
    ]
    for triple in candidates:
        present = [hasattr(M, n) for n in triple]
        marker = "OK" if all(present) else "FAIL"
        print(f"  [{marker}] {triple} -> present={present}")

    print(f"\n=== Vision backbone children (if found) ===")
    vb = None
    for n in ("vision_backbone", "vision_tower", "vision_model"):
        if hasattr(M, n):
            vb = getattr(M, n)
            print(f"  vision module attribute = {n!r}")
            break
    if vb is not None:
        for name, child in vb.named_children():
            print(f"    {name:32s} {type(child).__name__}")

    print("\nDone.  Verify these names match OpenVLABackboneWrapper.")


if __name__ == "__main__":
    main()
