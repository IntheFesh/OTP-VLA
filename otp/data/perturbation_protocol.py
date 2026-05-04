"""
PerturbationProtocol: structured perturbation generation + validation + manifest.

Paper §V.B reports this as a "controlled perturbation suite" — NOT a community
benchmark.  Supplementary material publishes (a) every generator rule, (b) the
vocab table at configs/perturbation_vocab.yaml, and (c) the per-task /
per-type valid-sample manifest produced by generate_full_suite().

Five perturbation types (PERTURBATION_TYPES):
  instruction_rephrase     — synonym replacement from public vocab
  object_referent_swap     — replace target object name with a scene distractor
  distractor_inject        — add a "ignore the X" clause referencing a distractor
  spatial_relation_perturb — flip spatial relation words (left↔right, etc.)
  visual_style             — hue / brightness / contrast jitter on RGB image

For every (gen_X, validate_X) pair:
  - gen_X may raise NoPerturbationAvailable when input has no valid candidate
    (e.g. instruction with no synonym-vocab words).
  - validate_X returns (ok: bool, reason: str).  reason is a short diagnostic
    string used in manifest entries when the candidate is rejected.

Workflow (generate_full_suite):
  for each demo:
    for each perturbation_type:
      for sample_idx in range(num_samples_per_task_per_type):
        for attempt in range(max_attempts_per_sample):
          candidate = gen_X(...)
          ok, reason = validate_X(...)
          if ok: save candidate, break
        else:
          manifest['invalid_count'] += 1
"""

from __future__ import annotations

import json
import logging
import re
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import yaml

logger = logging.getLogger(__name__)

# Default vocab file ships with the repo.
_DEFAULT_VOCAB_PATH = (
    Path(__file__).resolve().parents[2] / "configs" / "perturbation_vocab.yaml"
)


# ---------------------------------------------------------------------------
# Exceptions
# ---------------------------------------------------------------------------

class NoPerturbationAvailable(Exception):
    """Raised by a gen_X when the input has no valid perturbation candidate."""


# ---------------------------------------------------------------------------
# PerturbationProtocol
# ---------------------------------------------------------------------------

class PerturbationProtocol:
    """
    Structured perturbation generator + validator + manifest writer.

    Args:
        seed:                              RNG seed (numpy default_rng).
        num_samples_per_task_per_type:     Target valid samples per (task, type) cell.
        max_attempts_per_sample:           Retry budget before recording 'unavailable'.
        vocab_path:                        Path to perturbation_vocab.yaml.
                                           Defaults to configs/perturbation_vocab.yaml.
    """

    PERTURBATION_TYPES: Tuple[str, ...] = (
        "instruction_rephrase",
        "object_referent_swap",
        "distractor_inject",
        "spatial_relation_perturb",
        "visual_style",
    )

    def __init__(
        self,
        seed: int = 42,
        num_samples_per_task_per_type: int = 30,
        max_attempts_per_sample: int = 5,
        vocab_path: Optional[Union[str, Path]] = None,
    ) -> None:
        self.rng = np.random.default_rng(seed)
        self.num_samples_per_task_per_type = num_samples_per_task_per_type
        self.max_attempts_per_sample = max_attempts_per_sample

        path = Path(vocab_path) if vocab_path is not None else _DEFAULT_VOCAB_PATH
        with open(path, "r") as f:
            self.vocab: Dict[str, Any] = yaml.safe_load(f)

        # Pre-compute bidirectional spatial inversions.
        raw_inv = self.vocab.get("spatial_inversions", {})
        self._spatial_inversions: Dict[str, str] = {}
        for a, b in raw_inv.items():
            self._spatial_inversions[a] = b
            self._spatial_inversions[b] = a

    # ======================================================================
    # GENERATORS
    # ======================================================================

    # ---------- 1. instruction_rephrase ---------------------------------- #

    def gen_instruction_rephrase(self, original: str) -> str:
        """
        Replace one or more vocab words with a synonym from the public table.

        Raises:
            NoPerturbationAvailable: when no token in `original` matches any
                                     synonym key.
        """
        synonyms: Dict[str, List[str]] = self.vocab.get("synonyms", {})
        tokens = original.split()
        replaceable = [
            (i, tok) for i, tok in enumerate(tokens)
            if tok.lower().strip(".,!?") in synonyms
        ]
        if not replaceable:
            raise NoPerturbationAvailable(
                f"No vocab synonym matches any token in: {original!r}"
            )

        i, tok = replaceable[int(self.rng.integers(len(replaceable)))]
        key = tok.lower().strip(".,!?")
        choices = synonyms[key]
        new_word = choices[int(self.rng.integers(len(choices)))]

        # Preserve trailing punctuation if any.
        suffix = "".join(c for c in tok if c in ".,!?")
        tokens[i] = new_word + suffix
        return " ".join(tokens)

    # ---------- 2. object_referent_swap ---------------------------------- #

    def gen_object_referent_swap(
        self,
        original: str,
        scene_objects: List[str],
        target_object: str,
    ) -> Tuple[str, str]:
        """
        Replace target_object in instruction with a scene distractor.

        Returns:
            (perturbed_instruction, new_target_object).

        Raises:
            NoPerturbationAvailable: when scene has no distractor != target.
        """
        distractors = [o for o in scene_objects if o != target_object]
        if not distractors:
            raise NoPerturbationAvailable(
                f"No distractor in scene_objects to swap with {target_object!r}"
            )
        new_target = distractors[int(self.rng.integers(len(distractors)))]

        if target_object in original:
            perturbed = original.replace(target_object, new_target)
        else:
            # Fallback: append a clarifying phrase (still encodes intent change).
            perturbed = f"{original} (target: {new_target})"
        return perturbed, new_target

    # ---------- 3. distractor_inject ------------------------------------- #

    def gen_distractor_inject(
        self,
        original: str,
        scene_objects: List[str],
    ) -> str:
        """
        Append a distractor-mention clause via a vocab template.

        Raises:
            NoPerturbationAvailable: when scene_objects is empty.
        """
        if not scene_objects:
            raise NoPerturbationAvailable("scene_objects is empty")
        templates: List[str] = self.vocab.get("distractor_templates", [])
        if not templates:
            raise NoPerturbationAvailable("vocab.distractor_templates is empty")

        obj = scene_objects[int(self.rng.integers(len(scene_objects)))]
        template = templates[int(self.rng.integers(len(templates)))]
        return original + template.format(obj=obj)

    # ---------- 4. spatial_relation_perturb ------------------------------ #

    def gen_spatial_relation_perturb(self, original: str) -> str:
        """
        Flip one spatial-relation word per the inversions table.

        Raises:
            NoPerturbationAvailable: when no spatial relation found.
        """
        # Build a regex that matches any spatial key as a whole word/phrase.
        # Phrases with underscores (in_front_of) or spaces are normalised.
        keys = sorted(self._spatial_inversions.keys(), key=len, reverse=True)
        matches: List[Tuple[int, int, str]] = []
        for key in keys:
            pattern = key.replace("_", r"[ _]")
            for m in re.finditer(rf"\b{pattern}\b", original, flags=re.IGNORECASE):
                matches.append((m.start(), m.end(), key))

        if not matches:
            raise NoPerturbationAvailable(
                f"No spatial relation in instruction: {original!r}"
            )

        start, end, key = matches[int(self.rng.integers(len(matches)))]
        replacement = self._spatial_inversions[key]
        return original[:start] + replacement + original[end:]

    # ---------- 5. visual_style ------------------------------------------ #

    def gen_visual_style(
        self,
        image: np.ndarray,
        style: str = "random",
    ) -> np.ndarray:
        """
        Apply hue / brightness / contrast jitter using only numpy.

        Args:
            image: (H, W, 3) uint8 RGB array.
            style: 'hue_shift', 'brightness', 'contrast', or 'random'.

        Returns:
            Perturbed image, same shape and dtype.
        """
        if image.ndim != 3 or image.shape[-1] != 3:
            raise ValueError(f"Expected (H, W, 3), got {image.shape}")
        if image.dtype != np.uint8:
            raise ValueError(f"Expected uint8, got {image.dtype}")

        cfg = self.vocab.get("visual_style", {})

        if style == "random":
            style = ["hue_shift", "brightness", "contrast"][
                int(self.rng.integers(3))
            ]

        if style == "hue_shift":
            lo, hi = cfg.get("hue_shift_range", [-30, 30])
            shift = int(self.rng.integers(lo, hi + 1))
            # Crude RGB-channel rotation as a hue proxy (numpy-only).
            out = np.roll(image, shift // 15, axis=-1)
            out = np.clip(out.astype(np.int32) + shift, 0, 255).astype(np.uint8)

        elif style == "brightness":
            lo, hi = cfg.get("brightness_range", [0.6, 1.4])
            factor = float(self.rng.uniform(lo, hi))
            out = np.clip(image.astype(np.float32) * factor, 0, 255).astype(np.uint8)

        elif style == "contrast":
            lo, hi = cfg.get("contrast_range", [0.6, 1.4])
            factor = float(self.rng.uniform(lo, hi))
            out = np.clip(
                (image.astype(np.float32) - 128.0) * factor + 128.0, 0, 255
            ).astype(np.uint8)

        else:
            raise ValueError(f"Unknown style: {style!r}")

        return out

    # ======================================================================
    # VALIDATORS
    # ======================================================================

    @staticmethod
    def _basic_string_check(perturbed: str) -> Tuple[bool, str]:
        """Common reject criteria for any string-valued perturbation."""
        if not isinstance(perturbed, str):
            return False, "perturbed not a string"
        if not perturbed.strip():
            return False, "perturbed is empty"
        if len(perturbed) > 500:
            return False, f"perturbed too long ({len(perturbed)} chars)"
        return True, "ok"

    # ---------- 1. instruction_rephrase ---------------------------------- #

    def validate_instruction_rephrase(
        self,
        original: str,
        perturbed: str,
    ) -> Tuple[bool, str]:
        """
        Reject when perturbed equals original or fails basic string checks.
        Length must remain within ±50 % of original to preserve semantics.
        """
        ok, reason = self._basic_string_check(perturbed)
        if not ok:
            return False, reason
        if perturbed == original:
            return False, "perturbed identical to original"
        if not (0.5 * len(original) <= len(perturbed) <= 1.5 * len(original) + 5):
            return False, f"length out of range ({len(perturbed)} vs {len(original)})"
        return True, "ok"

    # ---------- 2. object_referent_swap ---------------------------------- #

    def validate_object_referent_swap(
        self,
        original: str,
        perturbed: str,
        original_target: str,
        new_target: str,
        scene_objects: List[str],
        original_action: np.ndarray,
        new_action: Optional[np.ndarray] = None,
        min_action_diff: float = 0.05,
    ) -> Tuple[bool, str]:
        """
        Reject when:
          - basic string check fails;
          - new_target == original_target;
          - new_target not in scene_objects;
          - perturbed equals original;
          - new_action provided AND ‖new_action - original_action‖ < min_action_diff
            (perturbation has no information value when GT actions coincide).
        """
        ok, reason = self._basic_string_check(perturbed)
        if not ok:
            return False, reason
        if perturbed == original:
            return False, "perturbed identical to original"
        if new_target == original_target:
            return False, "new_target equals original_target"
        if new_target not in scene_objects:
            return False, f"new_target {new_target!r} not in scene_objects"

        if new_action is not None:
            diff = float(np.linalg.norm(np.asarray(new_action) - np.asarray(original_action)))
            if diff < min_action_diff:
                return False, (
                    f"GT action diff {diff:.4f} < min_action_diff {min_action_diff} "
                    f"— perturbation has no information value"
                )
        return True, "ok"

    # ---------- 3. distractor_inject ------------------------------------- #

    def validate_distractor_inject(
        self,
        original: str,
        perturbed: str,
        scene_objects: List[str],
    ) -> Tuple[bool, str]:
        """
        Reject when basic string check fails, perturbed == original, or
        no scene_object name is mentioned in the new clause.
        """
        ok, reason = self._basic_string_check(perturbed)
        if not ok:
            return False, reason
        if perturbed == original:
            return False, "perturbed identical to original"
        if not any(obj in perturbed and obj not in original for obj in scene_objects):
            return False, "no new scene object referenced in perturbed instruction"
        return True, "ok"

    # ---------- 4. spatial_relation_perturb ------------------------------ #

    def validate_spatial_relation_perturb(
        self,
        original: str,
        perturbed: str,
    ) -> Tuple[bool, str]:
        """Reject when basic check fails or perturbed == original."""
        ok, reason = self._basic_string_check(perturbed)
        if not ok:
            return False, reason
        if perturbed == original:
            return False, "perturbed identical to original"
        return True, "ok"

    # ---------- 5. visual_style ------------------------------------------ #

    def validate_visual_style(
        self,
        original: np.ndarray,
        perturbed: np.ndarray,
    ) -> Tuple[bool, str]:
        """
        Reject when shapes/dtypes mismatch, images are byte-identical, or
        the relative L1 difference is below 1 % (perturbation too weak).
        """
        if original.shape != perturbed.shape:
            return False, f"shape mismatch: {original.shape} vs {perturbed.shape}"
        if original.dtype != perturbed.dtype:
            return False, f"dtype mismatch: {original.dtype} vs {perturbed.dtype}"
        if np.array_equal(original, perturbed):
            return False, "perturbed identical to original"
        l1 = float(np.abs(original.astype(np.int32) - perturbed.astype(np.int32)).mean())
        if l1 < 2.55:        # ~ 1 % of 255
            return False, f"L1 diff {l1:.2f} < 2.55 (perturbation too weak)"
        return True, "ok"

    # ======================================================================
    # TOP-LEVEL WORKFLOW
    # ======================================================================

    def generate_full_suite(
        self,
        libero_demos: List[dict],
        output_dir: Union[str, Path],
    ) -> dict:
        """
        Generate `num_samples_per_task_per_type` valid samples per (task, type).

        Each demo in `libero_demos` must be a dict with keys:
          task_id           (str)
          instruction       (str)
          target_object     (str)
          scene_objects     (list[str])
          image             (H,W,3) uint8 ndarray  — for visual_style
          action            (D,) ndarray           — for referent_swap validation
          object_actions    (dict[str, ndarray]) optional — per-object GT action
                                                            for referent_swap.

        Output layout:
          output_dir/
            manifest.json        — see _write_manifest
            samples/
              <task>_<type>_<idx>.json   — text-only samples
              <task>_<type>_<idx>.npy    — visual_style images
        """
        out = Path(output_dir)
        (out / "samples").mkdir(parents=True, exist_ok=True)

        manifest: Dict[str, Dict[str, Any]] = {}
        stats: Dict[str, Dict[str, int]] = {}

        for demo in libero_demos:
            task_id = demo["task_id"]
            manifest[task_id] = {"invalid_count": 0}
            stats[task_id] = {}

            for ptype in self.PERTURBATION_TYPES:
                valid_samples: List[str] = []
                invalid = 0

                for idx in range(self.num_samples_per_task_per_type):
                    sample_path = self._try_generate_sample(
                        demo, ptype, idx, out
                    )
                    if sample_path is None:
                        invalid += 1
                    else:
                        valid_samples.append(sample_path)

                manifest[task_id][ptype] = valid_samples
                manifest[task_id]["invalid_count"] += invalid
                stats[task_id][ptype] = len(valid_samples)

        manifest_path = out / "manifest.json"
        with open(manifest_path, "w") as f:
            json.dump(manifest, f, indent=2, sort_keys=True)

        logger.info("[PerturbationProtocol] manifest written: %s", manifest_path)
        return {"manifest_path": str(manifest_path), "stats": stats}

    # ------------------------------------------------------------------
    # Internal: per-sample generation with retry budget.
    # ------------------------------------------------------------------

    def _try_generate_sample(
        self,
        demo: dict,
        ptype: str,
        idx: int,
        out_dir: Path,
    ) -> Optional[str]:
        """Attempt up to `max_attempts_per_sample` to produce one valid sample."""
        for attempt in range(self.max_attempts_per_sample):
            try:
                sample = self._generate_one(demo, ptype)
            except NoPerturbationAvailable as e:
                # Cannot recover by retrying — input itself has no candidate.
                logger.debug(
                    "[%s/%s/%d] no perturbation available: %s",
                    demo.get("task_id", "?"), ptype, idx, e,
                )
                return None
            if sample is None:
                continue   # validator rejected, retry
            # Save and return path.
            return self._save_sample(sample, demo, ptype, idx, out_dir)
        return None

    def _generate_one(self, demo: dict, ptype: str) -> Optional[dict]:
        """Generate one candidate; return dict if valid, None if rejected."""
        if ptype == "instruction_rephrase":
            perturbed = self.gen_instruction_rephrase(demo["instruction"])
            ok, reason = self.validate_instruction_rephrase(
                demo["instruction"], perturbed
            )
            if not ok:
                return None
            return {"perturbed_instruction": perturbed}

        if ptype == "object_referent_swap":
            perturbed, new_target = self.gen_object_referent_swap(
                demo["instruction"], demo["scene_objects"], demo["target_object"]
            )
            object_actions = demo.get("object_actions") or {}
            new_action = object_actions.get(new_target)
            ok, reason = self.validate_object_referent_swap(
                demo["instruction"], perturbed,
                demo["target_object"], new_target,
                demo["scene_objects"], demo["action"],
                new_action=new_action,
            )
            if not ok:
                return None
            return {
                "perturbed_instruction": perturbed,
                "new_target_object": new_target,
            }

        if ptype == "distractor_inject":
            perturbed = self.gen_distractor_inject(
                demo["instruction"], demo["scene_objects"]
            )
            ok, reason = self.validate_distractor_inject(
                demo["instruction"], perturbed, demo["scene_objects"]
            )
            if not ok:
                return None
            return {"perturbed_instruction": perturbed}

        if ptype == "spatial_relation_perturb":
            perturbed = self.gen_spatial_relation_perturb(demo["instruction"])
            ok, reason = self.validate_spatial_relation_perturb(
                demo["instruction"], perturbed
            )
            if not ok:
                return None
            return {"perturbed_instruction": perturbed}

        if ptype == "visual_style":
            img = demo["image"]
            perturbed_img = self.gen_visual_style(img, style="random")
            ok, reason = self.validate_visual_style(img, perturbed_img)
            if not ok:
                return None
            return {"perturbed_image": perturbed_img}

        raise ValueError(f"Unknown perturbation type: {ptype}")

    def _save_sample(
        self,
        sample: dict,
        demo: dict,
        ptype: str,
        idx: int,
        out_dir: Path,
    ) -> str:
        """Write sample JSON (and NPY for visual_style) to disk; return JSON path."""
        base = f"{demo['task_id']}_{ptype}_{idx:03d}"
        json_path = out_dir / "samples" / f"{base}.json"

        record: Dict[str, Any] = {
            "task_id":            demo["task_id"],
            "perturbation_type":  ptype,
            "sample_idx":         idx,
            "original_instruction": demo["instruction"],
        }

        if "perturbed_image" in sample:
            npy_path = out_dir / "samples" / f"{base}.npy"
            np.save(npy_path, sample["perturbed_image"])
            record["perturbed_image_path"] = str(npy_path.name)
        else:
            record.update(sample)

        with open(json_path, "w") as f:
            json.dump(record, f, indent=2, sort_keys=True)
        return str(json_path.name)
