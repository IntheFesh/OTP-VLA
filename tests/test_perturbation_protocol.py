"""
Tests for otp/data/perturbation_protocol.py

Covers (Stage 3 spec):
  1. Each generator produces output != original (per type).
  2. Each validator rejects obvious invalid input.
  3. validate_object_referent_swap distinguishes informative vs uninformative
     perturbations via the action-difference check.
  4. generate_full_suite writes a manifest with the expected schema.
"""

import json
from pathlib import Path

import numpy as np
import pytest

from otp.data.perturbation_protocol import (
    NoPerturbationAvailable,
    PerturbationProtocol,
)

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def proto() -> PerturbationProtocol:
    return PerturbationProtocol(
        seed=42,
        num_samples_per_task_per_type=3,
        max_attempts_per_sample=10,
    )


def _fake_demo(task_id: str = "task_001") -> dict:
    """A minimal valid LIBERO-like demo dict used across suite tests."""
    return {
        "task_id":       task_id,
        "instruction":   "pick up the red bowl on the left",
        "target_object": "red_bowl",
        "scene_objects": ["red_bowl", "blue_plate", "green_cup"],
        "image":         np.random.default_rng(0).integers(
                            0, 256, (32, 32, 3), dtype=np.uint8
                        ),
        "action":        np.array([0.1, 0.0, 0.0, 0.0, 0.0, 0.0, -1.0]),
        "object_actions": {
            "blue_plate": np.array([0.5, 0.0, 0.0, 0.0, 0.0, 0.0, -1.0]),
            "green_cup":  np.array([0.0, 0.5, 0.0, 0.0, 0.0, 0.0, -1.0]),
        },
    }


# ---------------------------------------------------------------------------
# 1. Each generator produces a different output
# ---------------------------------------------------------------------------

class TestEachGeneratorProducesDifferentOutput:

    def test_instruction_rephrase(self, proto):
        original = "pick up the red bowl"
        perturbed = proto.gen_instruction_rephrase(original)
        assert perturbed != original, f"rephrase produced no change: {perturbed!r}"

    def test_object_referent_swap(self, proto):
        scene = ["red_bowl", "blue_plate", "green_cup"]
        perturbed, new_target = proto.gen_object_referent_swap(
            "pick up the red_bowl", scene, "red_bowl"
        )
        assert new_target != "red_bowl"
        assert new_target in scene
        assert perturbed != "pick up the red_bowl"

    def test_distractor_inject(self, proto):
        original = "pick up the red bowl"
        scene = ["blue_plate", "green_cup"]
        perturbed = proto.gen_distractor_inject(original, scene)
        assert perturbed != original
        assert any(obj in perturbed for obj in scene)

    def test_spatial_relation_perturb(self, proto):
        original = "pick up the bowl on the left"
        perturbed = proto.gen_spatial_relation_perturb(original)
        assert perturbed != original

    def test_visual_style(self, proto):
        img = np.random.default_rng(0).integers(0, 256, (16, 16, 3), dtype=np.uint8)
        for style in ("hue_shift", "brightness", "contrast", "random"):
            out = proto.gen_visual_style(img, style=style)
            assert out.shape == img.shape
            assert out.dtype == img.dtype
            assert not np.array_equal(out, img), f"style={style} produced no change"

    # Negative: raises when no candidate available.
    def test_rephrase_raises_when_no_vocab_match(self, proto):
        with pytest.raises(NoPerturbationAvailable):
            proto.gen_instruction_rephrase("xyzzy quux")

    def test_swap_raises_when_only_target_in_scene(self, proto):
        with pytest.raises(NoPerturbationAvailable):
            proto.gen_object_referent_swap(
                "pick up the bowl", ["bowl"], "bowl"
            )

    def test_spatial_raises_when_no_relation(self, proto):
        with pytest.raises(NoPerturbationAvailable):
            proto.gen_spatial_relation_perturb("pick up the bowl")


# ---------------------------------------------------------------------------
# 2. Each validator rejects obvious invalid input
# ---------------------------------------------------------------------------

class TestEachValidatorRejectsInvalid:

    def test_rephrase_rejects_identical(self, proto):
        ok, reason = proto.validate_instruction_rephrase("foo bar", "foo bar")
        assert not ok
        assert "identical" in reason

    def test_rephrase_rejects_empty(self, proto):
        ok, _ = proto.validate_instruction_rephrase("foo", "")
        assert not ok

    def test_swap_rejects_same_target(self, proto):
        ok, reason = proto.validate_object_referent_swap(
            "pick the bowl", "pick the bowl",
            "bowl", "bowl",
            ["bowl", "cup"],
            np.zeros(7),
        )
        assert not ok

    def test_swap_rejects_target_not_in_scene(self, proto):
        ok, reason = proto.validate_object_referent_swap(
            "pick the bowl", "pick the apple",
            "bowl", "apple",
            ["bowl", "cup"],
            np.zeros(7),
        )
        assert not ok
        assert "not in scene_objects" in reason

    def test_distractor_inject_rejects_no_new_obj(self, proto):
        ok, _ = proto.validate_distractor_inject(
            "pick the bowl", "pick the bowl now",
            ["bowl", "cup"],
        )
        # "cup" not present in perturbed → rejected.
        assert not ok

    def test_spatial_rejects_identical(self, proto):
        ok, _ = proto.validate_spatial_relation_perturb("foo", "foo")
        assert not ok

    def test_visual_rejects_identical(self, proto):
        img = np.zeros((4, 4, 3), dtype=np.uint8)
        ok, _ = proto.validate_visual_style(img, img)
        assert not ok

    def test_visual_rejects_shape_mismatch(self, proto):
        a = np.zeros((4, 4, 3), dtype=np.uint8)
        b = np.zeros((5, 4, 3), dtype=np.uint8)
        ok, reason = proto.validate_visual_style(a, b)
        assert not ok
        assert "shape" in reason


# ---------------------------------------------------------------------------
# 3. Referent swap actually changes the action
# ---------------------------------------------------------------------------

class TestReferentSwapActuallyChangesAction:

    def test_validator_accepts_when_actions_differ(self, proto):
        """When new_action is far from original_action, validator accepts."""
        original_action = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, -1.0])
        new_action = np.array([1.0, 0.5, 0.3, 0.0, 0.0, 0.0, -1.0])
        ok, reason = proto.validate_object_referent_swap(
            "pick the bowl", "pick the cup",
            "bowl", "cup",
            ["bowl", "cup"],
            original_action, new_action=new_action,
        )
        assert ok, f"Expected accept; got reject: {reason}"

    def test_validator_rejects_when_actions_coincide(self, proto):
        """When new_action ≈ original_action, perturbation is uninformative."""
        original_action = np.array([0.5, 0.2, 0.1, 0.0, 0.0, 0.0, -1.0])
        new_action = original_action + 0.001       # tiny diff < min_action_diff
        ok, reason = proto.validate_object_referent_swap(
            "pick the bowl", "pick the cup",
            "bowl", "cup",
            ["bowl", "cup"],
            original_action,
            new_action=new_action,
            min_action_diff=0.05,
        )
        assert not ok
        assert "no information value" in reason

    def test_validator_skips_action_check_when_new_action_none(self, proto):
        """Without new_action, validator falls back to non-action criteria only."""
        ok, _ = proto.validate_object_referent_swap(
            "pick the bowl", "pick the cup",
            "bowl", "cup",
            ["bowl", "cup"],
            np.zeros(7),
            new_action=None,
        )
        assert ok


# ---------------------------------------------------------------------------
# 4. generate_full_suite writes correct manifest schema
# ---------------------------------------------------------------------------

class TestFullSuiteManifestStructure:

    def test_manifest_file_created(self, proto, tmp_path):
        result = proto.generate_full_suite([_fake_demo()], str(tmp_path))
        assert "manifest_path" in result
        assert (tmp_path / "manifest.json").exists()

    def test_manifest_has_one_entry_per_task(self, proto, tmp_path):
        demos = [_fake_demo("task_a"), _fake_demo("task_b")]
        proto.generate_full_suite(demos, str(tmp_path))
        manifest = json.loads((tmp_path / "manifest.json").read_text())
        assert "task_a" in manifest
        assert "task_b" in manifest

    def test_manifest_has_all_perturbation_types(self, proto, tmp_path):
        proto.generate_full_suite([_fake_demo()], str(tmp_path))
        manifest = json.loads((tmp_path / "manifest.json").read_text())
        for ptype in PerturbationProtocol.PERTURBATION_TYPES:
            assert ptype in manifest["task_001"], f"Missing type: {ptype}"

    def test_manifest_records_invalid_count(self, proto, tmp_path):
        proto.generate_full_suite([_fake_demo()], str(tmp_path))
        manifest = json.loads((tmp_path / "manifest.json").read_text())
        assert "invalid_count" in manifest["task_001"]
        assert isinstance(manifest["task_001"]["invalid_count"], int)

    def test_samples_directory_populated(self, proto, tmp_path):
        proto.generate_full_suite([_fake_demo()], str(tmp_path))
        samples_dir = tmp_path / "samples"
        assert samples_dir.exists()
        # At least some samples should be generated successfully.
        n = len(list(samples_dir.glob("*.json")))
        assert n > 0, "No sample JSON files were created"

    def test_each_sample_json_has_required_fields(self, proto, tmp_path):
        proto.generate_full_suite([_fake_demo()], str(tmp_path))
        sample_files = list((tmp_path / "samples").glob("*.json"))
        assert sample_files, "No samples generated"
        for path in sample_files[:5]:
            record = json.loads(path.read_text())
            for key in ("task_id", "perturbation_type", "sample_idx",
                        "original_instruction"):
                assert key in record, f"Missing key {key} in {path.name}"

    def test_visual_style_writes_npy_alongside(self, proto, tmp_path):
        proto.generate_full_suite([_fake_demo()], str(tmp_path))
        visual_records = [
            p for p in (tmp_path / "samples").glob("*_visual_style_*.json")
        ]
        for record_path in visual_records:
            record = json.loads(record_path.read_text())
            assert "perturbed_image_path" in record
            npy_path = record_path.parent / record["perturbed_image_path"]
            assert npy_path.exists()
            arr = np.load(npy_path)
            assert arr.shape == (32, 32, 3)
            assert arr.dtype == np.uint8

    def test_stats_returned(self, proto, tmp_path):
        result = proto.generate_full_suite([_fake_demo()], str(tmp_path))
        assert "stats" in result
        assert "task_001" in result["stats"]
        for ptype in PerturbationProtocol.PERTURBATION_TYPES:
            assert ptype in result["stats"]["task_001"]


# ---------------------------------------------------------------------------
# 5. Determinism
# ---------------------------------------------------------------------------

class TestDeterminism:

    def test_same_seed_same_output(self):
        p1 = PerturbationProtocol(seed=123)
        p2 = PerturbationProtocol(seed=123)
        s1 = p1.gen_instruction_rephrase("pick up the red bowl")
        s2 = p2.gen_instruction_rephrase("pick up the red bowl")
        assert s1 == s2, f"Seeded determinism broken: {s1!r} vs {s2!r}"
