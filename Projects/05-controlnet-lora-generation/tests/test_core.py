from __future__ import annotations

import json
from pathlib import Path
import sys
import unittest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from controlnet_lora.contracts import ManifestError, validate_manifest  # noqa: E402
from controlnet_lora.evaluation import edge_metrics  # noqa: E402
from controlnet_lora.inference import load_config  # noqa: E402
from controlnet_lora.provenance import canonical_hash  # noqa: E402


class ContractTests(unittest.TestCase):
    def test_sample_manifest_is_valid(self) -> None:
        sample = ROOT / "data" / "sample"
        rows = validate_manifest(sample / "manifest.jsonl", sample)
        self.assertEqual([row.sample_id for row in rows], ["sku_001"])

    def test_missing_condition_is_rejected(self) -> None:
        sample = ROOT / "data" / "sample"
        manifest = ROOT / "tests" / "fixtures" / "missing_condition.jsonl"
        with self.assertRaisesRegex(ManifestError, "missing conditioning_image"):
            validate_manifest(manifest, sample)

    def test_capture_group_cannot_cross_splits(self) -> None:
        sample = ROOT / "data" / "sample"
        manifest = ROOT / "tests" / "fixtures" / "split_leakage.jsonl"
        with self.assertRaisesRegex(ManifestError, "leakage"):
            validate_manifest(manifest, sample)


class MetricTests(unittest.TestCase):
    def test_tolerance_accepts_one_pixel_shift(self) -> None:
        result = edge_metrics({(2, 2), (3, 3)}, {(2, 3), (3, 4)}, tolerance=1.0)
        self.assertEqual(result.f1, 1.0)

    def test_empty_prediction_has_zero_score(self) -> None:
        result = edge_metrics(set(), {(1, 1)}, tolerance=1.0)
        self.assertEqual(result.f1, 0.0)


class ReproducibilityTests(unittest.TestCase):
    def test_config_hash_ignores_mapping_order(self) -> None:
        self.assertEqual(canonical_hash({"b": 2, "a": 1}), canonical_hash({"a": 1, "b": 2}))

    def test_inference_config_rejects_invalid_scale(self) -> None:
        path = ROOT / "tests" / "fixtures" / "invalid_inference.json"
        with self.assertRaisesRegex(ValueError, "lora_scale"):
            load_config(path)


if __name__ == "__main__":
    unittest.main()
