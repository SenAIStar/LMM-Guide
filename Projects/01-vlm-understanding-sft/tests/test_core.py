import json
from pathlib import Path
import sys
import unittest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from vlm_product.contracts import parse_assistant_json, validate_prediction, validate_training_record  # noqa: E402
from vlm_product.data_pipeline import (  # noqa: E402
    abo_candidate,
    assert_no_group_leakage,
    assert_no_media_leakage,
    stable_group_split,
)
from vlm_product.evaluation import evaluate_records  # noqa: E402
from vlm_product.provenance import sha256_file, validate_media_files  # noqa: E402
from vlm_product.qwen3vl_adapter import InferenceConfig, Qwen3VLAdapter  # noqa: E402
from vlm_product.service import route_prediction  # noqa: E402


def valid_prediction(decision: str = "accept") -> dict:
    return {
        "schema_version": "1.0",
        "product_type": "chair",
        "attributes": {"color": ["black"], "material": ["wood"]},
        "visible_text": [],
        "evidence": [
            {"field": "product_type", "media_index": 0, "support": "image_level"},
            {"field": "attributes.color", "media_index": 0, "support": "image_level"},
            {"field": "attributes.material", "media_index": 0, "support": "image_level"},
        ],
        "decision": decision,
    }


def valid_record() -> dict:
    return {
        "sample_id": "s1",
        "group_id": "g1",
        "split": "train",
        "images": ["sample/assets/demo_product.ppm"],
        "media_sha256": ["bfb01feb120b746a4e277cf1f4dadd71b3bdb58c05e9e2df1877910cd6ae6a9b"],
        "messages": [
            {"role": "user", "content": "<image> extract fields"},
            {"role": "assistant", "content": json.dumps(valid_prediction())},
        ],
        "source": {
            "dataset": "synthetic",
            "snapshot_id": "v1",
            "license_id": "CC0-1.0",
            "source_uri": "local-generated",
        },
        "review_required": False,
    }


class ContractTests(unittest.TestCase):
    def test_current_message_format_is_valid(self) -> None:
        self.assertEqual(validate_training_record(valid_record()), [])

    def test_legacy_conversation_format_is_rejected(self) -> None:
        record = valid_record()
        record["conversations"] = record.pop("messages")
        self.assertTrue(any("messages" in error for error in validate_training_record(record)))

    def test_image_tag_count_is_checked(self) -> None:
        record = valid_record()
        record["images"].append("another.ppm")
        record["media_sha256"].append("0" * 64)
        self.assertIn("image tag count 1 does not match images count 2", validate_training_record(record))

    def test_fenced_json_can_be_parsed(self) -> None:
        self.assertEqual(parse_assistant_json("```json\n{\"a\": 1}\n```"), {"a": 1})

    def test_unexpected_output_key_is_rejected(self) -> None:
        prediction = valid_prediction()
        prediction["description"] = "not in schema"
        self.assertTrue(any("unexpected output keys" in error for error in validate_prediction(prediction, 1)))

    def test_accept_cannot_hide_unknown_fields(self) -> None:
        prediction = valid_prediction()
        prediction["attributes"]["material"] = []
        prediction["evidence"] = [item for item in prediction["evidence"] if item["field"] != "attributes.material"]
        self.assertIn("accept is not allowed when a required field is unknown", validate_prediction(prediction, 1))


class ProvenanceTests(unittest.TestCase):
    def test_media_sha_is_checked(self) -> None:
        data_root = ROOT / "data"
        self.assertEqual(validate_media_files(valid_record(), data_root), [])

    def test_media_sha_mismatch_is_rejected(self) -> None:
        record = valid_record()
        record["media_sha256"] = ["0" * 64]
        self.assertIn("images[0] SHA-256 mismatch", validate_media_files(record, ROOT / "data"))

    def test_sha256_file_is_deterministic(self) -> None:
        path = ROOT / "data" / "sample" / "assets" / "demo_product.ppm"
        self.assertEqual(sha256_file(path), sha256_file(path))


class DataTests(unittest.TestCase):
    def test_group_split_is_stable(self) -> None:
        self.assertEqual(stable_group_split("product-1"), stable_group_split("product-1"))

    def test_split_salt_is_part_of_definition(self) -> None:
        outcomes = {stable_group_split("product-1", salt=f"salt-{index}") for index in range(20)}
        self.assertGreater(len(outcomes), 1)

    def test_group_leakage_is_rejected(self) -> None:
        with self.assertRaises(ValueError):
            assert_no_group_leakage([
                {"group_id": "same", "split": "train"},
                {"group_id": "same", "split": "eval"},
            ])

    def test_media_leakage_is_rejected(self) -> None:
        with self.assertRaises(ValueError):
            assert_no_media_leakage([
                {"media_sha256": ["a" * 64], "split": "train"},
                {"media_sha256": ["a" * 64], "split": "test"},
            ])

    def test_abo_conversion_requires_review(self) -> None:
        candidate = abo_candidate(
            {
                "item_id": "B001",
                "product_type": [{"value": "SOFA"}],
                "color": [{"language_tag": "en_US", "value": "Black"}],
                "material": [{"language_tag": "en_US", "value": "Wood"}],
            },
            ["B001.jpg"],
            snapshot_id="2026-07-29",
            license_id="archive-license-sha256",
        )
        self.assertTrue(candidate["review_required"])
        self.assertEqual(candidate["label_source"], "abo_listing_metadata_candidate_only")


class EvaluationTests(unittest.TestCase):
    def test_unsupported_attribute_is_counted(self) -> None:
        prediction = valid_prediction("review")
        result = evaluate_records(
            [{
                "sample_id": "1",
                "observable_fields": {
                    "product_type": "chair",
                    "attributes.color": ["black"],
                    "attributes.material": [],
                },
                "allowed_evidence_media": [0],
            }],
            [{"sample_id": "1", "prediction": prediction}],
        )
        self.assertGreater(result["unsupported_attribute_rate"], 0.0)

    def test_missing_prediction_is_counted(self) -> None:
        result = evaluate_records(
            [{"sample_id": "1", "observable_fields": {}, "allowed_evidence_media": [0]}],
            [],
        )
        self.assertEqual(result["missing_prediction_count"], 1)

    def test_review_decision_routes_to_human(self) -> None:
        self.assertEqual(route_prediction(valid_prediction("review"), 1).destination, "manual_review")

    def test_invalid_evidence_routes_to_reject(self) -> None:
        prediction = valid_prediction()
        prediction["evidence"][0]["media_index"] = 2
        self.assertEqual(route_prediction(prediction, 1).destination, "reject")

    def test_valid_accept_routes_to_accept(self) -> None:
        self.assertEqual(route_prediction(valid_prediction(), 1).destination, "accept")


class AdapterTests(unittest.TestCase):
    def test_model_revision_is_mandatory(self) -> None:
        with self.assertRaises(ValueError):
            Qwen3VLAdapter(InferenceConfig())


if __name__ == "__main__":
    unittest.main()
