import json
from pathlib import Path
import sys
import unittest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from lmm_core import (  # noqa: E402
    FlightSafetyGate,
    exact_match_rate,
    grounded_rate,
    load_project_config,
    product_audit_reward,
    recall_at_k,
    validate_conversation_record,
)


class RecordTests(unittest.TestCase):
    def test_valid_multimodal_record(self) -> None:
        record = {
            "id": "1",
            "media": ["a.jpg"],
            "messages": [
                {"role": "user", "content": "question"},
                {"role": "assistant", "content": "answer"},
            ],
        }
        self.assertEqual(validate_conversation_record(record), [])

    def test_invalid_record_is_explained(self) -> None:
        errors = validate_conversation_record({"id": "", "media": [], "messages": []})
        self.assertGreaterEqual(len(errors), 3)


class MetricTests(unittest.TestCase):
    def test_exact_match_normalizes_space_and_case(self) -> None:
        self.assertEqual(exact_match_rate([" A  B "], ["a b"]), 1.0)

    def test_recall_at_k(self) -> None:
        self.assertEqual(recall_at_k([["b", "a"]], [{"a"}], 2), 1.0)

    def test_grounded_rate_requires_nonempty_citations(self) -> None:
        self.assertEqual(grounded_rate([{"a"}, set()], [{"a"}, {"b"}]), 0.5)


class RewardTests(unittest.TestCase):
    def test_exact_audit_output_scores_one(self) -> None:
        output = {"decision": "pass", "risk_type": "none", "evidence": ["box"], "confidence": 0.8}
        self.assertEqual(product_audit_reward(output, {"decision": "pass", "risk_type": "none"}), 1.0)


class SafetyTests(unittest.TestCase):
    def test_safe_command_is_allowed(self) -> None:
        result = FlightSafetyGate().evaluate(
            {"action": "takeoff", "altitude_m": 10, "human_approved": True},
            {"battery_pct": 80, "gps_fix": True},
        )
        self.assertTrue(result.allowed)

    def test_model_cannot_bypass_policy(self) -> None:
        result = FlightSafetyGate().evaluate(
            {"action": "goto", "altitude_m": 100, "human_approved": False},
            {"battery_pct": 10, "gps_fix": False},
        )
        self.assertFalse(result.allowed)
        self.assertGreaterEqual(len(result.reasons), 3)


class ConfigTests(unittest.TestCase):
    def test_all_project_configs(self) -> None:
        paths = sorted(ROOT.glob("[0-9][0-9]-*/project.json"))
        self.assertEqual(len(paths), 7)
        for path in paths:
            with self.subTest(path=path):
                config = load_project_config(path)
                json.dumps(config)


if __name__ == "__main__":
    unittest.main()

