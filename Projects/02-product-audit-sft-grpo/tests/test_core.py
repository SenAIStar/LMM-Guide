from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from product_audit.contracts import ContractError, parse_audit_output, validate_audit_output
from product_audit.datasets import assert_no_group_leakage
from product_audit.evaluation import expected_calibration_error
from product_audit.grpo import group_relative_advantages
from product_audit.policy import evaluate_hard_policy
from product_audit.rewards import score_completion
from product_audit.service import review_product


POLICY = {
    "schema_version": "audit-policy.v1",
    "policy_version": "test.v1",
    "effective_at": "2026-01-01T00:00:00Z",
    "rules": [
        {
            "id": "R-CONTACT-001",
            "risk_code": "CONTACT_DIVERSION",
            "severity": "high",
            "decision": "reject",
            "field": "ocr_tokens",
            "operator": "contains_any",
            "value": ["微信"],
        }
    ],
}


def output(
    decision: str = "pass",
    risk_codes: list[str] | None = None,
    evidence: list[dict] | None = None,
    policy_refs: list[str] | None = None,
) -> dict:
    return {
        "schema_version": "audit-output.v1",
        "decision": decision,
        "risk_codes": risk_codes or [],
        "evidence": evidence or [],
        "policy_refs": policy_refs or [],
        "explanation": "测试结论。",
    }


class ContractTests(unittest.TestCase):
    def test_strict_valid_output(self) -> None:
        parsed = validate_audit_output(output(), image_count=1)
        self.assertEqual(parsed["decision"], "pass")

    def test_trailing_text_is_rejected(self) -> None:
        text = json.dumps(output(), ensure_ascii=False) + " extra"
        with self.assertRaises(ContractError):
            parse_audit_output(text, image_count=1)

    def test_out_of_range_media_is_rejected(self) -> None:
        value = output(
            "reject",
            ["CONTACT_DIVERSION"],
            [
                {
                    "risk_code": "CONTACT_DIVERSION",
                    "media_index": 1,
                    "support": "image_level",
                    "bbox": None,
                    "policy_rule_id": "R-CONTACT-001",
                }
            ],
            ["R-CONTACT-001"],
        )
        with self.assertRaises(ContractError):
            validate_audit_output(value, image_count=1)


class PolicyAndRewardTests(unittest.TestCase):
    def test_hard_rule_rejects_contact(self) -> None:
        result = evaluate_hard_policy({"ocr_tokens": ["详情加微信"]}, POLICY)
        self.assertEqual(result.decision, "reject")
        self.assertEqual(result.hits[0].rule_id, "R-CONTACT-001")

    def test_reward_gates_policy_violation(self) -> None:
        context = {"ocr_tokens": ["详情加微信"], "image_count": 1}
        result = score_completion(json.dumps(output(), ensure_ascii=False), output(), POLICY, context)
        self.assertTrue(result.gated)
        self.assertEqual(result.total, -1.0)

    def test_reward_accepts_correct_reject(self) -> None:
        evidence = [
            {
                "risk_code": "CONTACT_DIVERSION",
                "media_index": 0,
                "support": "region",
                "bbox": [0.1, 0.1, 0.9, 0.9],
                "policy_rule_id": "R-CONTACT-001",
            }
        ]
        gold = output("reject", ["CONTACT_DIVERSION"], evidence, ["R-CONTACT-001"])
        result = score_completion(
            json.dumps(gold, ensure_ascii=False), gold, POLICY, {"ocr_tokens": ["微信"], "image_count": 1}
        )
        self.assertFalse(result.gated)
        self.assertAlmostEqual(result.total, 1.0)


class GrpoAndEvaluationTests(unittest.TestCase):
    def test_zero_variance_group_has_zero_advantage(self) -> None:
        self.assertEqual(group_relative_advantages([0.5, 0.5], ["a", "a"]), [0.0, 0.0])

    def test_nonzero_group_is_centered(self) -> None:
        values = group_relative_advantages([0.0, 1.0], ["a", "a"])
        self.assertAlmostEqual(sum(values), 0.0)
        self.assertLess(values[0], values[1])

    def test_ece_uses_external_confidence(self) -> None:
        self.assertAlmostEqual(expected_calibration_error([0.9, 0.8], [True, False], bins=2), 0.35)

    def test_group_leakage_is_rejected(self) -> None:
        records = [
            {"sample_id": "a", "product_id": "same"},
            {"sample_id": "b", "product_id": "same"},
        ]
        with self.assertRaises(ValueError):
            assert_no_group_leakage(records, {"a": "train", "b": "test"})


class ServiceTests(unittest.TestCase):
    def test_hard_rule_short_circuits_model(self) -> None:
        calls = []

        def generate(_: dict) -> str:
            calls.append(True)
            return json.dumps(output(), ensure_ascii=False)

        result = review_product(
            {"ocr_tokens": ["微信"], "images": ["a.jpg"]},
            POLICY,
            generate,
            lambda _request, _prediction: 0.99,
        )
        self.assertEqual(result.action, "reject")
        self.assertEqual(calls, [])

    def test_invalid_model_output_goes_to_review(self) -> None:
        result = review_product(
            {"ocr_tokens": [], "images": ["a.jpg"]},
            POLICY,
            lambda _: "not-json",
            lambda _request, _prediction: 0.99,
        )
        self.assertEqual(result.action, "manual_review")

    def test_low_external_confidence_goes_to_review(self) -> None:
        result = review_product(
            {"ocr_tokens": [], "images": ["a.jpg"]},
            POLICY,
            lambda _: json.dumps(output(), ensure_ascii=False),
            lambda _request, _prediction: 0.60,
        )
        self.assertEqual(result.action, "manual_review")


if __name__ == "__main__":
    unittest.main()
