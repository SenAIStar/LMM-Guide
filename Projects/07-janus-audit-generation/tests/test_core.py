from __future__ import annotations

import json
import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from janus_audit.contracts import AuditResult, Evidence
from janus_audit.datasets import read_jsonl, to_sft_conversation, validate_records
from janus_audit.evaluation import evaluate, macro_f1, relative_regression
from janus_audit.generation import build_generation_request
from janus_audit.grpo import group_relative_advantages
from janus_audit.policy import PolicyEngine
from janus_audit.provenance import sha256_file
from janus_audit.rewards import score_candidate, set_f1


POLICY = PolicyEngine.from_path(ROOT / "configs/policy.json")


def sample_payload() -> dict:
    row = read_jsonl(ROOT / "data/sample/audit_examples.jsonl")[0]
    return row["target"]


class CoreTests(unittest.TestCase):
    def test_sample_dataset_is_valid(self) -> None:
        records = read_jsonl(ROOT / "data/sample/audit_examples.jsonl")
        self.assertEqual(validate_records(records, ROOT), [])

    def test_sample_hash_is_reproducible(self) -> None:
        row = read_jsonl(ROOT / "data/sample/audit_examples.jsonl")[0]
        self.assertEqual(sha256_file(ROOT / row["media_path"]), row["media_sha256"])

    def test_contract_rejects_unknown_fields(self) -> None:
        payload = sample_payload()
        payload["unexpected"] = True
        with self.assertRaisesRegex(ValueError, "schema mismatch"):
            AuditResult.from_dict(payload)

    def test_contract_requires_evidence_for_risk(self) -> None:
        payload = sample_payload()
        payload["risk_labels"] = ["regulated_claim"]
        with self.assertRaisesRegex(ValueError, "grounded evidence"):
            AuditResult.from_dict(payload)

    def test_bbox_must_be_normalized(self) -> None:
        with self.assertRaisesRegex(ValueError, "normalized"):
            Evidence.from_dict({"kind": "bbox", "source": "image", "bbox": [0, 0, 2, 1]})

    def test_policy_allows_clean_generation(self) -> None:
        verdict = POLICY.evaluate(AuditResult.from_dict(sample_payload()))
        self.assertTrue(verdict.allowed)
        self.assertTrue(verdict.generation_allowed)

    def test_policy_blocks_unknown_version(self) -> None:
        payload = sample_payload()
        payload["policy_version"] = "future-policy"
        result = AuditResult.from_dict(payload)
        self.assertIn("unknown_policy_version", POLICY.evaluate(result).reasons)

    def test_generation_request_uses_only_approved_brief(self) -> None:
        request = build_generation_request(AuditResult.from_dict(sample_payload()), POLICY)
        self.assertIn("抽象配色海报", request.prompt)
        self.assertIn("个人敏感信息", request.prohibited_elements)

    def test_reward_is_one_for_identical_candidate(self) -> None:
        gold = AuditResult.from_dict(sample_payload())
        reward = score_candidate(sample_payload(), gold, POLICY)
        self.assertAlmostEqual(reward.total, 1.0)
        self.assertFalse(reward.hard_policy_violation)

    def test_invalid_schema_gets_hard_negative_reward(self) -> None:
        gold = AuditResult.from_dict(sample_payload())
        reward = score_candidate({"decision": "pass"}, gold, POLICY)
        self.assertEqual(reward.total, -1.0)
        self.assertTrue(reward.hard_policy_violation)

    def test_set_f1_handles_partial_match(self) -> None:
        self.assertAlmostEqual(set_f1({"a", "b"}, {"b", "c"}), 0.5)

    def test_grpo_advantages_are_group_normalized(self) -> None:
        batch = group_relative_advantages([1.0, 0.0, -1.0])
        self.assertAlmostEqual(sum(batch.advantages), 0.0)
        self.assertFalse(batch.zero_variance)

    def test_grpo_zero_variance_group_has_no_signal(self) -> None:
        batch = group_relative_advantages([0.5, 0.5])
        self.assertEqual(batch.advantages, (0.0, 0.0))
        self.assertTrue(batch.zero_variance)

    def test_sft_conversion_serializes_target_json(self) -> None:
        row = read_jsonl(ROOT / "data/sample/audit_examples.jsonl")[0]
        conversation = to_sft_conversation(row)
        target = json.loads(conversation["messages"][-1]["content"])
        self.assertEqual(target["asset_id"], row["asset_id"])

    def test_macro_f1_and_regression(self) -> None:
        self.assertEqual(macro_f1(["pass", "reject"], ["pass", "reject"]), 1.0)
        self.assertAlmostEqual(relative_regression(0.90, 0.87), 1 / 30)

    def test_evaluation_counts_invalid_json_as_violation(self) -> None:
        gold = [AuditResult.from_dict(sample_payload())]
        summary = evaluate(gold, [{"decision": "pass"}], POLICY)
        self.assertEqual(summary.json_valid_rate, 0.0)
        self.assertEqual(summary.policy_violation_rate, 1.0)


if __name__ == "__main__":
    unittest.main()
