import argparse
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from lmm_core import (  # noqa: E402
    FlightSafetyGate,
    grounded_rate,
    product_audit_reward,
    recall_at_k,
    validate_conversation_record,
)


def vlm_sft() -> dict:
    record = {
        "id": "abo-demo-001",
        "media": ["images/001.jpg"],
        "messages": [
            {"role": "user", "content": "识别商品材质和颜色"},
            {"role": "assistant", "content": '{"material":"wood","color":"brown"}'},
        ],
    }
    return {"validation_errors": validate_conversation_record(record)}


def product_audit() -> dict:
    prediction = {"decision": "reject", "risk_type": "logo", "evidence": ["box_0"], "confidence": 0.91}
    reference = {"decision": "reject", "risk_type": "logo"}
    return {"reward": product_audit_reward(prediction, reference)}


def product_rag() -> dict:
    recall = recall_at_k([["p2", "p1", "p3"]], [{"p1"}], 2)
    grounded = grounded_rate([{"p1"}], [{"p1", "p3"}])
    return {"recall_at_2": recall, "grounded_rate": grounded}


def drone_agent() -> dict:
    gate = FlightSafetyGate()
    accepted = gate.evaluate(
        {"action": "takeoff", "altitude_m": 10, "human_approved": True},
        {"battery_pct": 80, "gps_fix": True},
    )
    rejected = gate.evaluate(
        {"action": "goto", "altitude_m": 80, "human_approved": False},
        {"battery_pct": 18, "gps_fix": False},
    )
    return {"accepted": accepted.allowed, "rejected_reasons": rejected.reasons}


def generation(project: str) -> dict:
    return {
        "project": project,
        "request_contract": "validated",
        "note": "model execution is intentionally disabled in the dependency-free demo",
    }


DEMOS = {
    "vlm-sft": vlm_sft,
    "product-audit": product_audit,
    "product-rag": product_rag,
    "drone-agent": drone_agent,
    "controlnet-lora": lambda: generation("controlnet-lora"),
    "image-service": lambda: generation("image-service"),
    "janus": lambda: generation("janus"),
}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--project", choices=["all", *DEMOS], default="all")
    args = parser.parse_args()
    names = DEMOS if args.project == "all" else {args.project: DEMOS[args.project]}
    for name, demo in names.items():
        print(name, demo())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

