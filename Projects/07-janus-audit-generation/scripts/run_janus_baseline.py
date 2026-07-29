from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from janus_audit.janus_adapter import JanusBaseline, JanusConfig


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--revision", required=True, help="Reviewed Hugging Face commit hash")
    parser.add_argument("--image")
    parser.add_argument("--instruction")
    parser.add_argument("--generate-prompt")
    parser.add_argument("--output", default="artifacts/janus_generation.png")
    args = parser.parse_args()
    backend = JanusBaseline(JanusConfig(revision=args.revision))
    if args.image and args.instruction:
        print(backend.understand([ROOT / args.image], args.instruction))
        return 0
    if args.generate_prompt:
        path = backend.generate(args.generate_prompt, ROOT / args.output)
        print(json.dumps({"output": str(path)}, ensure_ascii=False))
        return 0
    raise SystemExit("provide --image with --instruction, or --generate-prompt")


if __name__ == "__main__":
    raise SystemExit(main())

