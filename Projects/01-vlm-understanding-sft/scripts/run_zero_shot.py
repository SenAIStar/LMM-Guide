import argparse
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from vlm_product.qwen3vl_adapter import InferenceConfig, Qwen3VLAdapter  # noqa: E402


DEFAULT_PROMPT = """只根据图片提取商品类型、颜色、材质和可见文字。
看不清的字段使用空数组，不要根据常识补写。
严格按 vlm_product.schema.v1 输出 JSON，不要输出 Markdown。"""


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("images", nargs="+", type=Path)
    parser.add_argument("--model-id", default="Qwen/Qwen3-VL-4B-Instruct")
    parser.add_argument("--revision", required=True)
    parser.add_argument("--prompt", default=DEFAULT_PROMPT)
    parser.add_argument("--max-new-tokens", type=int, default=512)
    args = parser.parse_args()
    missing = [str(path) for path in args.images if not path.is_file()]
    if missing:
        parser.error(f"images do not exist: {missing}")
    adapter = Qwen3VLAdapter(
        InferenceConfig(
            model_id=args.model_id,
            revision=args.revision,
            max_new_tokens=args.max_new_tokens,
        )
    )
    print(adapter.generate(args.images, args.prompt))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
