from __future__ import annotations

import argparse
import json
from pathlib import Path
import subprocess


SCRIPT_BY_STAGE = {
    "lora": Path("examples/text_to_image/train_text_to_image_lora_sdxl.py"),
    "controlnet": Path("examples/controlnet/train_controlnet_sdxl.py"),
}


def render(stage: str, config: dict[str, object], diffusers_root: Path) -> list[str]:
    script = diffusers_root / SCRIPT_BY_STAGE[stage]
    if not script.is_file():
        raise FileNotFoundError(f"official Diffusers script not found: {script}")
    command = ["accelerate", "launch", str(script)]
    ignored = {"notes", "diffusers_commit", "stage"}
    for key, value in config.items():
        if key in ignored or value is None or value is False:
            continue
        flag = f"--{key}"
        if value is True:
            command.append(flag)
        elif isinstance(value, list):
            command.extend([flag, *[str(item) for item in value]])
        else:
            command.extend([flag, str(value)])
    return command


def main() -> int:
    parser = argparse.ArgumentParser(description="Render a pinned official Diffusers training command.")
    parser.add_argument("--stage", choices=sorted(SCRIPT_BY_STAGE), required=True)
    parser.add_argument("--config", type=Path)
    parser.add_argument("--diffusers-root", type=Path, required=True)
    args = parser.parse_args()
    config_path = args.config or Path("configs") / f"{args.stage}_train.json"
    config = json.loads(config_path.read_text(encoding="utf-8"))
    print(f"Required diffusers commit: {config['diffusers_commit']}")
    print(subprocess.list2cmdline(render(args.stage, config, args.diffusers_root)))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
