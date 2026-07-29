from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from lmm_core import load_project_config  # noqa: E402


def main() -> int:
    paths = sorted(ROOT.glob("[0-9][0-9]-*/project.json"))
    if len(paths) != 7:
        raise SystemExit(f"expected 7 project configs, found {len(paths)}")
    for path in paths:
        config = load_project_config(path)
        print(f"OK {config['project_id']}: {path.parent.name}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

