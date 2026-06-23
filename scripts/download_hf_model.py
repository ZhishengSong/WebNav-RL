from __future__ import annotations

import argparse
import json
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(description="Download a Hugging Face model snapshot for server runs.")
    parser.add_argument("--repo-id", default="Qwen/Qwen2.5-0.5B-Instruct")
    parser.add_argument("--local-dir", default="models/qwen2.5-0.5b-instruct")
    parser.add_argument(
        "--allow-pattern",
        action="append",
        default=["*.json", "*.jinja", "*.txt", "*.safetensors"],
        help="Allowed file pattern. Can be repeated.",
    )
    args = parser.parse_args()

    try:
        from huggingface_hub import snapshot_download
    except ImportError as exc:
        raise RuntimeError("Install huggingface_hub first, or install requirements-model.txt.") from exc

    path = snapshot_download(
        repo_id=args.repo_id,
        local_dir=args.local_dir,
        allow_patterns=args.allow_pattern,
    )
    result = {
        "repo_id": args.repo_id,
        "local_dir": str(Path(path)),
        "allow_patterns": args.allow_pattern,
    }
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
