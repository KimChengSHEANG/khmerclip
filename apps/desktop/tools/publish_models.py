"""Upload exported TrOCR ONNX files to the public HF repo and pin them in trocr_manifest.json.

    .venv/bin/huggingface-cli login
    .venv/bin/python tools/publish_models.py exported/khmer --version khmer-v1
"""

import argparse
import json
from pathlib import Path

from huggingface_hub import HfApi

from khmerclip.ocr.trocr_models import MANIFEST_PATH, MODEL_FILES, build_manifest

REPO = "kcsheang/khmerclip-trocr-khmer-onnx"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("model_dir", type=Path)
    parser.add_argument("--version", required=True)
    parser.add_argument("--repo", default=REPO)
    args = parser.parse_args()

    api = HfApi()
    api.create_repo(args.repo, repo_type="model", private=False, exist_ok=True)
    commit = api.upload_folder(repo_id=args.repo, folder_path=str(args.model_dir),
                               allow_patterns=list(MODEL_FILES),
                               commit_message=f"KhmerClip TrOCR ONNX {args.version}")
    # Pin by commit revision so later uploads never change what an installed app downloads.
    base_url = f"https://huggingface.co/{args.repo}/resolve/{commit.oid}"
    data = build_manifest(args.model_dir, base_url, args.version)
    MANIFEST_PATH.write_text(json.dumps(data, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print("wrote", MANIFEST_PATH)


if __name__ == "__main__":
    main()
