"""One-command deploy of this Space to the Hugging Face Hub.

Prerequisites:
    pip install huggingface_hub
    export HF_TOKEN=hf_...          # write token: https://huggingface.co/settings/tokens
    export OPENAI_API_KEY=sk-...    # optional: also set as the Space secret

Usage:
    python space/deploy.py <hf-username> [space-name]
"""
from __future__ import annotations

import os
import sys
from pathlib import Path


def main() -> int:
    if len(sys.argv) < 2:
        print("usage: python space/deploy.py <hf-username> [space-name]", file=sys.stderr)
        return 2
    user = sys.argv[1]
    name = sys.argv[2] if len(sys.argv) > 2 else "ask-santosh"
    token = os.environ.get("HF_TOKEN")
    if not token:
        print("set HF_TOKEN (a write token from huggingface.co/settings/tokens)", file=sys.stderr)
        return 2

    from huggingface_hub import HfApi  # imported here so --help works without the dep

    repo_id = f"{user}/{name}"
    api = HfApi(token=token)
    api.create_repo(repo_id, repo_type="space", space_sdk="gradio", exist_ok=True)
    api.upload_folder(
        repo_id=repo_id,
        repo_type="space",
        folder_path=str(Path(__file__).parent),
        ignore_patterns=["deploy.py", "DEPLOY.md", "__pycache__/*"],
    )
    if os.environ.get("OPENAI_API_KEY"):
        api.add_space_secret(repo_id, "OPENAI_API_KEY", os.environ["OPENAI_API_KEY"])
        print("set OPENAI_API_KEY secret")

    print(f"deployed: https://huggingface.co/spaces/{repo_id}")
    print("the live app URL is shown on that page (…​.hf.space)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
