from __future__ import annotations

import os
import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))


def validate_model(path: Path) -> None:
    required = [path / "config.json", path / "tokenizer.json"]
    weights = [path / "model.safetensors", path / "pytorch_model.bin"]
    if not all(item.is_file() and item.stat().st_size > 0 for item in required) or not any(
        item.is_file() and item.stat().st_size > 0 for item in weights
    ):
        raise RuntimeError(f"模型快照文件缺失或为空：{path}")


def main() -> int:
    target = ROOT / "models" / "bge-reranker-base"
    target.parent.mkdir(parents=True, exist_ok=True)

    from huggingface_hub import snapshot_download

    # Reuse a valid existing fixed project location first.
    if target.is_dir():
        validate_model(target)
        print(f"Model already installed: {target}")
        return 0

    # Reuse the user's shared HF cache and copy only when the project-local
    # installation is missing. This avoids a second network transfer.
    try:
        cached = Path(snapshot_download("BAAI/bge-reranker-base", local_files_only=True))
        validate_model(cached)
        shutil.copytree(cached, target)
        validate_model(target)
        print(f"Installed from Hugging Face cache: {target}")
        return 0
    except Exception as cache_error:
        print(f"No complete Hugging Face cache found ({type(cache_error).__name__}); trying hf-mirror.com.")

    try:
        os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
        mirrored = Path(snapshot_download("BAAI/bge-reranker-base"))
        validate_model(mirrored)
        shutil.copytree(mirrored, target)
        validate_model(target)
        print(f"Downloaded via https://hf-mirror.com: {target}")
        return 0
    except Exception as mirror_error:
        print(f"Hugging Face mirror failed ({type(mirror_error).__name__}); trying ModelScope.")

    try:
        from modelscope.hub.snapshot_download import snapshot_download as modelscope_download

        downloaded = Path(modelscope_download("AI-ModelScope/bge-reranker-base", cache_dir=str(target.parent)))
        validate_model(downloaded)
        if downloaded.resolve() != target.resolve():
            shutil.copytree(downloaded, target)
        validate_model(target)
        print(f"Downloaded via ModelScope: {target}")
        return 0
    except Exception as modelscope_error:
        print("Unable to install the reranker automatically.")
        print(f"Model: BAAI/bge-reranker-base (CrossEncoder; safetensors/PyTorch weights)")
        print("Sources: https://huggingface.co/BAAI/bge-reranker-base and https://hf-mirror.com/BAAI/bge-reranker-base")
        print("ModelScope: AI-ModelScope/bge-reranker-base")
        print(f"Recommended path: {target}")
        print("Manual install: download the complete repository snapshot (config.json, tokenizer files, model.safetensors) into that folder, then rerun this script.")
        print(f"Errors: mirror={type(mirror_error).__name__}, modelscope={type(modelscope_error).__name__}")
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
