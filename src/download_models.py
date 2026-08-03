"""모델 4개를 Hugging Face 에서 미리 내려받는다.

RunPod 에서 벤치마크 도중 다운로드가 겹치면 시간 측정이 오염되므로 먼저 받아두는 게 좋다.
캐시 위치는 HF_HOME 환경변수를 따른다. 네트워크 볼륨에 두면 재시작해도 재다운로드가 없다:

    export HF_HOME=/workspace/hf_cache
    python -m src.download_models
"""

from __future__ import annotations

import os
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parent.parent


def main() -> None:
    from huggingface_hub import snapshot_download

    cfg = yaml.safe_load((ROOT / "config.yaml").read_text(encoding="utf-8"))
    token = cfg["runtime"].get("hf_token") or os.environ.get("HF_TOKEN")

    print(f"HF_HOME = {os.environ.get('HF_HOME', '(기본값 ~/.cache/huggingface)')}\n")
    for m in cfg["models"]:
        print(f"내려받는 중: {m['hf_id']}")
        path = snapshot_download(
            repo_id=m["hf_id"],
            token=token,
            # onnx / openvino 등 안 쓰는 대용량 사본 제외 (BGE-M3 는 특히 큼)
            ignore_patterns=["*.onnx", "onnx/*", "openvino/*", "*.msgpack", "*.h5"],
        )
        print(f"  → {path}\n")

    print("완료. 이제 `python -m src.run_benchmark` 를 실행하세요.")


if __name__ == "__main__":
    main()
