"""모델별 임베딩 백엔드 래퍼.

핵심: 모델마다 요구하는 prefix 규칙이 다르다. 이걸 안 맞추면 비교가 무의미해진다.
  - KURE-v1, BGE-M3           : prefix 없음
  - multilingual-e5-small-ko  : "query: " / "passage: "
  - e5-large-instruct         : 쿼리에만 "Instruct: {task}\nQuery: {q}", 문서는 raw
"""

from __future__ import annotations

import gc
import time
from dataclasses import dataclass, field
from typing import Any

import numpy as np


@dataclass
class ModelSpec:
    key: str
    hf_id: str
    backend: str = "sentence-transformers"
    prefix_style: str = "none"  # none | e5 | e5_inst | instruct_gemma
    dim: int | None = None
    instruction: str = "Given a question in Korean, retrieve the passage that answers it"
    hybrid: dict = field(default_factory=dict)
    # SentenceTransformer(model_kwargs=...) 로 그대로 넘길 값.
    # 여기서 torch_dtype 을 주면 runtime.fp16 보다 우선한다 (gemma2 는 bfloat16 필수).
    model_kwargs: dict = field(default_factory=dict)

    @classmethod
    def from_config(cls, d: dict) -> "ModelSpec":
        known = {f for f in cls.__dataclass_fields__}
        return cls(**{k: v for k, v in d.items() if k in known})


def resolve_device(setting: str) -> str:
    if setting != "auto":
        return setting
    import torch

    return "cuda" if torch.cuda.is_available() else "cpu"


class BaseEncoder:
    """dense 임베딩을 내는 공통 인터페이스."""

    spec: ModelSpec

    def encode_passages(self, texts: list[str], batch_size: int) -> np.ndarray:
        raise NotImplementedError

    def encode_queries(self, texts: list[str], batch_size: int) -> np.ndarray:
        raise NotImplementedError

    def close(self) -> None:
        for attr in ("model", "_model"):
            if hasattr(self, attr):
                delattr(self, attr)
        gc.collect()
        try:
            import torch

            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except ImportError:
            pass

    # ── prefix 적용 ────────────────────────────────────────────
    def _fmt_passages(self, texts: list[str]) -> list[str]:
        if self.spec.prefix_style == "e5":
            return [f"passage: {t}" for t in texts]
        return list(texts)  # none, e5_inst 모두 문서에는 prefix 없음

    def _fmt_queries(self, texts: list[str]) -> list[str]:
        if self.spec.prefix_style == "e5":
            return [f"query: {t}" for t in texts]
        if self.spec.prefix_style == "e5_inst":
            return [f"Instruct: {self.spec.instruction}\nQuery: {t}" for t in texts]
        if self.spec.prefix_style == "instruct_gemma":
            # bge-multilingual-gemma2 형식. e5_inst 와 태그가 달라 섞으면 성능이 떨어진다.
            return [f"<instruct>{self.spec.instruction}\n<query>{t}" for t in texts]
        return list(texts)


class STEncoder(BaseEncoder):
    """sentence-transformers 기반 dense 인코더."""

    def __init__(self, spec: ModelSpec, device: str, max_seq_length: int,
                 fp16: bool, hf_token: str | None = None):
        from sentence_transformers import SentenceTransformer

        self.spec = spec
        kwargs: dict[str, Any] = {"device": device, "trust_remote_code": True}
        if hf_token:
            kwargs["token"] = hf_token

        model_kwargs: dict[str, Any] = dict(spec.model_kwargs)
        if fp16 and device.startswith("cuda"):
            # bfloat16 로 배포된 대형 모델(gemma2, jina-v4)은 config 에서 덮어쓸 수 있게 둔다
            model_kwargs.setdefault("torch_dtype", "float16")
        if model_kwargs:
            kwargs["model_kwargs"] = model_kwargs

        self.model = SentenceTransformer(spec.hf_id, **kwargs)
        if max_seq_length:
            self.model.max_seq_length = min(
                max_seq_length, self.model.max_seq_length or max_seq_length
            )

    def _encode(self, texts: list[str], batch_size: int) -> np.ndarray:
        vecs = self.model.encode(
            texts,
            batch_size=batch_size,
            normalize_embeddings=True,   # 코사인 유사도 = 내적
            convert_to_numpy=True,
            show_progress_bar=False,
        )
        return np.asarray(vecs, dtype=np.float32)

    def encode_passages(self, texts, batch_size):
        return self._encode(self._fmt_passages(texts), batch_size)

    def encode_queries(self, texts, batch_size):
        return self._encode(self._fmt_queries(texts), batch_size)


class BGEM3Encoder(BaseEncoder):
    """BGE-M3 전용. dense 뿐 아니라 sparse(lexical weights)까지 뽑아
    하이브리드 검색 성능도 함께 평가한다.

    FlagEmbedding 설치가 실패하면 STEncoder 로 폴백(dense 단독)한다.
    """

    def __init__(self, spec: ModelSpec, device: str, max_seq_length: int,
                 fp16: bool, hf_token: str | None = None):
        from FlagEmbedding import BGEM3FlagModel

        self.spec = spec
        self.max_length = max_seq_length or 1024
        self.model = BGEM3FlagModel(
            spec.hf_id,
            use_fp16=fp16 and device.startswith("cuda"),
            devices=device,
        )
        self._want_sparse = bool(spec.hybrid.get("enabled", False))
        self.passage_sparse: list[dict] | None = None
        self.query_sparse: list[dict] | None = None

    def _encode(self, texts: list[str], batch_size: int, store: str) -> np.ndarray:
        out = self.model.encode(
            texts,
            batch_size=batch_size,
            max_length=self.max_length,
            return_dense=True,
            return_sparse=self._want_sparse,
            return_colbert_vecs=False,
        )
        if self._want_sparse:
            setattr(self, f"{store}_sparse", out["lexical_weights"])
        dense = np.asarray(out["dense_vecs"], dtype=np.float32)
        # BGE-M3 dense 는 이미 정규화되어 있지만, 안전하게 한 번 더.
        norms = np.linalg.norm(dense, axis=1, keepdims=True)
        return dense / np.clip(norms, 1e-12, None)

    def encode_passages(self, texts, batch_size):
        return self._encode(list(texts), batch_size, "passage")

    def encode_queries(self, texts, batch_size):
        return self._encode(list(texts), batch_size, "query")

    def sparse_score_matrix(self) -> np.ndarray | None:
        """(n_queries, n_passages) 어휘 매칭 점수 행렬."""
        if not self._want_sparse or self.query_sparse is None or self.passage_sparse is None:
            return None
        scores = self.model.compute_lexical_matching_score(
            self.query_sparse, self.passage_sparse
        )
        return np.asarray(scores, dtype=np.float32).reshape(
            len(self.query_sparse), len(self.passage_sparse)
        )


def load_encoder(spec: ModelSpec, runtime: dict) -> BaseEncoder:
    device = resolve_device(runtime.get("device", "auto"))
    max_seq = runtime.get("max_seq_length", 1024)
    fp16 = runtime.get("fp16", True)
    token = runtime.get("hf_token")

    t0 = time.time()
    if spec.backend == "bge-m3":
        try:
            enc = BGEM3Encoder(spec, device, max_seq, fp16, token)
        except ImportError:
            print("  [경고] FlagEmbedding 미설치 → BGE-M3 를 dense 단독으로만 평가합니다.")
            enc = STEncoder(spec, device, max_seq, fp16, token)
    else:
        enc = STEncoder(spec, device, max_seq, fp16, token)
    print(f"  모델 로드 완료 ({time.time() - t0:.1f}s, device={device})")
    return enc
