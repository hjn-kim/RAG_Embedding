# 한국어 임베딩 모델 검색 성능 비교

동일한 `.docx` 1개 + `.pdf` 1개를 대상으로 4개 임베딩 모델의 **검색(Retrieval) 정확도**를 비교합니다.

| 모델 | HF ID | 차원 | prefix 규칙 |
|---|---|---|---|
| KURE-v1 | `nlpai-lab/KURE-v1` | 1024 | 없음 |
| multilingual-e5-large-instruct | `intfloat/multilingual-e5-large-instruct` | 1024 | 쿼리에 `Instruct:...\nQuery:` |
| multilingual-e5-small-ko | `dragonkue/multilingual-e5-small-ko` | 384 | `query: ` / `passage: ` |
| BGE-M3 | `BAAI/bge-m3` | 1024 | 없음 (dense + sparse 하이브리드도 함께 측정) |

## 평가 방식

임베딩 모델은 답변을 생성하지 않고 **관련 청크를 찾아오는** 역할만 합니다.
따라서 "성능"은 생성 품질이 아니라 검색 정확도로 측정합니다.

```
문서 → 공통 청킹(모든 모델 동일) → 모델별 임베딩 → 질문과 코사인 유사도 → 랭킹 → 지표
```

정답 라벨은 `questions.json` 의 `must_include` / `any_include` 문구를 포함한 청크를
자동으로 gold 로 잡습니다. 청킹 설정을 바꿔도 라벨을 다시 안 만들어도 됩니다.

측정 지표:

| 지표 | 의미 |
|---|---|
| **Hit@1 / Hit@k** | 상위 k개 안에 정답이 하나라도 있었는가. RAG 체감 성능에 가장 직결 |
| **Recall@k** | 전체 정답 중 상위 k개에 들어온 비율 |
| **MRR@10** | 첫 정답의 순위 역수 (1위=1.0, 5위=0.2) |
| **nDCG@10** | 정답을 얼마나 위쪽에 몰아놨는가 (최종 정렬 기준) |
| chunks_per_s / query_ms | 인덱싱·검색 속도 |
| index_MB | 벡터 인덱스 용량 (차원에 비례) |

## 실행 순서

### 1. 준비 (로컬)

```bash
# data/ 에 비교할 문서 2개를 넣는다
data/내문서.docx
data/내문서.pdf
```

### 2. RunPod 세팅

RunPod PyTorch 템플릿(CUDA 12.x, GPU 16GB 이상 권장 — BGE-M3 fp16 기준 약 5GB)에서:

```bash
git clone <your-repo-url> && cd <repo>

# 모델 캐시를 네트워크 볼륨에 두면 파드 재시작해도 재다운로드가 없다
export HF_HOME=/workspace/hf_cache

pip install -r requirements.txt
python -m src.download_models     # 모델 4개 미리 받기 (합계 약 6~7GB)
```

> torch 버전 충돌이 나면 `requirements.txt` 의 `torch` 줄을 주석 처리하세요
> (RunPod 템플릿에 이미 설치돼 있습니다).

### 3. 청크 확인 후 질문 작성 ← **여기가 제일 중요합니다**

```bash
python -m src.inspect_chunks --limit 50
python -m src.inspect_chunks --grep 계약   # 특정 키워드 주변 확인
```

`results/chunks_preview.txt` 를 보면서 `questions/questions.json` 을 채웁니다.

```json
{
  "id": "docx-01",
  "question": "계약 해지는 며칠 전에 알려야 하나요?",
  "source": "내문서.docx",
  "must_include": ["30일 전 서면으로 통보"]
}
```

질문 작성 규칙:

- **원문 문장을 그대로 베끼지 마세요.** 베끼면 어휘만 겹쳐서 모든 모델이 다 맞히고
  변별력이 사라집니다. 실제 사용자가 쓸 법한 구어체·동의어로 바꿔 쓰세요.
- `must_include` 는 **원문에서 정확히 복사**하세요. 공백은 무시되지만 글자는 일치해야 합니다.
- 최소 20개, 문서당 10개 이상 권장. 질문이 5개면 한 문제 차이로 순위가 뒤집힙니다.
- 실행 시 `[경고] 정답 청크를 못 찾은 질문` 이 뜨면 `must_include` 문구가 원문과 다른 것입니다.

### 4. 벤치마크 실행

```bash
python -m src.run_benchmark

# 특정 모델만
python -m src.run_benchmark --models kure-v1 bge-m3
```

결과:

| 파일 | 내용 |
|---|---|
| `results/summary.csv` `summary.md` | 모델별 지표 요약표 |
| `results/details.json` | 질문별 상위 5개 검색 결과 — **왜 틀렸는지 확인용** |
| `results/chunks.json` | 사용된 청크 전체 |

## 튜닝 포인트 (`config.yaml`)

| 항목 | 설명 |
|---|---|
| `chunking.chunk_size` | 300 / 500 / 800 으로 바꿔가며 돌려보세요. 최적값은 문서 성격에 따라 다르고, **모델 순위 자체가 바뀌기도 합니다** |
| `retrieval.scope` | `all` = 두 문서를 한 인덱스에서 검색(어려움, 실전에 가까움) / `own_doc` = 문서 내부만 |
| `runtime.batch_size` | CUDA OOM 이면 8 → 4 로 |
| `runtime.max_seq_length` | BGE-M3 장문 성능을 보려면 4096~8192 로 (VRAM 많이 씀) |
| `models[].hybrid` | BGE-M3 의 sparse 가중치. dense 단독 대비 얼마나 오르는지 비교됨 |

## 결과 해석 시 주의

- 질문 20~30개는 통계적으로 작은 표본입니다. 0.02 차이는 노이즈로 보세요.
- 차원이 큰 모델이 항상 이기지 않습니다. `index_MB` 와 `chunks_per_s` 를 함께 보고
  **정확도 대비 비용**으로 판단하세요. e5-small-ko 가 nDCG 0.03 낮은데 3배 빠르면
  서비스에 따라 그쪽이 정답입니다.
- 실제 RAG 품질은 리랭커(BGE-reranker 등)를 얹으면 또 달라집니다. 이 벤치마크는
  1차 검색기(retriever) 성능만 측정합니다.

## 구조

```
config.yaml              설정 (모델 목록, 청킹, 지표)
questions/questions.json 질문 + 정답 라벨
data/                    비교 대상 문서
src/
  loaders.py             docx/pdf 텍스트 추출
  metadata.py            문서종류·기관·작성일·제목·섹션 추출 (규칙 기반, 기록 전용)
  chunker.py             공통 청킹
  gold.py                질문 로드 + 정답 청크 자동 라벨링
  models.py              모델별 prefix 규칙 + 인코더 래퍼
  metrics.py             Recall/Hit/MRR/nDCG
  run_benchmark.py       메인 실행
  inspect_chunks.py      청크 확인 헬퍼
  download_models.py     HF 사전 다운로드
```
