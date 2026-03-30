# RevLM — VLM 기반 Re-ID Snowball Loop 사용 가이드

## 1. 개요

RevLM은 **VLM(Vision-Language Model)을 활용한 Person Re-Identification Snowball Loop**를 구현한 프로젝트입니다.

### 왜 3단계인가?

기존 ReID 모델(임베딩 유사도)만으로는 외관 변화가 큰 케이스를 처리하기 어렵습니다. VLM을 추가하면 시각적 추론 능력으로 판별 정확도를 높일 수 있고, 사람의 피드백(HITL)으로 도메인 특화 LoRA를 학습하면 정확도가 반복적으로 향상됩니다.

```
┌──────────────────────────────────────────────────────────────────────┐
│                    Accuracy Snowball Loop                            │
│                                                                      │
│  Stage 1               Stage 2                  Stage 3             │
│  ReID backbone         VLM verifier             VLM + HITL LoRA     │
│  (코사인 유사도)  ──▶   (Qwen3-VL-8B)  ──▶      (fine-tuned)        │
│  ~73%                  ~83%                     ~89%+               │
│                                                    ▲                 │
│                        낮은 confidence 예측        │                 │
│                              │                    │                 │
│                              ▼                    │                 │
│                        HITL 큐 → 사람 레이블링 ──▶ LoRA 학습         │
└──────────────────────────────────────────────────────────────────────┘
```

| 단계 | 방법 | 핵심 특징 |
|------|------|----------|
| Stage 1 | ReID 임베딩 코사인 유사도 | 빠름, 도메인 무관 |
| Stage 2 | VLM 시각적 추론 | 외관 묘사 기반, zero-shot |
| Stage 3 | HITL 데이터로 LoRA 학습 | 도메인 특화, 반복 개선 |

---

## 2. VLM 관련 파일 구조

```
revlm/
├── src/
│   ├── models/
│   │   ├── reid.py               # ReID 임베딩 백엔드 (Stage 1)
│   │   ├── vlm_verifier.py       # VLM 판별기 (Stage 2·3 핵심)
│   │   ├── hitl_collector.py     # HITL 큐 관리 및 CLI 리뷰
│   │   └── lora_trainer.py       # LoRA fine-tuning (TRL SFTTrainer)
│   └── evaluation/
│       ├── eval_dataset.py       # 고정 평가셋 생성·저장·로드
│       └── metrics.py            # accuracy / precision / recall / F1
│
├── scripts/
│   ├── run_hitl_inference.py     # VLM 추론 + HITL 큐 채우기
│   ├── hitl_review.py            # 사람 레이블링 CLI
│   ├── lora_train.py             # LoRA 학습 실행
│   └── evaluate_snowball.py      # 3단계 정확도 비교 평가
│
└── data/                         # 런타임 데이터 (git 미추적)
    ├── eval_pairs.jsonl          # 고정 평가셋 (최초 실행 시 자동 생성)
    └── hitl/
        ├── images/               # 저장된 이미지 쌍
        ├── queue.jsonl           # 미검토 예측 큐
        └── labeled.jsonl         # 사람이 레이블링한 학습 데이터
```

---

## 3. 아키텍처

### 3.1 VLM Verifier (`src/models/vlm_verifier.py`)

두 장의 사람 이미지를 받아 "같은 사람인가"를 VLM이 추론합니다.

#### 모델 정보

| 항목 | 내용 |
|------|------|
| 모델 ID | `Qwen/Qwen3-VL-8B-Instruct` |
| 정밀도 | bfloat16 |
| VRAM | ~16GB (CUDA 필수) |
| 클래스 | `Qwen3VLForConditionalGeneration` (transformers 5.x) |

#### 내부 동작 흐름

```
bgr_a, bgr_b (numpy BGR)
      │
      ▼  BGR → RGB → PIL.Image 변환
PIL 이미지 2장
      │
      ▼  processor.apply_chat_template()
토크나이즈된 멀티모달 입력
  ┌─────────────────────────────────────────┐
  │ system: "You are an expert in person    │
  │          re-identification..."          │
  │ user:   [image_a] [image_b]             │
  │         "Answer in exactly this format: │
  │          SAME_PERSON: <YES or NO>       │
  │          CONFIDENCE: <0.0 to 1.0>       │
  │          REASONING: <one sentence>"     │
  └─────────────────────────────────────────┘
      │
      ▼  model.generate(max_new_tokens=256)
raw text 출력
      │
      ▼  _parse_output() — 정규식 파싱
VerificationResult
  ├── is_same: bool
  ├── confidence: float [0.0, 1.0]
  ├── reasoning: str
  └── raw_output: str  (디버깅용)
```

#### 출력 파싱 규칙

모델이 다음 포맷으로 응답하도록 프롬프트를 구성합니다:

```
SAME_PERSON: YES
CONFIDENCE: 0.92
REASONING: Both crops show identical red jacket and dark trousers.
```

| 필드 | 파싱 패턴 | 파싱 실패 시 기본값 |
|------|----------|-------------------|
| `SAME_PERSON` | `SAME_PERSON:\s*(YES\|NO)` | `False` |
| `CONFIDENCE` | `CONFIDENCE:\s*([0-9]*\.?[0-9]+)` | `0.0` |
| `REASONING` | `REASONING:\s*(.+)` | raw_output 전체 |

confidence는 `np.clip(value, 0.0, 1.0)`으로 범위 보장.

#### 코드 예시

```python
from src.models.vlm_verifier import load_vlm_verifier

# Stage 2: 기본 모델
verifier = load_vlm_verifier(
    model_id="Qwen/Qwen3-VL-8B-Instruct",  # 기본값
    hitl_threshold=0.7,    # confidence < 0.7이면 자동으로 HITL 큐에 저장
    hitl_data_dir="data/hitl",
)

# Stage 3: LoRA 어댑터 적용
verifier = load_vlm_verifier(
    hitl_threshold=0.7,
    lora_adapter_path="models/vlm_verifier_lora/latest",
)

result = verifier.verify(bgr_a, bgr_b)
print(result.is_same)      # True / False
print(result.confidence)   # 0.0 ~ 1.0
print(result.reasoning)    # "Both crops show identical..."
```

#### HITL 자동 큐잉

`hitl_threshold`가 설정된 경우, `verify()` 내부에서 confidence가 임계값 미만이면 자동으로 `HITLCollector.log()`를 호출합니다:

```python
# vlm_verifier.py verify() 내부 동작
result = _parse_output(raw)
if self._hitl is not None and result.confidence < self._hitl_threshold:
    self._hitl.log(bgr_a, bgr_b, result)  # 이미지 저장 + queue.jsonl 기록
return result
```

---

### 3.2 HITL 데이터 수집 (`src/models/hitl_collector.py`)

VLM이 불확실하게 판별한 케이스를 사람이 직접 레이블링하여 LoRA 학습 데이터를 구축합니다.

#### HITLSample 데이터 구조

```python
@dataclass
class HITLSample:
    id: str           # UUID4 (고유 식별자)
    img_path_a: str   # 첫 번째 이미지 절대 경로
    img_path_b: str   # 두 번째 이미지 절대 경로
    pred_is_same: bool  # VLM의 예측값
    confidence: float   # VLM의 confidence [0.0, 1.0]
    reasoning: str      # VLM의 한 줄 추론
    label: bool | None  # 사람이 부여한 정답 (None = 미검토)
```

#### log() — 큐에 저장

```python
collector = HITLCollector(data_dir="data/hitl")
sample = collector.log(bgr_a, bgr_b, result)
```

내부 동작:
1. `{uuid}_a.jpg`, `{uuid}_b.jpg`를 `data/hitl/images/`에 JPEG로 저장
2. HITLSample을 JSON으로 직렬화하여 `data/hitl/queue.jsonl`에 append

#### review_pending_cli() — 터미널 리뷰

```bash
uv run python scripts/hitl_review.py
```

각 아이템마다 다음 정보를 출력하고 입력을 받습니다:

```
=== HITL Review — 17 pending samples ===
Commands: s=same  d=different  q=quit

[1/17] ID: 3f8a2b1c...
  VLM prediction : DIFFERENT
  Confidence     : 0.612
  Reasoning      : Different clothing but similar build makes it uncertain.
  Image A        : data/hitl/images/3f8a2b1c_a.jpg
  Image B        : data/hitl/images/3f8a2b1c_b.jpg
  Label [s/d/q]:
```

| 입력 | 동작 |
|------|------|
| `s` | label=True (같은 사람)로 저장 |
| `d` | label=False (다른 사람)로 저장 |
| `q` | 세션 종료 (진행된 레이블링은 저장) |

리뷰 완료된 샘플은 `queue.jsonl`에서 제거되고 `labeled.jsonl`로 이동합니다.

#### 상태 확인

```python
collector = HITLCollector("data/hitl")
print(collector.queue_size)    # 미검토 샘플 수
print(collector.labeled_size)  # 레이블링 완료 샘플 수
```

---

### 3.3 LoRA 학습 (`src/models/lora_trainer.py`)

`labeled.jsonl`의 사람 레이블 데이터로 Qwen3-VL-8B-Instruct를 LoRA fine-tuning합니다.

#### 학습 데이터 포맷

TRL 0.29.x에서는 이미지를 메시지 안 경로가 아닌 **별도 `"images"` 컬럼에 PIL Image 객체**로 제공해야 합니다.
메시지에는 `{"type": "image"}` placeholder만 포함합니다:

```python
{
    "messages": [
        {
            "role": "system",
            "content": [{"type": "text", "text": "You are an expert in person re-identification..."}]
        },
        {
            "role": "user",
            "content": [
                {"type": "image"},          # placeholder (경로 없음)
                {"type": "image"},          # placeholder (경로 없음)
                {"type": "text", "text": "The two images below are person crops..."}
            ]
        },
        {
            "role": "assistant",
            "content": [{"type": "text", "text": "SAME_PERSON: YES\nCONFIDENCE: 0.95\nREASONING: ..."}]
        }
    ],
    "images": [PIL.Image, PIL.Image]    # RGB PIL Image 객체 리스트
}
```

#### Qwen3-VL 멀티모달 학습 필수 설정

Qwen3-VL은 동적 해상도(dynamic resolution)를 사용하며, 이미지를 28×28 패치 단위로 처리합니다.
VLM LoRA 학습 시 텍스트 전용 학습과 달리 아래 설정이 **필수**입니다:

| 설정 | 값 | 이유 |
|------|---|------|
| `min_pixels` / `max_pixels` | `128*28*28` / `512*28*28` | ReID crop은 작은 이미지(64×128 등). 미설정 시 이미지당 토큰이 폭발하여 OOM 위험 |
| `skip_prepare_dataset` | `True` | SFTTrainer의 자체 전처리가 멀티모달 데이터를 올바르게 처리하지 못함. 반드시 우회 |
| custom `data_collator` | 아래 참고 | 이미지 토큰과 패딩에 `labels=-100` 마스킹 필수 |
| `attn_implementation` | `"sdpa"` | padded 멀티모달 입력에서 flash attention 호환 문제 방지 |
| `gradient_checkpointing_kwargs` | `{"use_reentrant": False}` | LoRA + gradient checkpointing 호환성 |

##### 이미지 토큰 Label Masking

Qwen3-VL의 이미지 관련 special token:

| 토큰 | ID | 용도 |
|------|---|------|
| `<\|vision_start\|>` | 151652 | 이미지 시작 마커 |
| `<\|vision_end\|>` | 151653 | 이미지 종료 마커 |
| `<\|image_pad\|>` | 151655 | 이미지 픽셀 데이터 placeholder |

이 토큰들에 loss를 걸면 모델이 이미지 토큰 재생산을 학습하게 됩니다.
custom `data_collator`에서 이 토큰들과 padding 토큰의 labels를 `-100`으로 설정해야 합니다:

```python
def collate_fn(examples):
    texts = [
        processor.apply_chat_template(
            ex["messages"], tokenize=False, add_generation_prompt=False
        ).strip()
        for ex in examples
    ]
    image_inputs = [ex["images"] for ex in examples]
    batch = processor(
        text=texts, images=image_inputs, return_tensors="pt", padding=True
    )
    labels = batch["input_ids"].clone()
    # 이미지 토큰 + 패딩 토큰은 loss에서 제외
    labels[labels == processor.tokenizer.pad_token_id] = -100
    for token_id in [151652, 151653, 151655]:
        labels[labels == token_id] = -100
    batch["labels"] = labels
    return batch
```

#### LoRA 하이퍼파라미터

| 파라미터 | 기본값 | 설명 |
|----------|--------|------|
| `lora_r` | 16 | LoRA rank (낮을수록 파라미터 적음) |
| `lora_alpha` | 32 | scaling factor (alpha/r = 2.0) |
| `lora_dropout` | 0.05 | 드롭아웃 비율 |
| `target_modules` | `["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]` | attention + MLP 레이어. vision encoder는 제외 (고정) |
| `learning_rate` | 2e-4 | AdamW 학습률 |
| `warmup_ratio` | 0.03 | 학습 초기 warm-up 비율 |
| `lr_scheduler_type` | `"constant"` | warm-up 후 고정 학습률 |
| `max_grad_norm` | 0.3 | gradient clipping (학습 안정성) |
| `num_epochs` | 3 | 전체 데이터 반복 횟수 |
| `batch_size` | 1 | gradient accumulation steps=8 → 유효 배치=8 |
| `max_samples` | 100 | labeled.jsonl에서 랜덤 샘플링할 최대 수 (CLI `--max-samples`) |
| `dtype` | bfloat16 | gradient checkpointing 활성화 |

#### 어댑터 버전 관리

학습이 완료되면 versioned 디렉터리에 어댑터만 저장합니다 (전체 모델 저장 안 함):

```
models/vlm_verifier_lora/
├── v1/
│   ├── adapter_config.json         # LoRA 설정 (r, alpha, target_modules 등)
│   └── adapter_model.safetensors  # 학습된 LoRA 가중치
├── v2/
│   └── ...
└── latest -> v2                    # 항상 최신 버전을 가리키는 symlink
```

새 버전이 저장될 때마다 `latest` symlink가 자동 갱신됩니다.

#### TRL SFTTrainer VLM 학습 시 주의사항

- `remove_unused_columns=False` 필수 — images 컬럼이 삭제되지 않도록
- `dataset_kwargs={"skip_prepare_dataset": True}` — SFTTrainer 자체 멀티모달 전처리 우회
- `data_collator=collate_fn` — 이미지 토큰 label masking이 적용된 custom collator 사용
- vision encoder는 LoRA 대상에서 제외 — 이미 잘 학습된 feature extractor이므로 고정
- `max_length` 설정 시 이미지 토큰이 잘리지 않도록 충분히 크게 설정

---

### 3.4 평가 파이프라인 (`src/evaluation/`)

3개 스테이지를 **동일한 고정 평가셋**으로 공정하게 비교합니다.

#### 데이터 분리 규칙 (중요)

```
bounding_box_train/ ──▶ run_hitl_inference.py ──▶ HITL 큐 ──▶ LoRA 학습
bounding_box_test/  ──▶ evaluate_snowball.py  ──▶ 3단계 정확도 평가
```

test 이미지는 LoRA 학습에 절대 사용되지 않으므로 평가 결과가 공정합니다.

#### EvalPair 구조

```python
@dataclass
class EvalPair:
    img_path_a: str    # query 이미지 절대 경로
    img_path_b: str    # bounding_box_test/ 이미지 절대 경로
    label: bool        # True=같은 사람, False=다른 사람
    person_id_a: str   # query 인물 ID (파일명에서 추출)
    person_id_b: str   # gallery 인물 ID
```

#### 고정 평가셋을 쓰는 이유

3개 스테이지를 서로 다른 시점에 실행하더라도 **완전히 동일한 이미지 쌍**으로 비교해야 정확도 향상이 모델 개선에 의한 것임을 증명할 수 있습니다.

`data/eval_pairs.jsonl`은 최초 실행 시 `seed=42`로 한 번만 생성되고, 이후 모든 평가에서 재사용됩니다.

평가셋 구성: `n_queries`개의 query마다 positive 1개 + negative 1개 = 총 `2 × n_queries`쌍 (50:50 균형).

#### EvalMetrics

```python
@dataclass
class EvalMetrics:
    accuracy: float    # (TP + TN) / 전체
    precision: float   # TP / (TP + FP)
    recall: float      # TP / (TP + FN)
    f1: float          # 2 × precision × recall / (precision + recall)
    n_correct: int
    n_total: int
    tp: int   # 같은 사람으로 예측 & 실제 같은 사람
    tn: int   # 다른 사람으로 예측 & 실제 다른 사람
    fp: int   # 같은 사람으로 예측 & 실제 다른 사람 (오검출)
    fn: int   # 다른 사람으로 예측 & 실제 같은 사람 (미검출)
```

#### find_best_threshold (Stage 1용)

ReID 코사인 유사도는 연속값이므로, 이진 분류를 위한 임계값이 필요합니다. `find_best_threshold()`는 `[min_sim, max_sim]` 구간을 101개 후보로 grid search하여 accuracy를 최대화하는 임계값을 반환합니다.

---

## 4. 환경 설정

### 필요 사양

| 항목 | 요구사항 |
|------|---------|
| Python | 3.10 |
| 패키지 관리 | uv (pip 사용 금지) |
| GPU | CUDA ≥16GB VRAM (VLM 추론·학습 필수) |
| 권장 환경 | GCP L4 (24GB) |

### 설치

```bash
git clone https://github.com/kokyungmin7/revlm.git
cd revlm
uv sync   # pyproject.toml 기반 의존성 설치
```

VLM 모델(`Qwen3-VL-8B-Instruct`)은 최초 실행 시 HuggingFace에서 자동 다운로드됩니다.

### 데이터셋 경로

```
/home/kokyungmin/data/WB_WoB-ReID/
├── both_large/
│   ├── bounding_box_test/    # 평가 전용 갤러리
│   ├── bounding_box_train/   # HITL 추론 전용 갤러리
│   └── query/                # 쿼리 이미지
├── both_small/
├── with_bag/
└── without_bag/
```

---

## 5. 스크립트 레퍼런스

### `scripts/run_hitl_inference.py`

VLM으로 train 갤러리 쌍을 추론하고, confidence가 낮은 예측을 HITL 큐에 저장합니다.

| 옵션 | 기본값 | 설명 |
|------|--------|------|
| `--data-root` | `/home/kokyungmin/data/WB_WoB-ReID` | 데이터셋 루트 경로 |
| `--split` | `both_large` | 사용할 split (both_large / both_small / with_bag / without_bag) |
| `--n-queries` | `50` | 처리할 query 이미지 수 |
| `--threshold` | `0.7` | 이 값 미만의 confidence → HITL 큐 등록 |
| `--hitl-dir` | `data/hitl` | HITL 데이터 저장 경로 |

```bash
uv run python scripts/run_hitl_inference.py \
  --split both_large \
  --n-queries 100 \
  --threshold 0.7
```

출력 예시:
```
────────────────────────────────────────────────────────────────────────
[1/100] Query  person_id=0042  file=0042_c1_f0012.jpg  size=64x128
  [+] Positive  person_id=0042  file=0042_c3_f0087.jpg  size=64x128
      Prediction : SAME       (GT: SAME)  ✓ correct
      Confidence : 0.823
      Reasoning  : Both crops show identical red jacket and black pants.
  [-] Negative  person_id=0117  file=0117_c2_f0034.jpg  size=64x128
      Prediction : DIFFERENT  (GT: DIFFERENT)  ✓ correct
      Confidence : 0.612  ← QUEUED (< 0.7)
      Reasoning  : Different clothing but similar build makes it uncertain.
...
=== Run Summary ===
  Queries processed : 100  (skipped: 0)
  Pairs evaluated   : 200
  VLM accuracy      : 167/200 = 83.5%
  Queued for review : 23  (confidence < 0.7)
```

---

### `scripts/hitl_review.py`

HITL 큐에 쌓인 예측을 터미널에서 직접 레이블링합니다.

| 옵션 | 기본값 | 설명 |
|------|--------|------|
| `--data-dir` | `data/hitl` | HITL 데이터 디렉터리 경로 |

```bash
uv run python scripts/hitl_review.py
```

실행 시 현재 큐/레이블 상태를 먼저 출력하고, 미검토 샘플이 있으면 순차 리뷰를 시작합니다:

```
HITL data directory : data/hitl
Pending review      : 23
Already labeled     : 77

=== HITL Review — 23 pending samples ===
Commands: s=same  d=different  q=quit

[1/23] ID: 3f8a2b1c...
  VLM prediction : DIFFERENT
  Confidence     : 0.612
  Reasoning      : Different clothing but similar build...
  Image A        : data/hitl/images/3f8a2b1c_a.jpg
  Image B        : data/hitl/images/3f8a2b1c_b.jpg
  Label [s/d/q]: s
  → Labeled: SAME

...
Review complete. Labeled 23 samples.
Total labeled: 100
```

---

### `scripts/lora_train.py`

`labeled.jsonl`의 데이터로 LoRA fine-tuning을 실행합니다. CUDA 필수.

| 옵션 | 기본값 | 설명 |
|------|--------|------|
| `--labeled-jsonl` | `data/hitl/labeled.jsonl` | 학습 데이터 경로 |
| `--output-base` | `models/vlm_verifier_lora` | 어댑터 저장 루트 |
| `--model-id` | `Qwen/Qwen3-VL-8B-Instruct` | 베이스 모델 |
| `--min-samples` | `100` | 이 미만이면 학습 건너뜀 |
| `--epochs` | `3` | 학습 에폭 수 |
| `--lora-r` | `16` | LoRA rank |
| `--lora-alpha` | `32` | LoRA alpha |

```bash
uv run python scripts/lora_train.py \
  --min-samples 100 \
  --epochs 3
```

실행 흐름:
```
[LoRATrainer] Loading base model...
trainable params: 8,388,608 || all params: 8,197,050,368 || trainable%: 0.1023
[LoRATrainer] Starting training...
{'loss': 0.4231, 'epoch': 1.0}
{'loss': 0.3102, 'epoch': 2.0}
{'loss': 0.2589, 'epoch': 3.0}
[LoRATrainer] Adapter saved to: models/vlm_verifier_lora/v1

Training complete. Adapter saved to: models/vlm_verifier_lora/v1
To use: load_vlm_verifier(lora_adapter_path='models/vlm_verifier_lora/v1')
```

---

### `scripts/evaluate_snowball.py`

3단계를 동일한 test 평가셋으로 실행하고 정확도 비교표를 출력합니다.

| 옵션 | 기본값 | 설명 |
|------|--------|------|
| `--data-root` | `/home/kokyungmin/data/WB_WoB-ReID` | 데이터셋 루트 |
| `--eval-pairs` | `data/eval_pairs.jsonl` | 고정 평가셋 경로 (없으면 자동 생성) |
| `--n-queries` | `100` | 평가셋 생성 시 사용할 query 수 |
| `--split` | `both_large` | 데이터셋 split |
| `--reid-model` | `arcface-dinov2` | Stage 1 모델 (arcface-dinov2 / siglip2) |
| `--hitl-threshold` | `0.7` | Stage 2·3에서 HITL 큐잉 임계값 |
| `--lora-adapter` | `models/vlm_verifier_lora/latest` | Stage 3 어댑터 경로 |
| `--skip-stage` | `` | 건너뛸 스테이지 (예: `3` 또는 `2,3`) |
| `--output` | `experiments/results/snowball.json` | 결과 JSON 저장 경로 |
| `--seed` | `42` | 평가셋 생성 시 랜덤 시드 |

```bash
# Stage 1·2만 (LoRA 학습 전)
uv run python scripts/evaluate_snowball.py \
  --split both_large \
  --skip-stage 3

# 전체 3단계 비교
uv run python scripts/evaluate_snowball.py --split both_large
```

최종 출력:
```
════════════════════════════════════════════════════════════════════════
  ReID Accuracy Snowball Loop — Summary
════════════════════════════════════════════════════════════════════════
  Stage                                    Acc    Prec  Recall      F1
  ──────────────────────────────────────────────────────────────────────
  1. ReID only (arcface-dinov2)          73.50%  71.20%  77.40%  74.17%
  2. VLM verifier (base)                 83.00%  81.50%  85.20%  83.31%
  3. VLM + HITL LoRA (v1)               89.50%  88.10%  91.30%  89.67%
  ──────────────────────────────────────────────────────────────────────
  Overall improvement (stage 1 → 3): +16.00pp
  Stage 1 → 2: +9.50pp
  Stage 2 → 3: +6.50pp
════════════════════════════════════════════════════════════════════════

Results saved → experiments/results/snowball.json
```

---

## 6. 전체 실험 워크스루

### Step 1 — 환경 준비

```bash
git clone https://github.com/kokyungmin7/revlm.git && cd revlm
uv sync
```

생성 파일: 없음

---

### Step 2 — Stage 1·2 베이스라인 평가

LoRA 학습 전 ReID와 VLM base 성능을 먼저 측정합니다.

```bash
uv run python scripts/evaluate_snowball.py \
  --split both_large \
  --n-queries 100 \
  --skip-stage 3
```

생성 파일:
- `data/eval_pairs.jsonl` — 200쌍 고정 평가셋 (이후 모든 평가에 재사용)
- `experiments/results/snowball.json` — Stage 1·2 결과

---

### Step 3 — HITL 큐 채우기

VLM을 train 갤러리에 돌려 불확실한 케이스를 수집합니다.

```bash
uv run python scripts/run_hitl_inference.py \
  --split both_large \
  --n-queries 100 \
  --threshold 0.7
```

생성 파일:
- `data/hitl/images/{uuid}_a.jpg`, `{uuid}_b.jpg` — 이미지 쌍
- `data/hitl/queue.jsonl` — 미검토 예측 큐

---

### Step 4 — 사람 레이블링

터미널에서 큐의 예측을 검토하고 정답을 입력합니다.

```bash
uv run python scripts/hitl_review.py
# s / d / q 로 레이블링
# 목표: ≥100개 레이블링
```

생성 파일:
- `data/hitl/labeled.jsonl` — LoRA 학습 입력 데이터

---

### Step 5 — LoRA 학습

레이블링된 데이터로 어댑터를 학습합니다. CUDA 필수.

```bash
uv run python scripts/lora_train.py \
  --min-samples 100 \
  --epochs 3
```

생성 파일:
- `models/vlm_verifier_lora/v1/adapter_config.json`
- `models/vlm_verifier_lora/v1/adapter_model.safetensors`
- `models/vlm_verifier_lora/latest` → `v1` (symlink)

---

### Step 6 — 전체 3단계 최종 평가

Step 2에서 만든 동일한 `eval_pairs.jsonl`로 3단계를 모두 비교합니다.

```bash
uv run python scripts/evaluate_snowball.py --split both_large
```

생성 파일:
- `experiments/results/snowball.json` — Stage 1·2·3 결과 갱신

---

### 반복 개선 (Snowball)

```bash
# Stage 3 결과를 바탕으로 추가 HITL 수집
uv run python scripts/run_hitl_inference.py --split both_large

# 추가 레이블링 (누적 데이터 활용)
uv run python scripts/hitl_review.py

# v2 어댑터 학습 (labeled.jsonl에 누적된 전체 데이터 사용)
uv run python scripts/lora_train.py --min-samples 50

# 재평가 (동일한 eval_pairs.jsonl 사용)
uv run python scripts/evaluate_snowball.py \
  --lora-adapter models/vlm_verifier_lora/latest \
  --output experiments/results/snowball_v2.json
```

---

## 7. 데이터·모델 저장 구조

### HITL 데이터 (`data/hitl/`)

```
data/hitl/
├── images/
│   ├── {uuid4}_a.jpg     # VLM 추론 시 저장된 첫 번째 crop
│   └── {uuid4}_b.jpg     # VLM 추론 시 저장된 두 번째 crop
├── queue.jsonl            # 미검토 예측 (label=null)
└── labeled.jsonl          # 레이블링 완료 (label=true/false)
```

`queue.jsonl` / `labeled.jsonl` 각 라인 스키마:

```json
{
  "id": "3f8a2b1c-4d5e-6f7a-8b9c-0d1e2f3a4b5c",
  "img_path_a": "data/hitl/images/3f8a2b1c_a.jpg",
  "img_path_b": "data/hitl/images/3f8a2b1c_b.jpg",
  "pred_is_same": false,
  "confidence": 0.612,
  "reasoning": "Different clothing but similar build makes it uncertain.",
  "label": null
}
```

`label`은 queue에서 `null`, labeled로 이동 후 `true` 또는 `false`.

### 평가셋 (`data/eval_pairs.jsonl`)

각 라인 스키마:

```json
{
  "img_path_a": "/home/kokyungmin/data/WB_WoB-ReID/both_large/query/0042_c1_f0012.jpg",
  "img_path_b": "/home/kokyungmin/data/WB_WoB-ReID/both_large/bounding_box_test/0042_c3_f0087.jpg",
  "label": true,
  "person_id_a": "0042",
  "person_id_b": "0042"
}
```

- `label=true`: positive 쌍 (같은 사람)
- `label=false`: negative 쌍 (다른 사람)
- `img_path_b`는 항상 `bounding_box_test/` 소속 (train 이미지 없음)

### LoRA 어댑터 (`models/vlm_verifier_lora/`)

```
models/vlm_verifier_lora/
├── v1/
│   ├── adapter_config.json         # r, alpha, target_modules 등 설정
│   └── adapter_model.safetensors  # 학습된 LoRA 가중치 (~64MB)
├── v2/
│   └── ...
└── latest -> v2                    # 최신 버전 symlink (자동 갱신)
```

어댑터 로드:
```python
from peft import PeftModel
model = PeftModel.from_pretrained(base_model, "models/vlm_verifier_lora/latest")
```

또는 `load_vlm_verifier(lora_adapter_path="models/vlm_verifier_lora/latest")`로 자동 처리.

---

## 8. 테스트

### `tests/smoke/test_hitl.py` — CPU, 모델 로드 불필요

| 테스트 | 검증 내용 |
|--------|----------|
| `test_hitl_collector_log_creates_files` | log() 호출 시 이미지 파일과 queue.jsonl 생성 확인 |
| `test_hitl_collector_log_metadata` | 저장된 메타데이터가 VerificationResult와 일치 확인 |
| `test_hitl_queue_size_increments` | log() 호출마다 queue_size 증가 확인 |
| `test_hitl_labeled_size_starts_at_zero` | 초기 labeled_size=0 확인 |
| `test_hitl_review_cli_labels_sample` | `s` 입력 시 SAME 레이블로 labeled.jsonl 이동 확인 |
| `test_hitl_review_cli_quit_stops_early` | `q` 입력 시 0건 레이블링 확인 |
| `test_hitl_review_cli_different_label` | `d` 입력 시 label=false로 저장 확인 |
| `test_hitl_review_empty_queue` | 빈 큐에서 0 반환 확인 |
| `test_hitl_labeled_jsonl_format` | labeled.jsonl JSONL 스키마 검증 |

```bash
uv run pytest tests/smoke/test_hitl.py -v
```

### `tests/smoke/test_vlm_verifier.py` — CUDA 필수 (~16GB)

| 테스트 | 검증 내용 |
|--------|----------|
| `test_vlm_verifier_loads` | 모델·프로세서 로드 및 속성 존재 확인 |
| `test_verify_returns_result_type` | 반환 타입이 VerificationResult인지 확인 |
| `test_confidence_in_valid_range` | confidence가 [0.0, 1.0] 범위인지 확인 |
| `test_same_person_is_same` | 같은 사람 이미지 쌍 → is_same=True 확인 |
| `test_different_person_is_different` | 다른 사람 이미지 쌍 → is_same=False 확인 |
| `test_verify_saves_comparison` | 3-panel 비교 이미지 저장 확인 |

```bash
uv run pytest tests/smoke/test_vlm_verifier.py -v -s
```
