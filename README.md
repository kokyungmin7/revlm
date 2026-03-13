# RevLM

VLM(Vision Language Model)을 활용한 Re-Identification 프로젝트.
인간 검수 비용 절감과 Appearance Alignment를 통한 ReID 성능 고도화를 목표로 합니다.

## 주요 목표
- VLM 기반 자동화로 ReID 파이프라인의 인간 개입 최소화
- Cross-modal Appearance Alignment로 ReID 정확도 향상

## 실행 환경 설정

### 요구사항
- Python 3.10
- [uv](https://github.com/astral-sh/uv) (패키지 관리자)

### 설치

```bash
# 저장소 클론
git clone <repo-url>
cd revlm

# 가상환경 생성 및 의존성 설치
uv sync
```

### 실행

```bash
# 메인 실행
uv run python main.py

# 테스트
uv run pytest

# 의존성 추가
uv add <package-name>
```

## Inference 환경 설정

새 서버/머신에서 처음 세팅하는 경우:

### 1. uv 설치 (없는 경우)

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

### 2. 저장소 클론 및 의존성 설치

```bash
git clone <repo-url>
cd revlm
uv sync   # uv.lock 기반으로 의존성 정확히 재현
```

### 3. MEBOW 소스 설치

```bash
uv run python scripts/setup_mebow.py
```

### 4. 모델 가중치 배치

`models/` 폴더에 다음 파일을 직접 복사하거나 다운로드합니다:

```
models/
├── detect/
│   └── yolo26n-pose.pt              # Ultralytics YOLO Pose
└── body_orientation/
    ├── model_hboe.pth               # MEBOW HOE 가중치
    └── pose_hrnet_w32_256x192.pth   # HRNet pretrained 가중치
```

### 5. 실행 확인

```bash
uv run pytest tests/preprocessing/ -v   # 단위 테스트
uv run python main.py                   # 메인 실행
```

## 프로젝트 구조

```
revlm/
├── src/
│   ├── preprocessing/   # 전처리 (각도 정규화 등)
│   ├── models/          # VLM/ReID 모델
│   └── utils/           # 공통 유틸리티
├── data/                # 데이터셋 (git 미추적)
├── experiments/         # 실험 결과
├── docs/                # 기획서 및 도메인 문서
└── tests/               # 테스트 코드
```

## 문서

- [기획서](docs/PLANNING.md) - 프로젝트 목표, 로드맵
- [도메인 맥락](docs/CONTEXT.md) - ReID/VLM 배경지식
- [TODO](TODO.md) - 현재 작업 목록
