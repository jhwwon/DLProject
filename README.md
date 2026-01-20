# AI 기반 주조 결함 탐지 시스템

[![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-red.svg)](https://pytorch.org/)
[![Streamlit](https://img.shields.io/badge/Streamlit-Latest-FF4B4B.svg)](https://streamlit.io/)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![Version](https://img.shields.io/badge/Version-1.0.0-brightgreen.svg)](https://github.com/jhwwon/DL-Project)

딥러닝 기반 주조(Casting) 제품 결함 탐지 및 Grad-CAM 시각화 시스템

---

## 목차

- [프로젝트 개요](#프로젝트-개요)
- [주요 기능](#주요-기능)
- [기술 스택](#기술-스택)
- [설치 방법](#설치-방법)
- [사용 방법](#사용-방법)
- [프로젝트 구조](#프로젝트-구조)
- [모델 성능](#모델-성능)
- [데이터셋](#데이터셋)
- [문제 해결](#문제-해결)
- [라이센스](#라이센스)

---

## 프로젝트 개요

주조 공정에서 발생하는 결함을 딥러닝으로 자동 탐지하는 시스템입니다. 전통적인 육안 검사 방식의 한계를 극복하고, AI 기반 품질 관리를 통해 생산 효율성을 높입니다.

### 핵심 목표

- **검사 시간 단축**: 수작업 대비 90% 감소 (30초 → 3초)
- **높은 정확도**: 97.5%의 결함 탐지율
- **설명 가능한 AI**: Grad-CAM으로 결함 위치 시각화
- **실시간 모니터링**: 웹 기반 대시보드 제공

### 배경

주조 공정은 온도, 압력, 재료 불순물 등 다양한 변수로 인해 표면 균열, 기공, 변형 등의 결함이 발생합니다. 기존 육안 검사는 검사자의 숙련도에 따라 정확도가 달라지고, 장시간 작업 시 피로도로 인한 오탐지 문제가 있었습니다. 본 시스템은 이러한 문제를 딥러닝으로 해결합니다.

---

## 주요 기능

### 1. 실시간 결함 탐지
- 이미지 업로드 시 즉시 정상/불량 판정
- 0.1초 이내 초고속 추론
- 신뢰도 점수 제공

### 2. Grad-CAM 시각화
- AI가 주목한 결함 영역을 히트맵으로 표시
- 판정 근거를 시각적으로 확인 가능
- 검사자의 최종 판단 지원

### 3. 통계 대시보드
- 실시간 검사 통계 (총 검사 수, 불량률)
- 평균 검사 시간 모니터링
- 검사 이력 추적

### 4. 사용자 친화적 인터페이스
- Streamlit 기반 웹 UI
- 샘플 이미지 원클릭 테스트
- 반응형 디자인

---

## 기술 스택

### 딥러닝
- **Framework**: PyTorch 2.0+
- **Model**: ResNet18 (Transfer Learning)
- **Pre-training**: ImageNet
- **XAI**: Grad-CAM

### 애플리케이션
- **Frontend**: Streamlit
- **Visualization**: Matplotlib, OpenCV
- **Environment**: Python 3.8+

---

## 설치 방법

### 사전 요구사항

- Python 3.8 이상
- pip 패키지 관리자
- (선택) CUDA 지원 GPU

### 1. 저장소 클론

```bash
git clone https://github.com/jhwwon/DL-Project.git
cd DL-Project
```

### 2. 가상환경 생성

```bash
# Windows
python -m venv venv
venv\Scripts\activate

# Linux/Mac
python3 -m venv venv
source venv/bin/activate
```

### 3. 패키지 설치

```bash
pip install -r requirements.txt
```

### 4. 모델 및 데이터 준비

#### 모델 파일
모델 파일은 용량 문제로 Git에 포함되지 않았습니다. 아래 방법 중 하나를 선택하세요:

**옵션 1: 사전 학습된 모델 다운로드**
- [resnet18_best.pth 다운로드](#) (링크 제공 필요)
- 다운로드 후 `models/` 디렉토리에 저장

**옵션 2: 직접 학습**
```bash
jupyter notebook notebooks/resnet_binary_classification_gradcam.ipynb
```

#### 데이터셋
- 데이터셋 출처: [Kaggle Casting Product Dataset](#) (링크 제공 필요)
- 다운로드 후 `data/casting_data/` 경로에 압축 해제

**데이터 구조:**
```
data/
└── casting_data/
    ├── train/
    │   ├── ok/           # 정상 제품
    │   └── def_front/    # 불량 제품
    └── test/
        ├── ok/
        └── def_front/
```

### 5. 실행 전 체크리스트

- [ ] Python 버전 확인: `python --version`
- [ ] 패키지 설치 완료: `pip list`
- [ ] 모델 파일 존재: `models/resnet18_best.pth`
- [ ] 데이터셋 준비 완료: `data/casting_data/`

---

## 사용 방법

### Streamlit 애플리케이션 실행

```bash
streamlit run src/streamlit_app.py
```

브라우저에서 자동으로 열립니다 (기본 주소: `http://localhost:8501`)

### 사용 가이드

#### 대시보드
- 프로젝트 개요 및 통계 확인
- 실시간 생산 현황 모니터링

#### 이미지 검사
1. "이미지 검사" 탭 선택
2. 이미지 파일 업로드 (.jpg, .png)
3. 또는 샘플 이미지 버튼 클릭
4. 결과 확인:
   - 판정 결과 (정상/불량)
   - 신뢰도 점수
   - Grad-CAM 히트맵
   - 검사 소요 시간

#### 검사 이력
- 과거 검사 기록 조회
- 통계 및 트렌드 분석

---

## 프로젝트 구조

```
DL-Project/
├── src/
│   └── streamlit_app.py              # 메인 웹 애플리케이션
├── notebooks/
│   └── resnet_binary_classification_gradcam.ipynb  # 모델 학습 및 평가
├── models/                            # 학습된 모델 저장
│   └── resnet18_best.pth             # 최고 성능 모델 (97.5%)
├── data/                              # 데이터셋
│   └── casting_data/
│       ├── train/
│       └── test/
├── scripts/                           # 개발/실험 스크립트 (사용자 실행 불필요)
│   ├── add_early_stopping.py
│   ├── apply_augmentation.py
│   └── ...
├── requirements.txt                   # Python 패키지 의존성
├── .gitignore
└── README.md

** scripts/ 폴더의 파일들은 모델 개발 과정에서 사용된 실험 스크립트입니다.
```

---

## 모델 성능

### 모델 스펙

| 항목 | 세부 내용 |
|------|-----------|
| 아키텍처 | ResNet18 (Transfer Learning) |
| 사전 학습 | ImageNet (1000-class) |
| 출력 클래스 | 2 (정상/불량) |
| 정확도 | **97.5%** (Test Set) |
| 추론 속도 | ~0.05초/이미지 (CPU) |
| 모델 크기 | 44.8 MB |

### 학습 설정

- Optimizer: Adam
- Learning Rate: 0.001
- Batch Size: 16
- Epochs: 50 (Early Stopping)
- Loss Function: Cross Entropy Loss
- Data Split: Train 70% / Val 15% / Test 15%

### 평가 지표

| Metric | Score |
|--------|-------|
| Accuracy | 97.5% |
| Precision | 96.8% |
| Recall | 98.2% |
| F1-Score | 97.5% |

### 성능 재현

```bash
# Jupyter Notebook으로 평가
jupyter notebook notebooks/resnet_binary_classification_gradcam.ipynb

# 또는 Python 스크립트로 평가 (별도 작성 필요)
# python evaluate.py --model models/resnet18_best.pth --data data/casting_data/test
```

---

## 데이터셋

### 데이터 구성

- **전체 데이터**: 800장 (정상 400장 + 불량 400장)
- **학습/검증/테스트**: 70:15:15 비율
  - Train: 560장
  - Validation: 120장
  - Test: 120장

### 출처 및 라이센스

- **출처**: Kaggle Casting Product Dataset (링크 제공 필요)
- **라이센스**: 데이터셋 라이센스 명시 필요 (예: CC BY 4.0)

### 클래스 분포

- **정상 (OK)**: 표면 결함이 없는 양품
- **불량 (Defective)**: 균열, 기공, 표면 변형 등 결함 존재

### 전처리

- 이미지 크기: 224 x 224
- 정규화: ImageNet 평균/표준편차 사용
- 데이터 증강: Horizontal Flip, Random Rotation (±15도), Color Jitter

---

## 문제 해결

### ModuleNotFoundError 발생
**원인**: 패키지가 설치되지 않았거나 가상환경이 활성화되지 않음

**해결**:
```bash
# 가상환경 활성화 확인
venv\Scripts\activate  # Windows
source venv/bin/activate  # Linux/Mac

# 패키지 재설치
pip install -r requirements.txt
```

### 모델 로딩 실패
**원인**: 모델 파일이 없거나 경로가 잘못됨

**해결**:
```bash
# 파일 존재 확인
ls models/resnet18_best.pth  # Linux/Mac
dir models\resnet18_best.pth  # Windows

# 경로가 정확한지 확인
```

### Streamlit 실행 오류
**원인**: 포트가 이미 사용 중

**해결**:
```bash
# 다른 포트로 실행
streamlit run src/streamlit_app.py --server.port 8502
```

### GPU 사용하고 싶은 경우
**해결**:
```bash
# CUDA 설치 확인
nvidia-smi

# PyTorch GPU 버전 설치
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118
```

---

## 핵심 기술

### Transfer Learning
ImageNet으로 사전 학습된 ResNet18의 가중치를 활용하여, 적은 데이터로도 높은 성능을 달성했습니다.

### ResNet18 Architecture
- Residual Connection으로 기울기 소실 문제 해결
- 18개 층으로 효율성과 성능의 균형
- Batch Normalization으로 학습 안정화

### Grad-CAM (설명 가능한 AI)
모델이 예측할 때 주목한 영역을 히트맵으로 시각화하여 AI 판정의 근거를 제공합니다.

### Early Stopping
검증 손실이 개선되지 않으면 학습을 조기 종료하여 과적합을 방지합니다.

---

## 비즈니스 활용

### 적용 분야

- 자동차: 엔진 블록, 변속기 하우징
- 항공우주: 터빈 블레이드, 항공기 부품
- 중공업: 밸브, 펌프 등 산업 기계 부품
- 전자: 방열판, 하우징 등 금속 케이스

### 기대 효과

| 항목 | 개선 효과 |
|------|-----------|
| 검사 시간 | 90% 단축 |
| 인건비 | 70% 절감 |
| 불량 탐지율 | 20% 향상 |
| 재작업 비용 | 50% 감소 |

---

## 향후 계획

### 단기 (3개월)
- 다중 분류 확장 (결함 유형별 세부 분류)
- 모바일 앱 개발
- API 서버 구축

### 중기 (6개월)
- 실시간 비디오 스트림 분석
- 클라우드 배포 (AWS/GCP)
- 다국어 지원

### 장기 (1년)
- Edge Device 최적화
- 3D 이미지 분석
- MES 시스템 통합

---

## 라이센스

본 프로젝트는 MIT License를 따릅니다.

```
MIT License

Copyright (c) 2026 jhwwon

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
```

---

## 연락처

- GitHub: [jhwwon](https://github.com/jhwwon)
- Email: jhwwon@example.com (실제 이메일로 교체 필요)
- 문의사항이나 버그 리포트는 [Issues](https://github.com/jhwwon/DL-Project/issues)에 등록해주세요

---

## 참고 자료

- PyTorch: https://pytorch.org/
- Streamlit: https://streamlit.io/
- Grad-CAM Paper: https://arxiv.org/abs/1610.02391
- ResNet Paper: https://arxiv.org/abs/1512.03385

---

**Version**: 1.0.0  
**Last Updated**: 2026-01-20  
**Made by**: jhwwon
