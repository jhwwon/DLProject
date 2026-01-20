# AI 기반 주조 결함 탐지 시스템

[![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-red.svg)](https://pytorch.org/)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

딥러닝으로 주조 제품의 결함을 자동으로 탐지하고 Grad-CAM으로 결함 위치를 시각화하는 시스템입니다.

## 주요 기능

- **자동 결함 탐지**: 이미지 업로드만으로 정상/불량 판정 (정확도 97.5%)
- **Grad-CAM 시각화**: AI가 주목한 결함 영역을 히트맵으로 표시
- **웹 인터페이스**: Streamlit 기반 사용자 친화적 UI
- **실시간 통계**: 검사 현황 및 불량률 모니터링

## 빠른 시작

### 1. 설치

```bash
git clone https://github.com/jhwwon/DL-Project.git
cd DL-Project
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

### 2. 모델 및 데이터 준비

**모델 파일**: `models/resnet18_best.pth` (별도 다운로드 필요)
- [다운로드 링크](#) (제공 필요)

**데이터셋**: `data/casting_data/` (별도 다운로드 필요)
- [Kaggle Dataset](#) (제공 필요)

### 3. 실행

```bash
streamlit run src/streamlit_app.py
```

브라우저에서 `http://localhost:8501` 접속

## 프로젝트 구조

```
DL-Project/
├── src/
│   └── streamlit_app.py          # 메인 웹 애플리케이션
├── notebooks/
│   └── resnet_binary_classification_gradcam.ipynb  # 모델 학습
├── models/
│   └── resnet18_best.pth         # 학습된 모델
├── data/
│   └── casting_data/             # 데이터셋
└── requirements.txt
```

## 모델 성능

- **아키텍처**: ResNet18 (Transfer Learning)
- **정확도**: 97.5%
- **추론 속도**: ~0.05초/이미지
- **데이터셋**: 800장 (정상 400, 불량 400)

## 기술 스택

- PyTorch 2.0+
- Streamlit
- OpenCV, Matplotlib
- Grad-CAM (설명 가능한 AI)

## 라이센스

MIT License - 자세한 내용은 [LICENSE](LICENSE) 참조

## 연락처

- GitHub: [jhwwon](https://github.com/jhwwon)
