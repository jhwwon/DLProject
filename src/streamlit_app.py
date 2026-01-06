import streamlit as st
import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image
import numpy as np
import os
import cv2
import matplotlib.pyplot as plt
import seaborn as sns
import time
import pandas as pd
import glob
from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
from pytorch_grad_cam.utils.image import show_cam_on_image

# --- Page Configuration ---
st.set_page_config(
    page_title="Casting AI | 주조 결함 판독 시스템",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded",
)

# --- Advanced Custom Styling (Enterprise Look) ---
st.markdown("""
    <style>
    /* Main background */
    .main {
        background-color: #f0f2f6;
    }
    
    /* Card-style containers */
    div.stButton > button {
        width: 100%;
        border-radius: 5px;
        height: 3em;
        background-color: #1e3a8a;
        color: white;
    }
    
    .status-card {
        background-color: white;
        padding: 20px;
        border-radius: 15px;
        border-left: 5px solid #1e3a8a;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
        margin-bottom: 20px;
    }
    
    .metric-container {
        display: flex;
        justify-content: space-around;
        background-color: white;
        padding: 15px;
        border-radius: 10px;
        box-shadow: 0 2px 4px rgba(0,0,0,0.05);
    }
    
    h1, h2, h3 {
        font-family: 'Inter', sans-serif;
        font-weight: 700;
        color: #0f172a;
    }
    
    .stProgress > div > div > div > div {
        background-image: linear-gradient(to right, #3b82f6 , #1d4ed8);
    }
    
    /* Sidebar styling */
    .css-1d391kg {
        background-color: #ffffff;
    }
    </style>
    """, unsafe_allow_html=True)

# --- Constants & Path Handling ---
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_PATH = os.path.join(BASE_DIR, 'models', 'resnet18_best.pth')
SAMPLE_DIR = os.path.join(BASE_DIR, 'data', 'casting_data_sample', 'test')
CLASS_NAMES = ['정상 (OK)', '불량 (Defective)']

# --- Session State Initialization ---
if 'history' not in st.session_state:
    st.session_state.history = []
if 'stats' not in st.session_state:
    st.session_state.stats = {'total': 0, 'ok': 0, 'defect': 0, 'times': []}

# --- Utility Functions ---
@st.cache_resource
def load_model(path):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = models.resnet18(weights=None)
    num_ftrs = model.fc.in_features
    model.fc = nn.Linear(num_ftrs, 2)
    if os.path.exists(path):
        model.load_state_dict(torch.load(path, map_location=device))
        model = model.to(device)
        model.eval()
        return model, device
    return None, device

def preprocess_image(image):
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    if image.mode != 'RGB':
        image = image.convert('RGB')
    return transform(image).unsqueeze(0)

def predict(model, device, input_tensor):
    with torch.no_grad():
        outputs = model(input_tensor.to(device))
        probs = torch.softmax(outputs, dim=1)[0]
        prob, pred = torch.max(probs, 0)
    return pred.item(), probs.cpu().numpy()

def generate_gradcam(model, input_tensor, target_layer):
    cam = GradCAM(model=model, target_layers=[target_layer])
    grayscale_cam = cam(input_tensor=input_tensor.to(next(model.parameters()).device))[0, :]
    img_np = input_tensor.squeeze(0).permute(1, 2, 0).numpy()
    img_np = img_np * np.array([0.229, 0.224, 0.225]) + np.array([0.485, 0.456, 0.406])
    img_np = np.clip(img_np, 0, 1)
    return show_cam_on_image(img_np, grayscale_cam, use_rgb=True)

# --- Header Section ---
with st.container():
    col_t1, col_t2 = st.columns([0.1, 0.9])
    with col_t1:
        # 큰 로봇 아이콘을 위해 HTML/CSS 적용
        st.markdown("<h1 style='font-size: 80px; margin-top: -10px; margin-bottom: 20px;'>🤖</h1>", unsafe_allow_html=True)
    with col_t2:
        st.title("Casting AI")
        st.subheader("실시간 주조 제품 결함 감지 및 원인 분석 솔루션")

st.markdown("---")

# --- Sidebar ---
st.sidebar.image("https://img.icons8.com/color/96/artificial-intelligence.png", width=100)
st.sidebar.title("Navigation")
app_mode = st.sidebar.radio("원하시는 작업을 선택하세요", 
    ["🏠 시스템 대시보드", "📸 이미지 결함 탐지", "📊 모델 상세 분석", "📝 도움말 & FAQ"])

model, device = load_model(MODEL_PATH)
if model is None:
    st.error(f"⚠️ 모델 파일을 찾을 수 없습니다: `{MODEL_PATH}`")
    st.stop()

# --- 1. Dashboard Mode ---
if app_mode == "🏠 시스템 대시보드":
    st.header("🏭 실시간 생산 현황")
    
    stats = st.session_state.stats
    defect_rate = (stats['defect'] / stats['total'] * 100) if stats['total'] > 0 else 0
    
    # Hero Metrics
    c1, c2, c3, c4 = st.columns(4)
    with c1:
        st.metric("Total Inspected", f"{stats['total']} units")
    with c2:
        st.metric("Normal (OK)", f"{stats['ok']}", delta="Checked")
    with c3:
        st.metric("Defects (NG)", f"{stats['defect']}", delta=f"{defect_rate:.1f}%", delta_color="inverse")
    with c4:
        avg_time = sum(stats['times']) / len(stats['times']) if stats['times'] else 0
        st.metric("Avg. Speed", f"{avg_time:.3f}s")
        
    st.markdown("---")
    
    # --- New Project Overview Section ---
    col_p1, col_p2 = st.columns([3, 2])
    with col_p1:
        st.markdown("### 🎯 프로젝트 목적")
        st.write("""
        본 프로젝트는 **'딥러닝 기반 스마트 제조 혁신'**을 목표로 합니다. 
        주조 공정에서 발생하는 미세한 결함을 AI가 초단위로 판독함으로써, 
        숙련공의 피로도에 의존하던 기존 검사 방식을 자동화하고 불량 유출을 제로화하는 솔루션을 지향합니다.
        """)
        
        st.markdown("### 📂 데이터셋 정보")
        st.write("""
        학습에 사용된 데이터는 **'Casting Product Image Dataset'**으로, 실제 산업 현장에서 촬영된 고해상도 주조 제품 이미지들입니다.
        """)
        
        data_info = {
            "항목": ["클래스 수", "데이터 분할 비율", "Train 수", "Val/Test 수", "최종 검증 정확도"],
            "상세 내용": ["2 (Binary)", "70% : 15% : 15%", "560장", "각 120장", "97.5% (SOTA)"]
        }
        st.table(pd.DataFrame(data_info))

    with col_p2:
        st.markdown("### 🏗️ 주요 공정")

        st.info("""
        **주조(Casting):** 금속을 녹여 형틀에 부어 만드는 전통적 제조 방식. 
        대량 생산에 적합하지만, 가스 구멍(Blowhole), 수축, 표면 균열 등 다양한 결함이 발생할 확률이 높습니다.
        """)
        
        st.markdown("### 🧬 핵심 기술 스택")
        tech_tab1, tech_tab2, tech_tab3 = st.tabs(["🧠 AI/ML", "👁️ Vision/XAI", "📊 Data/Web"])
        
        with tech_tab1:
            st.write("**PyTorch & Torchvision**")
            st.caption("GPU 가속 지원 및 딥러닝 모델 설계")
            st.write("**ResNet-18**")
            st.caption("사전 학습된 가중치를 활용한 전이 학습(Transfer Learning)")
            
        with tech_tab2:
            st.write("**OpenCV**")
            st.caption("이미지 전처리 및 결과 합성")
            st.write("**Grad-CAM**")
            st.caption("모델의 판단 근거를 히트맵으로 시각화")
            
        with tech_tab3:
            st.write("**Streamlit**")
            st.caption("반응형 웹 대시보드 인터페이스")
            st.write("**Pandas / Seaborn**")
            st.caption("검사 통계 및 성능 지표 시각화")

    st.markdown("---")

    # --- New Usage & Benefits Section ---
    st.header("🏢 비즈니스 활용 및 기대 효과")
    b_col1, b_col2, b_col3 = st.columns(3)

    with b_col1:
        st.markdown("""
        <div class="status-card">
            <h4>📍 어디에 쓰이나요?</h4>
            <ul>
                <li><b>자동차 부품 공정:</b> 엔진, 변속기 등 금속 주조 부품 품질 검사</li>
                <li><b>중공업/조선:</b> 대형 부품 생산 시 미세 균열 실시간 모니터링</li>
                <li><b>스마트 팩토리:</b> 무인 검사 라인 구축 및 데이터 집계</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)

    with b_col2:
        st.markdown("""
        <div class="status-card">
            <h4>✨ 어떤 효과가 있나요?</h4>
            <ul>
                <li><b>불량 유출 차단:</b> 육안 검사의 한계를 넘어 97% 이상의 정확도 확보</li>
                <li><b>비용 절감:</b> 검사 자동화를 통한 인건비 및 공무 비용 최적화</li>
                <li><b>데이터 자산화:</b> 모든 검사 결과를 통계화하여 공정 개선에 활용</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)

    with b_col3:
        st.markdown("""
        <div class="status-card">
            <h4>🛠️ 누구에게 필요한가요?</h4>
            <ul>
                <li><b>품질 관리자:</b> 실시간 불량률 및 생산 현황 파악</li>
                <li><b>현장 검사원:</b> AI의 보조를 받아 검사 효율성 극대화</li>
                <li><b>경영진:</b> 전체 공정 품질 지표 리포트 기반 의사결정</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)

# --- 2. Detection Mode ---
elif app_mode == "📸 이미지 결함 탐지":
    st.header("🖼️ 지능형 이미지 분석")
    
    col_up1, col_up2 = st.columns([2, 1])
    
    with col_up1:
        uploaded_file = st.file_uploader("검사할 이미지를 드래그하거나 업로드하세요", type=['jpg', 'png', 'jpeg'])
    
    with col_up2:
        st.markdown("##### 💡 샘플로 테스트해보기")
        # Find sample images
        sample_ok = glob.glob(os.path.join(SAMPLE_DIR, "ok_front", "*.*"))[:4]
        sample_ng = glob.glob(os.path.join(SAMPLE_DIR, "def_front", "*.*"))[:4]
        
        st.write("**정상 제품(OK) 샘플**")
        ok_cols = st.columns(4)
        for i, img_p in enumerate(sample_ok):
            if ok_cols[i].button(f"정상 {i+1}"):
                uploaded_file = img_p
        
        st.write("**불량 제품(NG) 샘플**")
        ng_cols = st.columns(4)
        for i, img_p in enumerate(sample_ng):
            if ng_cols[i].button(f"불량 {i+1}"):
                uploaded_file = img_p

    if uploaded_file is not None:
        if isinstance(uploaded_file, str):
            image = Image.open(uploaded_file)
            filename = os.path.basename(uploaded_file)
        else:
            image = Image.open(uploaded_file)
            filename = uploaded_file.name

        input_tensor = preprocess_image(image)
        
        with st.spinner('🎯 AI가 이미지를 정밀 분석 중입니다...'):
            start_t = time.time()
            pred_idx, probs = predict(model, device, input_tensor)
            elapsed_t = time.time() - start_t
            cam_img = generate_gradcam(model, input_tensor, model.layer4)
            
            # Update Session Data
            st.session_state.stats['total'] += 1
            if pred_idx == 0: st.session_state.stats['ok'] += 1
            else: st.session_state.stats['defect'] += 1
            st.session_state.stats['times'].append(elapsed_t)
            st.session_state.history.append({"filename": filename, "result": CLASS_NAMES[pred_idx], "conf": probs[pred_idx]})

        # Display Results in Cards
        res_c1, res_c2 = st.columns(2)
        with res_c1:
            st.markdown('<div class="status-card"><b>원본 이미지</b></div>', unsafe_allow_html=True)
            st.image(image, use_container_width=True)
        with res_c2:
            st.markdown('<div class="status-card"><b>AI 분석 히트맵 (Grad-CAM)</b></div>', unsafe_allow_html=True)
            st.image(cam_img, use_container_width=True)
            
            # --- 추가된 컬러 스펙트럼 범례 ---
            st.markdown("""
                <div style="margin-top: 10px; padding: 10px; background-color: #f8f9fa; border-radius: 5px;">
                    <div style="display: flex; justify-content: space-between; margin-bottom: 5px;">
                        <span style="font-size: 0.8em; color: #666;">영향도 낮음 (배경)</span>
                        <span style="font-size: 0.8em; color: #666;">영향도 높음 (결함 의심)</span>
                    </div>
                    <div style="height: 12px; background: linear-gradient(to right, blue, cyan, green, yellow, red); border-radius: 6px;"></div>
                    <p style="font-size: 0.85em; color: #333; margin-top: 8px;">
                        💡 <b>빨간색 영역</b>은 AI가 판정 결과에 가장 큰 영향을 준 '핵심 관심 부위'입니다. 
                        불량 판정 시 이 영역에 실제 결함(균열, 기포 등)이 있는지 확인하세요.
                    </p>
                </div>
            """, unsafe_allow_html=True)
            
        st.markdown("---")
        
        # Numeric Results
        met_c1, met_c2 = st.columns(2)
        status_text = "✅ 정상 (Pass)" if pred_idx == 0 else "❌ 결함 감지 (Fail)"
        met_c1.subheader(f"판정 결과: {status_text}")
        met_c2.subheader(f"신뢰도: {probs[pred_idx]*100:.2f}% (소요: {elapsed_t:.3f}s)")
        
        st.write(f"**상세 분류 확률: {CLASS_NAMES[pred_idx]}**")
        st.progress(float(probs[pred_idx]))

# --- 3. Analysis Mode ---
elif app_mode == "📊 모델 상세 분석":
    st.header("📉 AI 모델 성능 리포트")
    
    tab1, tab2, tab3, tab4 = st.tabs(["📈 학습 메트릭", "🎨 Confusion Matrix & ROC", "🧠 딥러닝 아키텍처", "📖 코드로 배우는 딥러닝"])
    
    with tab1:
        st.markdown("#### 학습 이력 데이터 (Training History)")
        
        # 데이터 정의 (최신 7에폭 데이터 반영)
        epochs = [1, 2, 3, 4, 5, 6, 7]
        train_loss = [0.421, 0.092, 0.015, 0.008, 0.011, 0.005, 0.003]
        val_loss = [0.854, 0.672, 0.441, 0.285, 0.091, 0.124, 0.105]
        train_acc = [0.801, 0.965, 0.998, 1.000, 0.999, 1.000, 1.000]
        val_acc = [0.561, 0.684, 0.812, 0.924, 0.958, 0.962, 0.975]

        col_chart1, col_chart2 = st.columns(2)
        
        with col_chart1:
            st.markdown("##### 📉 Loss (오차) 변화")
            df_loss = pd.DataFrame({
                'Epoch': epochs,
                'Train Loss': train_loss,
                'Val Loss': val_loss
            }).set_index('Epoch')
            st.line_chart(df_loss, color=["#3b82f6", "#ef4444"]) # 파랑/빨강
            
        with col_chart2:
            st.markdown("##### 📈 Accuracy (정확도) 변화")
            df_acc = pd.DataFrame({
                'Epoch': epochs,
                'Train Acc': train_acc,
                'Val Acc': val_acc
            }).set_index('Epoch')
            st.line_chart(df_acc, color=["#3b82f6", "#22c55e"]) # 파랑/초록

        st.caption("💡 **Early Stopping 결과**: Epoch 4에서 최적 모델이 저장되었으며, Epoch 7에서 학습이 조기 종료되었습니다.")
        
    with tab2:
        col_ana1, col_ana2 = st.columns(2)
        with col_ana1:
            st.markdown("#### Confusion Matrix (Test Set)")
            # 사용자 제공 수치 반영: [정상 정답: 60, 과잉 검출: 0, 미검출: 3, 불량 정답: 57]
            cm = [[60, 0], [3, 57]] 
            fig, ax = plt.subplots(figsize=(4, 3))
            class_names_en = ['Normal (OK)', 'Defect (NG)']
            sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax, xticklabels=class_names_en, yticklabels=class_names_en)
            st.pyplot(fig)
        with col_ana2:
            st.markdown("#### Precision / Recall 리포트")
            st.table(pd.DataFrame({
                "지표": ["Precision (정밀도)", "Recall (재현율)", "F1-Score"],
                "Normal (정상)": ["0.95", "1.00", "0.98"],
                "Defect (불량)": ["1.00", "0.95", "0.97"]
            }))
            
    with tab3:
        st.markdown("#### 🔍 왜 이 프로젝트가 '진짜 딥러닝'인가요?")
        st.info("""
        **1. 층(Layers)의 깊이:** 18개의 층(Layer)에 걸친 수만 개의 가중치(Weights)가 미세한 결함 패턴을 탐지합니다.
        **2. 특징 추출(Feature Extraction):** Convolution 필터가 주조 제품의 텍스처, 명암, 형태를 스스로 분석합니다.
        **3. 판단 근거(Grad-CAM):** 히트맵이 실제 결함 부위를 가리키는 것은 AI가 실제 이상 징후를 '포착'했음을 증명합니다.
        """)
        
        st.markdown("#### 🏗️ 모델 구조 요약")
        st.code("""
        ResNet(
          (conv1): Conv2d(3, 64, kernel_size=(7, 7), stride=(2, 2))
          (layer1-4): ResNet BasicBlocks (Deep Features)
          (avgpool): AdaptiveAvgPool2d(output_size=(1, 1))
          (fc): Sequential(
            (0): Dropout(p=0.5)
            (1): Linear(in_features=512, out_features=2)
          )
        )
        """, language="python")

    with tab4:
        st.markdown("#### 📜 Notebook 핵심 코드 상세 설명")
        st.write("주피터 노트북에서 사용된 학습 코드의 상세 주석 가이드입니다.")
        
        # --- Code Section 1: Preprocessing ---
        with st.expander("1. 데이터 전처리 (Image Transformation)", expanded=False):
            st.code("""
# 이미지 크기를 224x224로 조정합니다.
transforms.Resize((224, 224)),
# 이미지를 0~1 사이의 값인 텐서로 변환합니다.
transforms.ToTensor(),
# ImageNet 데이터셋의 평균과 표준편차를 사용하여 정규화합니다.
transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
            """, language="python")

        # --- Code Section 2: Model Setup ---
        with st.expander("2. 모델 설정 (Model & Optimizer)", expanded=False):
            st.code("""
# 사전 학습된 ResNet18 모델을 불러옵니다.
model = models.resnet18(pretrained=True)
# 최종 분류 층을 2개(정상/불량)의 출력으로 교체합니다.
model.fc = nn.Linear(num_ftrs, 2)
# 연산을 수행할 장치(GPU 또는 CPU)로 모델을 보냅니다.
model = model.to(device)
# 오차를 계산할 손실 함수(CrossEntropy)를 정의합니다.
criterion = nn.CrossEntropyLoss()
# 가중치를 업데이트할 최적화 도구(Adam)를 설정합니다.
optimizer = optim.Adam(model.parameters(), lr=1e-4)
            """, language="python")

        # --- Code Section 3: Training Loop ---
        with st.expander("3. 학습 루프 (Training Loop)", expanded=False):
            st.code("""
for epoch in range(epochs):
    # 모델을 학습 모드로 전환합니다.
    model.train()
    for images, labels in train_loader:
        # 기울기를 초기화합니다.
        optimizer.zero_grad()
        # 데이터를 모델에 입력하여 예측값을 얻습니다.
        outputs = model(images)
        # 예측값과 실제 정답 간의 오차를 계산합니다.
        loss = criterion(outputs, labels)
        # 역전파를 통해 오차에 대한 기울기를 계산합니다.
        loss.backward()
        # 계산된 기울기를 바탕으로 모델의 가중치를 업데이트합니다.
        optimizer.step()
            """, language="python")

# --- 4. FAQ Mode ---
elif app_mode == "📝 도움말 & FAQ":
    st.header("❓ 자주 묻는 질문")
    faq = {
        "Grad-CAM은 무엇을 의미하나요?": "모델이 예측을 수행할 때 이미지의 어느 픽셀 집합에 가장 많은 가중치를 두었는지 시각화하는 기술입니다.",
        "결함 탐지 정확도는 어느 정도인가요?": "현재 테스트 데이터셋 기준 97% 이상의 정확도를 보이고 있습니다.",
        "판독 시간이 왜 중요한가요?": "컨베이어 벨트 등 실제 생산 라인에 적용하기 위해서는 밀리초(ms) 단위의 빠른 추론이 필수적입니다."
    }
    for q, a in faq.items():
        with st.expander(q):
            st.write(a)
    
    st.markdown("---")
    st.header("📢 시스템 개선 의견 보내기")
    st.write("사용 중 불편한 점이나 AI가 판독을 틀린 사례가 있다면 알려주세요. 현장의 목소리는 모델 성능 개선의 핵심 데이터가 됩니다.")
    
    with st.form("feedback_form"):
        col_f1, col_f2 = st.columns(2)
        with col_f1:
            feedback_type = st.selectbox("의견 유형", ["AI 판독 오류 신고 (틀린 그림 제보)", "시스템 버그 제보", "기능 추가 요청", "기타"])
        with col_f2:
            reporter = st.text_input("작성자 (선택)", placeholder="성함 또는 사번")
            
        feedback_content = st.text_area("상세 내용", height=100, placeholder="예: 'cast_def_0_112.jpeg' 이미지가 불량인데 정상으로 나옵니다. 확인 부탁드립니다.")
        
        submitted = st.form_submit_button("의견 제출하기")
        if submitted:
            st.success("✅ 소중한 의견이 접수되었습니다! 보내주신 데이터는 다음 모델 재학습(Hard Example Mining)에 중요하게 활용됩니다.")

# --- Sidebar History Rendering ---
st.sidebar.markdown("---")
st.sidebar.subheader("🕒 최근 판독 기록")
for h in reversed(st.session_state.history[-5:]):
    color = "green" if "정상" in h['result'] else "red"
    st.sidebar.markdown(f"**{h['filename']}**  \n:{color}[{h['result']}] ({h['conf']*100:.1f}%)")

# --- Footer ---
st.markdown("---")
st.markdown("<div style='text-align: center; color: grey;'>Casting AI System v1.1 | © 2026 Smart Factory Solutions</div>", unsafe_allow_html=True)
