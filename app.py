import streamlit as st
import cv2
import numpy as np
import tempfile
from PIL import Image
from algorithm import apply_daltonization 

# --- [1] 페이지 기본 설정 ---
st.set_page_config(page_title="Spectrum Shift", page_icon="🎨", layout="wide")

# --- [2] 사이드바: 공통 설정 및 모드 전환 ---
with st.sidebar:
    st.header("🎛️ 분석 모드 선택")
    # '웹캠' 대신 '동영상 파일'로 변경
    app_mode = st.radio(
        "기능을 선택하세요:",
        ("📁 이미지 분석 (Image)", "🎬 동영상 분석 (Video)")
    )
    
    st.divider()
    
    st.header("⚙️ 보정 설정")
    # 색맹 타입 선택
    type_option = st.radio(
        "색각 이상 타입",
        ("제2 적록색맹 (녹색맹, Deutan)", "제1 적록색맹 (적색맹, Protan)")
    )
    cb_type = "deutan" if "제2" in type_option else "protan"
    
    # 보정 강도 슬라이더
    intensity = st.slider("보정 강도 (Intensity)", 0.0, 2.0, 1.0, 0.1)
    
    st.divider()
    
    # 보기 모드 (공통)
    view_mode = st.radio(
        "오른쪽 화면 모드",
        ("OFF (적록색맹 시점)", "ON (보정 후)")
    )
    is_correction_mode = True if "ON" in view_mode else False

# --- [3] 메인 타이틀 ---
st.title(" 적록색맹을 위한 실시간 색각 보정 시스템")
st.markdown("""
**이미지**나 **동영상** 파일을 업로드하여 Daltonization 알고리즘의 효과를 검증합니다.
""")
st.divider()

# ==========================================
# [모드 1] 이미지 파일 분석
# ==========================================
if app_mode == "📁 이미지 분석 (Image)":
    st.subheader("1️⃣ 이미지 정밀 분석")
    uploaded_file = st.file_uploader("이미지 파일 업로드 (JPG, PNG)", type=['jpg', 'png', 'jpeg'])

    if uploaded_file is not None:
        # 이미지 읽기
        image = Image.open(uploaded_file)
        img_array = np.array(image.convert('RGB'))
        
        # 알고리즘 적용
        sim_off, sim_on = apply_daltonization(img_array, cb_type, intensity)
        
        # 화면 배치
        col1, col2 = st.columns(2)
        with col1:
            st.caption(" 원본 (Normal)")
            st.image(img_array, use_container_width=True)
        with col2:
            if not is_correction_mode:
                st.caption(f" {type_option} 시각 (Simulation)")
                st.image(sim_off, use_container_width=True)
            else:
                st.caption(f" 보정된 시각 (Correction)")
                st.image(sim_on, use_container_width=True)

# ==========================================
# [모드 2] 동영상 파일 분석 (변경된 부분)
# ==========================================
elif app_mode == "🎬 동영상 분석 (Video)":
    st.subheader("2 동영상 검증")
    st.info("영상을 업로드하세요. (mp4, mov, avi)")
    
    uploaded_video = st.file_uploader("동영상 파일 업로드", type=['mp4', 'mov', 'avi'])

    if uploaded_video is not None:
        # 1. 임시 파일로 저장 (OpenCV가 파일을 읽으려면 실제 경로가 필요함)
        tfile = tempfile.NamedTemporaryFile(delete=False) 
        tfile.write(uploaded_video.read())
        
        # 2. OpenCV로 비디오 열기
        cap = cv2.VideoCapture(tfile.name)
        
        # 3. 화면 레이아웃 잡기 (플레이어 영역)
        col1, col2 = st.columns(2)
        with col1:
            st.caption("원본 영상")
            frame_placeholder1 = st.empty() # 빈 공간 확보
        with col2:
            st.caption(f" 보정 결과 ({'보정 ON' if is_correction_mode else '시뮬레이션'})")
            frame_placeholder2 = st.empty() # 빈 공간 확보

        stop_button = st.button("⏹️ 재생 중지")

        # 4. 프레임 반복 처리 (재생 루프)
        while cap.isOpened() and not stop_button:
            ret, frame = cap.read()
            if not ret:
                break # 영상 끝나면 종료
            
            # BGR -> RGB 변환
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            
            # 알고리즘 적용
            sim_off, sim_on = apply_daltonization(frame_rgb, cb_type, intensity)
            
            # 오른쪽 화면 결정
            right_frame = sim_on if is_correction_mode else sim_off

            # 화면 업데이트 (마치 동영상처럼 보임)
            frame_placeholder1.image(frame_rgb, channels="RGB", use_container_width=True)
            frame_placeholder2.image(right_frame, channels="RGB", use_container_width=True)
            
        cap.release()
        st.success("영상 재생이 완료되었습니다.")