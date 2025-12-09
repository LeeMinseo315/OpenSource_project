import streamlit as st
from streamlit_webrtc import webrtc_streamer, VideoTransformerBase, WebRtcMode
import cv2
import numpy as np
import av
from PIL import Image
from algorithm import apply_daltonization 

# --- [1] 페이지 기본 설정 ---
st.set_page_config(page_title="ColorBlind Aid Pro", page_icon="🎨", layout="wide")

# --- [2] 사이드바: 공통 설정 및 모드 전환 ---
with st.sidebar:
    st.header("🎛️ 모드 선택")
    # 여기서 '이미지' vs '웹캠'을 선택합니다 (버튼 역할)
    app_mode = st.radio(
        "사용할 기능을 선택하세요:",
        ("📁 이미지 파일 업로드", "📹 실시간 웹캠")
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
st.title(" 적록색맹을 위한 실시간 색각 보정 시스템 ")
st.markdown(f"현재 모드: **{app_mode}**")
st.divider()

# ==========================================
# [모드 1] 이미지 파일 업로드 기능
# ==========================================
if app_mode == "📁 이미지 파일 업로드":
    st.info("👇 아래 버튼을 눌러 이미지를 업로드하세요.")
    uploaded_file = st.file_uploader("이미지 파일 (JPG, PNG)", type=['jpg', 'png', 'jpeg'])

    if uploaded_file is not None:
        # 이미지 읽기
        image = Image.open(uploaded_file)
        img_array = np.array(image.convert('RGB'))
        
        # 알고리즘 적용
        sim_off, sim_on = apply_daltonization(img_array, cb_type, intensity)
        
        # 화면 배치 (2단 컬럼)
        col1, col2 = st.columns(2)
        with col1:
            st.subheader(" 원본 ")
            st.image(img_array, use_container_width=True)
        with col2:
            if not is_correction_mode:
                st.subheader(f" {type_option} 시각 (Before)")
                st.image(sim_off, use_container_width=True)
            else:
                st.subheader(f" 보정된 시각 (After)")
                st.image(sim_on, use_container_width=True)

# ==========================================
# [모드 2] 실시간 웹캠 기능
# ==========================================
elif app_mode == "📹 실시간 웹캠":
    st.info("👇 아래 **START** 버튼을 누르면 웹캠이 켜집니다.")

    # 비디오 처리기 클래스
    class VideoProcessor(VideoTransformerBase):
        def __init__(self):
            self.cb_type = "deutan"
            self.intensity = 1.0
            self.is_correction_mode = False

        def recv(self, frame):
            img = frame.to_ndarray(format="bgr24")
            img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            
            # 알고리즘 적용
            sim_off, sim_on = apply_daltonization(img_rgb, self.cb_type, self.intensity)
            
            if self.is_correction_mode:
                right_img = sim_on
            else:
                right_img = sim_off
                
            # 화면 합치기
            combined = np.hstack((img_rgb, right_img))
            return av.VideoFrame.from_ndarray(combined, format="rgb24")

    # 웹캠 스트리머 실행
    ctx = webrtc_streamer(
    key="mode-switch-example",
    mode=WebRtcMode.SENDRECV,
    # rtc_configuration 줄을 아예 삭제했습니다.
    video_processor_factory=VideoProcessor,
    media_stream_constraints={"video": True, "audio": False},
    async_processing=True,
    )

    # 실시간 값 전달
    if ctx.video_transformer:
        ctx.video_transformer.cb_type = cb_type
        ctx.video_transformer.intensity = intensity
        ctx.video_transformer.is_correction_mode = is_correction_mode