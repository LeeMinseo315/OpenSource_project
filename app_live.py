import streamlit as st
from streamlit_webrtc import webrtc_streamer, VideoTransformerBase, WebRtcMode
import cv2
import numpy as np
import av
from algorithm import apply_daltonization # 기존 알고리즘 재사용

# 1. 페이지 설정
st.set_page_config(page_title="ColorBlind Aid Live", layout="wide")
st.title("📹 실시간 색각 보정 라이브 (Live Cam)")

# 2. 사이드바 설정 (제어 패널)
with st.sidebar:
    st.header("⚙️ 설정 (Settings)")
    
    # 색맹 타입 선택
    type_option = st.radio(
        "색각 이상 타입 선택",
        ("제2 적록색맹 (녹색맹, Deutan)", "제1 적록색맹 (적색맹, Protan)")
    )
    cb_type = "deutan" if "제2" in type_option else "protan"
    
    # 보정 강도
    intensity = st.slider("보정 강도", 0.0, 2.0, 1.0, 0.1)
    
    st.divider()
    
    # 보기 모드 선택
    view_mode = st.radio(
        "오른쪽 화면 모드",
        ("⛔ 문제 상황 (Simulation)", "✅ 해결책 (Correction)")
    )
    is_correction_mode = True if "해결책" in view_mode else False

# 3. 비디오 프로세서 클래스 정의 (핵심!)
# 들어오는 영상 프레임을 하나씩 받아서 변환하고 내보내는 공장 역할입니다.
class VideoProcessor(VideoTransformerBase):
    def __init__(self):
        # 초기 설정값
        self.cb_type = "deutan"
        self.intensity = 1.0
        self.is_correction_mode = False

    def recv(self, frame):
        # 1) 웹캠에서 프레임 받기 (NumPy 배열로 변환)
        img = frame.to_ndarray(format="bgr24")
        
        # 2) OpenCV 처리를 위해 RGB로 변환
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        
        # 3) 우리가 만든 알고리즘 적용!
        sim_off, sim_on = apply_daltonization(img_rgb, self.cb_type, self.intensity)
        
        # 4) 모드에 따라 오른쪽 화면 결정
        if self.is_correction_mode:
            right_img = sim_on
        else:
            right_img = sim_off
            
        # 5) 화면 합치기 (왼쪽: 원본 / 오른쪽: 변환본)
        # 웹상에서는 너무 크면 느려질 수 있으므로 리사이징 (선택 사항)
        h, w, c = img_rgb.shape
        # 화면 반반 붙이기 (numpy hstack)
        combined = np.hstack((img_rgb, right_img))
        
        # 6) 다시 BGR로 변환해서 내보내기 (Streamlit은 RGB로 보여주지만 av는 포맷 맞춤 필요)
        # 보통 av.VideoFrame으로 나갈 때는 RGB 포맷을 유지해도 되지만, 색이 이상하면 BGR 변환 필요
        # 여기서는 RGB 상태로 내보냅니다.
        return av.VideoFrame.from_ndarray(combined, format="rgb24")

# 4. 메인 화면에 스트리머 배치
ctx = webrtc_streamer(
    key="example",
    mode=WebRtcMode.SENDRECV,
    rtc_configuration={"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]},
    video_processor_factory=VideoProcessor,
    media_stream_constraints={"video": True, "audio": False},
    async_processing=True,
)

# 5. 실시간 제어 값 전달
# 사용자가 슬라이더를 움직이면 비디오 프로세서에게 값을 전달합니다.
if ctx.video_transformer:
    ctx.video_transformer.cb_type = cb_type
    ctx.video_transformer.intensity = intensity
    ctx.video_transformer.is_correction_mode = is_correction_mode

st.markdown("---")
st.info("💡 **Tip:** 위 **[START]** 버튼을 누르면 웹캠이 켜집니다. (브라우저 권한 허용 필요)")