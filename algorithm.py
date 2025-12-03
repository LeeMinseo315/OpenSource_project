import numpy as np

# Daltonization 상수 행렬
M_RGB2LMS = np.array([[17.8824, 43.5161, 4.11935], [3.45565, 27.1554, 3.86714], [0.0299566, 0.184309, 1.46709]]).T
M_LMS2RGB = np.linalg.inv(M_RGB2LMS)
M_SIM_PROTAN = np.array([[0, 2.02344, -2.52581], [0, 1, 0], [0, 0, 1]]).T
M_SHIFT = np.array([[0, 0, 0], [0.7, 1, 0], [0.7, 0, 1]]).T

def apply_daltonization(img, intensity=1.0):
    """
    반환값: (simulated_img_OFF, simulated_img_ON)
    - simulated_img_OFF: 보정 전 원본을 색맹이 봤을 때 (문제 상황)
    - simulated_img_ON:  보정된 결과를 색맹이 봤을 때 (최종 해결 화면)
    """
    img_float = img.astype(np.float32)
    
    # --- [Step 1] 원본에 대한 시뮬레이션 (OFF 모드용) ---
    lms = np.dot(img_float, M_RGB2LMS)
    lms_sim = np.dot(lms, M_SIM_PROTAN)
    rgb_sim_off = np.dot(lms_sim, M_LMS2RGB) # 이게 OFF 화면
    
    # --- [Step 2] 보정 알고리즘 적용 ---
    error = img_float - rgb_sim_off
    correction = np.dot(error, M_SHIFT) * intensity
    corrected_img_float = img_float + correction # 이게 정상인이 보는 보정 화면 (분홍색)
    
    # --- [Step 3] 보정된 결과에 대해 다시 시뮬레이션 (ON 모드용) ---
    # 핵심 추가: 보정된 이미지를 색맹이 보면 어떻게 보일까?
    lms_corr = np.dot(corrected_img_float, M_RGB2LMS)
    lms_sim_corr = np.dot(lms_corr, M_SIM_PROTAN)
    rgb_sim_on = np.dot(lms_sim_corr, M_LMS2RGB) # 이게 ON 화면 (최종 결과)
    
    # --- 결과 정리 (uint8 변환) ---
    simulated_output_OFF = np.clip(rgb_sim_off, 0, 255).astype(np.uint8)
    simulated_output_ON = np.clip(rgb_sim_on, 0, 255).astype(np.uint8)
    
    return simulated_output_OFF, simulated_output_ON