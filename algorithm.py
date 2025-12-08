# algorithm.py
import numpy as np

import numpy as np

# RGB → LMS 변환 행렬
M_RGB2LMS = np.array([
    [17.8824, 43.5161, 4.11935],
    [3.45565, 27.1554, 3.86714],
    [0.0299566, 0.184309, 1.46709]
]).T

# LMS → RGB 역행렬
M_LMS2RGB = np.linalg.inv(M_RGB2LMS)

# Protanopia(제1 적록색맹) 시뮬레이션 행렬 (L-cone 손실)
M_SIM_PROTAN = np.array([
    [0.0,      2.02344,  -2.52581],
    [0.0,      1.0,       0.0],
    [0.0,      0.0,       1.0]
]).T


# Deuteranopia (제2 적록색맹)
M_SIM_DEUTAN = np.array([
    [1.0,      0.0,       0.0],
    [0.494207, 0.0,       1.24827],
    [0.0,      0.0,       1.0]
]).T

# 오류를 보정해서 정상인에게 더 구분 잘 되게 만드는 행렬
M_SHIFT = np.array([
    [0.0, 0.0, 0.0],
    [0.7, 1.0, 0.0],
    [0.7, 0.0, 1.0]
]).T


def apply_daltonization(img, cb_type: str = "protan", intensity: float = 1.0):
    """
    Daltonization 적용 함수

    Parameters
    ----------
    img : np.ndarray
        원본 RGB 이미지 (H, W, 3, uint8)
    cb_type : str
        "protan" (제1 적록색맹) 또는 "deutan" (제2 적록색맹)
    intensity : float
        보정 강도 (0.0 ~ 1.0 이상)

    Returns
    -------
    simulated_output_OFF : np.ndarray
        보정 전 원본을 색맹(선택한 cb_type)이 봤을 때의 시뮬레이션 이미지
    simulated_output_ON : np.ndarray
        보정된 결과를 색맹(선택한 cb_type)이 봤을 때의 시뮬레이션 이미지
    """
    img_float = img.astype(np.float32)

    # 색각 유형에 따른 시뮬레이션 행렬 선택
    if cb_type == "protan":
        M_SIM = M_SIM_PROTAN
    elif cb_type == "deutan":
        M_SIM = M_SIM_DEUTAN
    else:
        raise ValueError(f"지원하지 않는 cb_type입니다: {cb_type}")

    # --- [Step 1] 원본에 대한 시뮬레이션 (OFF 모드용) ---
    lms = np.dot(img_float, M_RGB2LMS)
    lms_sim = np.dot(lms, M_SIM)
    rgb_sim_off = np.dot(lms_sim, M_LMS2RGB)  # OFF 화면

    # --- [Step 2] 보정 알고리즘 적용 ---
    error = img_float - rgb_sim_off
    correction = np.dot(error, M_SHIFT) * intensity
    corrected_img_float = img_float + correction  # 정상인이 보는 보정 화면

    # --- [Step 3] 보정된 결과에 대해 다시 시뮬레이션 (ON 모드용) ---
    lms_corr = np.dot(corrected_img_float, M_RGB2LMS)
    lms_sim_corr = np.dot(lms_corr, M_SIM)
    rgb_sim_on = np.dot(lms_sim_corr, M_LMS2RGB)  # ON 화면 (최종 결과)

    # --- 결과 정리 (uint8 변환) ---
    simulated_output_OFF = np.clip(rgb_sim_off, 0, 255).astype(np.uint8)
    simulated_output_ON = np.clip(rgb_sim_on, 0, 255).astype(np.uint8)

    return simulated_output_OFF, simulated_output_ON
