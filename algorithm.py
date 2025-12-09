import numpy as np

# RGB → LMS 변환 행렬
M_RGB2LMS = np.array([
    [17.8824, 43.5161, 4.11935],
    [3.45565, 27.1554, 3.86714],
    [0.0299566, 0.184309, 1.46709]
]).T

# LMS → RGB 역행렬
M_LMS2RGB = np.linalg.inv(M_RGB2LMS)

# Protanopia(제1 적록색맹) 시뮬레이션 행렬
M_SIM_PROTAN = np.array([
    [0.0,      2.02344,  -2.52581],
    [0.0,      1.0,       0.0],
    [0.0,      0.0,       1.0]
]).T

# Deuteranopia(제2 적록색맹) 시뮬레이션 행렬
M_SIM_DEUTAN = np.array([
    [1.0,      0.0,       0.0],
    [0.494207, 0.0,       1.24827],
    [0.0,      0.0,       1.0]
]).T


def _simulate_cvd(rgb_img: np.ndarray, cb_type: str) -> np.ndarray:
    """
    색각 유형에 따라 CVD 시뮬레이션 (Protan / Deutan)
    rgb_img: (H, W, 3), uint8, RGB
    """
    img_float = rgb_img.astype(np.float32)

    # RGB → LMS
    lms = np.dot(img_float, M_RGB2LMS)

    # 유형 선택
    if cb_type == "protan":
        M_SIM = M_SIM_PROTAN
    elif cb_type == "deutan":
        M_SIM = M_SIM_DEUTAN
    else:
        raise ValueError(f"지원하지 않는 cb_type: {cb_type}")

    # LMS에서 색각 이상 시뮬레이션
    lms_sim = np.dot(lms, M_SIM)

    # LMS → RGB
    rgb_sim = np.dot(lms_sim, M_LMS2RGB)
    rgb_sim_uint8 = np.clip(rgb_sim, 0, 255).astype(np.uint8)
    return rgb_sim_uint8


def apply_daltonization(
    img: np.ndarray,
    cb_type: str = "deutan",
    intensity: float = 1.0,
    alpha_g: float = 0.9, # 함수 호출 시 넘어오는 기본값 사용
    alpha_b: float = 0.7, # 함수 호출 시 넘어오는 기본값 사용
):
    """
    Daltonization 적용 함수
    """
    
    # 1) 보정 전 시뮬레이션 (OFF 화면)
    sim_off = _simulate_cvd(img, cb_type=cb_type)

    img_f = img.astype(np.float32)
    sim_f = sim_off.astype(np.float32)

    # 2) 에러 계산 (원본 - 시뮬레이션)
    error = img_f - sim_f

    # ▼▼▼ [수정됨] 제2색맹(Deutan)만 최적값 적용 / 제1색맹은 건드리지 않음 ▼▼▼
    if cb_type == "deutan":
        # 실험을 통해 얻은 제2색맹 최적값
        alpha_g = 0.5
        alpha_b = 0.9
    
    # (참고) cb_type == "protan"일 때는 위에서 받은 인자(alpha_g, alpha_b)를 그대로 사용함


    # 3) M_SHIFT 생성
    M_SHIFT = np.array([
        [0.0,      0.0, 0.0],
        [alpha_g,  1.0, 0.0],
        [alpha_b,  0.0, 1.0]
    ]).T

    # 에러를 다른 채널로 옮겨 싣기
    correction = np.dot(error, M_SHIFT) * intensity

    # 4) 보정된 RGB (정상인이 보는 화면)
    corr_rgb_f = img_f + correction
    corr_rgb = np.clip(corr_rgb_f, 0, 255).astype(np.uint8)

    # 5) 보정된 RGB를 다시 색각이상자가 봤을 때 시뮬레이션 (ON 화면)
    sim_on = _simulate_cvd(corr_rgb, cb_type=cb_type)

    return sim_off, sim_on