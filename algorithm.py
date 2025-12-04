# algorithm.py
import numpy as np

# === 공통 RGB <-> LMS 변환 행렬 ===
M_RGB2LMS = np.array([
    [17.8824, 43.5161, 4.11935],
    [3.45565, 27.1554, 3.86714],
    [0.0299566, 0.184309, 1.46709]
]).T

M_LMS2RGB = np.linalg.inv(M_RGB2LMS)

# === 색각 유형별 시뮬레이션 행렬 ===
# 제1 적록색맹 (Protanopia)
M_SIM_PROTAN = np.array([
    [0.0,     2.02344, -2.52581],
    [0.0,     1.0,      0.0    ],
    [0.0,     0.0,      1.0    ]
]).T

# 제2 적록색맹 (Deuteranopia) – 예시용 행렬
M_SIM_DEUTAN = np.array([
    [1.0,      0.0,       0.0    ],
    [0.494207, 0.0,       1.24827],
    [0.0,      0.0,       1.0    ]
]).T

# === 에러 스프레드(보정) 행렬: 타입별로 다르게 줄 수도 있음 ===
M_SHIFT_PROTAN = np.array([
    [0.0, 0.0, 0.0],
    [0.7, 1.0, 0.0],
    [0.7, 0.0, 1.0]
]).T

M_SHIFT_DEUTAN = np.array([
    [1.0, 0.7, 0.0],
    [0.0, 1.0, 0.0],
    [0.0, 0.7, 1.0]
]).T


def apply_daltonization(img, cb_type="protan", intensity=1.0):
    """
    cb_type: "protan" (제1 적록색맹) 또는 "deutan" (제2 적록색맹)
    반환값: (simulated_img_OFF, simulated_img_ON)
      - OFF : 보정 전, 해당 색각 유형이 본 원본
      - ON  : 보정 후, 해당 색각 유형이 본 화면
    """
    img_float = img.astype(np.float32)

    # 1) 타입에 따라 행렬 선택
    if cb_type == "deutan":
        M_SIM = M_SIM_DEUTAN
        M_SHIFT = M_SHIFT_DEUTAN
    else:
        M_SIM = M_SIM_PROTAN
        M_SHIFT = M_SHIFT_PROTAN

    # 2) 원본 → LMS
    lms = np.dot(img_float, M_RGB2LMS)

    # 3) 색각 결손 시뮬레이션 (OFF 화면용)
    lms_sim = np.dot(lms, M_SIM)
    rgb_sim_off = np.dot(lms_sim, M_LMS2RGB)

    # 4) 에러 기반 보정 (정상인 기준 보정)
    error = img_float - rgb_sim_off
    correction = np.dot(error, M_SHIFT) * intensity
    corrected_img_float = img_float + correction

    # 5) 보정된 결과를 다시 색각 결손으로 시뮬레이션 (ON 화면용)
    lms_corr = np.dot(corrected_img_float, M_RGB2LMS)
    lms_sim_corr = np.dot(lms_corr, M_SIM)
    rgb_sim_on = np.dot(lms_sim_corr, M_LMS2RGB)

    # 6) 결과 정리
    simulated_output_OFF = np.clip(rgb_sim_off, 0, 255).astype(np.uint8)
    simulated_output_ON = np.clip(rgb_sim_on, 0, 255).astype(np.uint8)

    return simulated_output_OFF, simulated_output_ON
