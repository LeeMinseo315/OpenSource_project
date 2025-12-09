import numpy as np
from algorithm_common import simulate_cvd_core

# [제1색맹] Protanopia 시뮬레이션 행렬
M_SIM_PROTAN = np.array([
    [0.0,      2.02344,  -2.52581],
    [0.0,      1.0,       0.0],
    [0.0,      0.0,       1.0]
]).T

def get_protan_params(input_alpha_g, input_alpha_b):
    """
    제1색맹용 파라미터 결정 로직
    - 특별한 튜닝 없이 입력값 그대로 사용 (사용자 요청 반영)
    """
    return input_alpha_g, input_alpha_b

def simulate_protan(img):
    return simulate_cvd_core(img, M_SIM_PROTAN)