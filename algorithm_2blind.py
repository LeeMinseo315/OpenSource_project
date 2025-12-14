import numpy as np
from algorithm_common import simulate_cvd_core

# [제2색맹] Deuteranopia 시뮬레이션 행렬
M_SIM_DEUTAN = np.array([
    [1.0,      0.0,       0.0],
    [0.494207, 0.0,       1.24827],
    [0.0,      0.0,       1.0]
]).T

# 최적값 설정; 실험 결과 alpha_g=0.5, alpha_b=0.9
def get_deutan_params():
    return 0.5, 0.9

def simulate_deutan(img):
    return simulate_cvd_core(img, M_SIM_DEUTAN)