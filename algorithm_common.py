import numpy as np

# RGB ↔ LMS 변환 행렬
M_RGB2LMS = np.array([
    [17.8824, 43.5161, 4.11935],
    [3.45565, 27.1554, 3.86714],
    [0.0299566, 0.184309, 1.46709]
]).T

M_LMS2RGB = np.linalg.inv(M_RGB2LMS)

# 공통 로직
def simulate_cvd_core(rgb_img: np.ndarray, sim_matrix: np.ndarray) -> np.ndarray:

    img_float = rgb_img.astype(np.float32)

    # RGB -> LMS
    lms = np.dot(img_float, M_RGB2LMS)
    
    # LMS 변조
    lms_sim = np.dot(lms, sim_matrix)
    
    # LMS -> RGB
    rgb_sim = np.dot(lms_sim, M_LMS2RGB)
    
    # Clipping
    return np.clip(rgb_sim, 0, 255).astype(np.uint8)