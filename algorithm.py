import numpy as np
import algorithm_1blind as protan
import algorithm_2blind as deutan

def apply_daltonization(
    img: np.ndarray,
    cb_type: str = "deutan",
    intensity: float = 1.0,     # 보정 강도
    alpha_g: float = 0.9,
    alpha_b: float = 0.7,
):
    
    # protan (제 1 색맹)
    if cb_type == "protan":
        sim_off = protan.simulate_protan(img)
        final_g, final_b = protan.get_protan_params(alpha_g, alpha_b)
        
    # deutan (제 2 색맹)
    elif cb_type == "deutan":
        sim_off = deutan.simulate_deutan(img)
        final_g, final_b = deutan.get_deutan_params() 
        
    else:
        sim_off = deutan.simulate_deutan(img)
        final_g, final_b = alpha_g, alpha_b


    # 공통 보정 로직 
    img_f = img.astype(np.float32)
    sim_f = sim_off.astype(np.float32)
    
    # 에러 계산
    error = img_f - sim_f

    # M_SHIFT 생성 
    M_SHIFT = np.array([
        [0.0,      0.0, 0.0],
        [final_g,  1.0, 0.0],
        [final_b,  0.0, 1.0]
    ]).T

    # 보정 적용
    correction = np.dot(error, M_SHIFT) * intensity
    corr_rgb_f = img_f + correction
    corr_rgb = np.clip(corr_rgb_f, 0, 255).astype(np.uint8)

    # 3. ON 화면 
    if cb_type == "protan":
        sim_on = protan.simulate_protan(corr_rgb)
    else:
        sim_on = deutan.simulate_deutan(corr_rgb)

    return sim_off, sim_on