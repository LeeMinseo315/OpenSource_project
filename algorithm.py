import numpy as np
# 분리한 파일들 불러오기
import algorithm_1blind as protan
import algorithm_2blind as deutan

def apply_daltonization(
    img: np.ndarray,
    cb_type: str = "deutan",
    intensity: float = 1.0,
    alpha_g: float = 0.9,
    alpha_b: float = 0.7,
):
    """
    Daltonization 메인 함수
    cb_type에 따라 적절한 모듈(protan/deutan)을 호출
    """
    
    # 1. 타입에 따른 설정 분기
    if cb_type == "protan":
        # 제1색맹: 시뮬레이션 함수 & 파라미터 가져오기
        sim_off = protan.simulate_protan(img)
        final_g, final_b = protan.get_protan_params(alpha_g, alpha_b)
        
    elif cb_type == "deutan":
        # 제2색맹: 시뮬레이션 함수 & 최적 파라미터 가져오기
        sim_off = deutan.simulate_deutan(img)
        final_g, final_b = deutan.get_deutan_params() # 0.5, 0.9가 옴
        
    else:
        # 예외 처리 (기본은 deutan으로)
        sim_off = deutan.simulate_deutan(img)
        final_g, final_b = alpha_g, alpha_b

    # 2. 공통 보정 로직 (계산은 여기서 수행)
    img_f = img.astype(np.float32)
    sim_f = sim_off.astype(np.float32)
    
    # 에러 계산
    error = img_f - sim_f

    # M_SHIFT 생성 (위에서 결정된 final_g, final_b 사용)
    M_SHIFT = np.array([
        [0.0,      0.0, 0.0],
        [final_g,  1.0, 0.0],
        [final_b,  0.0, 1.0]
    ]).T

    # 보정 적용
    correction = np.dot(error, M_SHIFT) * intensity
    corr_rgb_f = img_f + correction
    corr_rgb = np.clip(corr_rgb_f, 0, 255).astype(np.uint8)

    # 3. ON 화면 시뮬레이션 (결과 확인용)
    if cb_type == "protan":
        sim_on = protan.simulate_protan(corr_rgb)
    else:
        sim_on = deutan.simulate_deutan(corr_rgb)

    return sim_off, sim_on