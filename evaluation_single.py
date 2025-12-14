# evaluation_single.py
# 이미지 1장을, 지정한 intensity 한 개로만 평가하는 심플 버전
# 표용 데이터 뽑기에 적합

import numpy as np
import cv2
from skimage import color
from scipy.stats import entropy
from algorithm import apply_daltonization


# -------------------------------------------------------
# 1. ΔE2000 계산
# -------------------------------------------------------
def compute_deltaE(sim_off: np.ndarray, sim_on: np.ndarray) -> float:
    off = sim_off.astype(np.float32) / 255.0
    on  = sim_on.astype(np.float32) / 255.0

    lab_off = color.rgb2lab(off)
    lab_on  = color.rgb2lab(on)

    deltaE = color.deltaE_ciede2000(lab_off, lab_on)
    return float(deltaE.mean())


# -------------------------------------------------------
# 2. KL Divergence 계산
# -------------------------------------------------------
def compute_hist_kl(sim_off: np.ndarray, sim_on: np.ndarray):
    gray_off = cv2.cvtColor(sim_off, cv2.COLOR_RGB2GRAY)
    gray_on  = cv2.cvtColor(sim_on,  cv2.COLOR_RGB2GRAY)

    hist_off, _ = np.histogram(gray_off, bins=256, range=(0, 255), density=True)
    hist_on,  _ = np.histogram(gray_on,  bins=256, range=(0, 255), density=True)

    eps = 1e-10
    hist_off += eps
    hist_on  += eps

    kl_on_off = float(entropy(hist_on, hist_off))   # KL(sim_on || sim_off)
    kl_off_on = float(entropy(hist_off, hist_on))   # KL(sim_off || sim_on)

    return kl_on_off, kl_off_on


# -------------------------------------------------------
# 3. 대비(표준편차)
# -------------------------------------------------------
def compute_contrast(gray: np.ndarray) -> float:
    return float(gray.std())


# -------------------------------------------------------
# 4. 단일 intensity 평가
# -------------------------------------------------------
def evaluate_image(
    image_path: str,
    cb_type: str = "deutan",
    intensity: float = 1.0,
    alpha_g: float = 0.9,   # ← 여기 네 튜닝값으로 맞춰 써
    alpha_b: float = 0.7,   # ← 여기도 마찬가지
):
    bgr = cv2.imread(image_path, cv2.IMREAD_COLOR)
    if bgr is None:
        raise FileNotFoundError(f"이미지를 찾을 수 없습니다: {image_path}")

    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)

    # Daltonization 적용
    sim_off, sim_on = apply_daltonization(
        rgb,
        cb_type=cb_type,
        intensity=intensity,
        alpha_g=alpha_g,
        alpha_b=alpha_b,
    )
    

    # 1) ΔE2000
    mean_deltaE = compute_deltaE(sim_off, sim_on)

    # 2) 대비
    gray_off = cv2.cvtColor(sim_off, cv2.COLOR_RGB2GRAY)
    gray_on  = cv2.cvtColor(sim_on,  cv2.COLOR_RGB2GRAY)

    contrast_off = compute_contrast(gray_off)
    contrast_on  = compute_contrast(gray_on)

    contrast_improve_pct = 0.0
    if contrast_off != 0:
        contrast_improve_pct = (contrast_on - contrast_off) / contrast_off * 100.0

    # 3) 히스토그램 KL Divergence
    kl_on_off, kl_off_on = compute_hist_kl(sim_off, sim_on)

    return {
        "image_path": image_path,
        "cb_type": cb_type,
        "intensity": intensity,
        "alpha_g": alpha_g,
        "alpha_b": alpha_b,
        "mean_deltaE": mean_deltaE,
        "contrast_off": contrast_off,
        "contrast_on": contrast_on,
        "contrast_improve_pct": contrast_improve_pct,
        "kl_on_off": kl_on_off,
        "kl_off_on": kl_off_on,
    }


# -------------------------------------------------------
# 5. 콘솔 출력 (사람 읽기용 + 표용 한 줄)
# -------------------------------------------------------
def pretty_print_result(result: dict):
    image_name = result["image_path"]
    cb_type    = result["cb_type"]
    cb_kor = "제1 적록색맹" if cb_type == "protan" else "제2 적록색맹"

    print("==============================================")
    print(f"[이미지] {image_name}")
    print(f"[색각 유형] {cb_kor} ({cb_type})")
    print(f"[보정 강도] intensity = {result['intensity']}")
    print(f"[M_SHIFT] alpha_g = {result['alpha_g']}, alpha_b = {result['alpha_b']}")
    print("----------------------------------------------")
    print(f"① 평균 색차 ΔE2000         : {result['mean_deltaE']:.3f}")
    print(
        f"② 대비(표준편차) OFF/ON : "
        f"{result['contrast_off']:.3f} → {result['contrast_on']:.3f} "
        f"({result['contrast_improve_pct']:+.2f} %)"
    )
    print(
        f"③ 히스토그램 KL Divergence: "
        f"KL(ON‖OFF)={result['kl_on_off']:.5f}, KL(OFF‖ON)={result['kl_off_on']:.5f}"
    )
    print("==============================================\n")

    # 표/엑셀용 한 줄 출력 (복붙해서 표 만들기 편하게)
    print(
        f"TABLE_ROW,"
        f"{image_name},"
        f"{cb_type},"
        f"{result['intensity']:.2f},"
        f"{result['alpha_g']:.2f},"
        f"{result['alpha_b']:.2f},"
        f"{result['mean_deltaE']:.3f},"
        f"{result['contrast_improve_pct']:.2f},"
        f"{result['kl_on_off']:.5f}"
    )


# -------------------------------------------------------
# 6. main
# -------------------------------------------------------
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Daltonization 정량 평가 (단일 intensity용 심플 버전)"
    )

    parser.add_argument("image_path", help="입력 이미지 경로 (예: samples/img.jpg)")
    parser.add_argument(
        "--type",
        choices=["protan", "deutan"],
        default="deutan",
        help="색각 유형 선택",
    )
    parser.add_argument(
        "--intensity",
        type=float,
        default=1.0,
        help="보정 강도 (기본값=1.0)",
    )
    parser.add_argument(
        "--alpha-g",
        type=float,
        default=0.9,   # 여기 네 튜닝값으로 맞춰
        help="M_SHIFT alpha_g 값",
    )
    parser.add_argument(
        "--alpha-b",
        type=float,
        default=0.7,   # 여기도
        help="M_SHIFT alpha_b 값",
    )

    args = parser.parse_args()

    res = evaluate_image(
        args.image_path,
        cb_type=args.type,
        intensity=args.intensity,
        alpha_g=args.alpha_g,
        alpha_b=args.alpha_b,
    )
    pretty_print_result(res)
