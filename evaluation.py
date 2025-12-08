# evaluation.py
# 정적 이미지(image.jpg)를 이용해 Daltonization 성능을 정량 평가하는 스크립트
# 사용 예시:
#   python evaluation.py image.jpg --type protan
#   python evaluation.py image.jpg --type deutan --intensity 1.2

# evaluation.py
# 정적 이미지(image.jpg)를 이용해 Daltonization 성능을 정량 평가하는 스크립트

import numpy as np
import cv2
from skimage import color
from scipy.stats import entropy
from algorithm import apply_daltonization


def compute_deltaE(sim_off: np.ndarray, sim_on: np.ndarray) -> float:
    ...
    # (사용자 코드 그대로, 생략 가능)
    ...


def compute_hist_kl(sim_off: np.ndarray, sim_on: np.ndarray):
    ...
    # (사용자 코드 그대로, 생략 가능)
    ...


def compute_contrast(gray: np.ndarray) -> float:
    ...
    # (사용자 코드 그대로, 생략 가능)
    ...


def evaluate_image(image_path: str, cb_type: str = "protan", intensity: float = 1.0):
    """
    한 장의 이미지에 대해 3가지 정량 지표를 계산한다.

    cb_type   : "protan" (제1 적록색맹) 또는 "deutan" (제2 적록색맹)
    intensity : 보정 강도
    """
    # 1) 이미지 로드 (BGR) → RGB 변환
    bgr = cv2.imread(image_path, cv2.IMREAD_COLOR)
    if bgr is None:
        raise FileNotFoundError(f"이미지를 찾을 수 없습니다: {image_path}")

    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)

    # 2) Daltonization 적용 (보정 전/후 시뮬레이션)
    sim_off, sim_on = apply_daltonization(
        rgb,
        cb_type=cb_type,
        intensity=intensity
    )

    # 아래는 기존 코드 그대로...
    mean_deltaE = compute_deltaE(sim_off, sim_on)
    gray_off = cv2.cvtColor(sim_off, cv2.COLOR_RGB2GRAY)
    gray_on = cv2.cvtColor(sim_on, cv2.COLOR_RGB2GRAY)
    ...


def compute_hist_kl(sim_off: np.ndarray, sim_on: np.ndarray):
    """
    히스토그램 기반 KL Divergence 계산 (그레이스케일 기준)
    KL(sim_on || sim_off), KL(sim_off || sim_on) 두 값을 반환
    """
    gray_off = cv2.cvtColor(sim_off, cv2.COLOR_RGB2GRAY)
    gray_on = cv2.cvtColor(sim_on, cv2.COLOR_RGB2GRAY)

    hist_off, _ = np.histogram(gray_off, bins=256, range=(0, 255), density=True)
    hist_on, _ = np.histogram(gray_on, bins=256, range=(0, 255), density=True)

    eps = 1e-10
    hist_off = hist_off + eps
    hist_on = hist_on + eps

    # KL(p || q) = sum p * log(p / q)
    kl_on_off = float(entropy(hist_on, hist_off))   # KL(on || off)
    kl_off_on = float(entropy(hist_off, hist_on))   # KL(off || on)

    return kl_on_off, kl_off_on


def compute_contrast(gray: np.ndarray) -> float:
    """
    간단한 대비 지표: 픽셀 값의 표준편차 (RMS contrast와 유사 개념)
    """
    return float(gray.std())


def evaluate_image(image_path: str, cb_type: str = "protan", intensity: float = 1.0):
    """
    한 장의 이미지에 대해 3가지 정량 지표를 계산한다.

    cb_type   : "protan" (제1 적록색맹) 또는 "deutan" (제2 적록색맹)
    intensity : 보정 강도
    """
    # 1) 이미지 로드 (BGR) → RGB 변환
    bgr = cv2.imread(image_path, cv2.IMREAD_COLOR)
    if bgr is None:
        raise FileNotFoundError(f"이미지를 찾을 수 없습니다: {image_path}")

    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)

    # 2) Daltonization 적용 (보정 전/후 시뮬레이션)
    sim_off, sim_on = apply_daltonization(rgb, cb_type=cb_type, intensity=intensity)

    # 3) ΔE2000 평균 색차
    mean_deltaE = compute_deltaE(sim_off, sim_on)

    # 4) 대비(Contrast) 계산
    gray_off = cv2.cvtColor(sim_off, cv2.COLOR_RGB2GRAY)
    gray_on = cv2.cvtColor(sim_on, cv2.COLOR_RGB2GRAY)

    contrast_off = compute_contrast(gray_off)
    contrast_on = compute_contrast(gray_on)

    if contrast_off != 0:
        contrast_improve_pct = (contrast_on - contrast_off) / contrast_off * 100.0
    else:
        contrast_improve_pct = 0.0

    # 5) 히스토그램 KL Divergence
    kl_on_off, kl_off_on = compute_hist_kl(sim_off, sim_on)

    # 결과를 딕셔너리로 정리해서 반환
    return {
        "cb_type": cb_type,
        "intensity": intensity,
        "mean_deltaE": mean_deltaE,
        "contrast_off": contrast_off,
        "contrast_on": contrast_on,
        "contrast_improve_pct": contrast_improve_pct,
        "kl_on_off": kl_on_off,
        "kl_off_on": kl_off_on,
    }


def pretty_print_result(image_path: str, result: dict):
    cb_kor = "제1 적록색맹" if result["cb_type"] == "protan" else "제2 적록색맹"

    print("==============================================")
    print(f"[이미지] {image_path}")
    print(f"[색각 유형] {cb_kor} ({result['cb_type']})")
    print(f"[보정 강도] intensity = {result['intensity']}")
    print("----------------------------------------------")
    print(f"① 평균 색차 ΔE2000         : {result['mean_deltaE']:.3f}")
    print(
        f"② 대비(표준편차)  OFF/ON : "
        f"{result['contrast_off']:.3f}  →  {result['contrast_on']:.3f} "
        f"({result['contrast_improve_pct']:+.2f} %)"
    )
    print(
        f"③ 히스토그램 KL Divergence: "
        f"KL(ON‖OFF)={result['kl_on_off']:.5f},  KL(OFF‖ON)={result['kl_off_on']:.5f}"
    )
    print("==============================================\n")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Daltonization 정량 평가 스크립트 (ΔE2000, Contrast, KL Divergence)"
    )
    parser.add_argument("image_path", help="입력 이미지 경로 (예: image.jpg)")
    parser.add_argument(
        "--type",
        choices=["protan", "deutan"],
        default="protan",
        help="색각 유형 (protan=제1 적록색맹, deutan=제2 적록색맹)",
    )
    parser.add_argument(
        "--intensity",
        type=float,  
        default=1.0,
        help="보정 강도 (기본값=1.0)",
    )

    args = parser.parse_args()

    res = evaluate_image(args.image_path, cb_type=args.type, intensity=args.intensity)
    pretty_print_result(args.image_path, res)