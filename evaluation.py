# evaluation.py
# 이미지 1장을 평가하고, ΔE / Contrast / KL 그래프까지 자동 생성하는 통합 스크립트

import numpy as np
import cv2
import os  # [추가] 경로 설정을 위해 필요
from skimage import color
from scipy.stats import entropy
import matplotlib.pyplot as plt
from algorithm import apply_daltonization


# -------------------------------------------------------
# 1. ΔE2000 계산
# -------------------------------------------------------
def compute_deltaE(sim_off: np.ndarray, sim_on: np.ndarray) -> float:
    off = sim_off.astype(np.float32) / 255.0
    on = sim_on.astype(np.float32) / 255.0

    lab_off = color.rgb2lab(off)
    lab_on = color.rgb2lab(on)

    deltaE = color.deltaE_ciede2000(lab_off, lab_on)
    return float(deltaE.mean())


# -------------------------------------------------------
# 2. KL Divergence 계산
# -------------------------------------------------------
def compute_hist_kl(sim_off: np.ndarray, sim_on: np.ndarray):
    gray_off = cv2.cvtColor(sim_off, cv2.COLOR_RGB2GRAY)
    gray_on = cv2.cvtColor(sim_on, cv2.COLOR_RGB2GRAY)

    hist_off, _ = np.histogram(gray_off, bins=256, range=(0, 255), density=True)
    hist_on, _ = np.histogram(gray_on, bins=256, range=(0, 255), density=True)

    eps = 1e-10
    hist_off += eps
    hist_on += eps

    kl_on_off = float(entropy(hist_on, hist_off))   # KL(sim_on || sim_off)
    kl_off_on = float(entropy(hist_off, hist_on))   # KL(sim_off || sim_on)

    return kl_on_off, kl_off_on


# -------------------------------------------------------
# 3. 대비(표준편차)
# -------------------------------------------------------
def compute_contrast(gray: np.ndarray) -> float:
    return float(gray.std())


# -------------------------------------------------------
# 4. 단일 강도(intensity)의 평가 실행
# -------------------------------------------------------
def evaluate_image(
    image_path: str,
    cb_type: str = "deutan",
    intensity: float = 1.0,
    alpha_g: float = 0.9,
    alpha_b: float = 0.7,
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
    gray_on = cv2.cvtColor(sim_on, cv2.COLOR_RGB2GRAY)

    contrast_off = compute_contrast(gray_off)
    contrast_on = compute_contrast(gray_on)

    contrast_improve_pct = 0.0
    if contrast_off != 0:
        contrast_improve_pct = (contrast_on - contrast_off) / contrast_off * 100.0

    # 3) 히스토그램 KL Divergence
    kl_on_off, kl_off_on = compute_hist_kl(sim_off, sim_on)

    return {
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
# 5. 그래프 생성 기능 (Intensity Range 평가)
# -------------------------------------------------------
def plot_graphs(image_path: str, cb_type: str):
    intensities = np.linspace(0.2, 1.2, 9)

    deltaEs = []
    contrasts = []
    kl_vals = []

    for intensity in intensities:
        result = evaluate_image(image_path, cb_type=cb_type, intensity=float(intensity))
        deltaEs.append(result["mean_deltaE"])
        contrasts.append(result["contrast_improve_pct"])
        kl_vals.append(result["kl_on_off"])

    # ▼▼▼ [수정된 부분: 저장 경로 및 파일명 설정] ▼▼▼
    # 저장할 폴더 경로 (요청하신 절대 경로)
    save_dir = "/Users/yueun/Desktop/OpensourcePj/OpenSource_project/그래프"
    
    # 폴더가 없으면 생성
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
        print(f"폴더 생성됨: {save_dir}")

    # 파일명 접미사
    suffix = "_2blind"
    # ▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲

    # ΔE 그래프
    plt.figure(figsize=(7, 5))
    plt.plot(intensities, deltaEs, marker="o")
    plt.title("Intensity vs ΔE2000")
    plt.xlabel("Intensity")
    plt.ylabel("ΔE2000")
    plt.grid(True)
    plt.savefig(os.path.join(save_dir, f"plot_deltaE{suffix}.png"), dpi=200) # 경로 수정

    # Contrast 변화 그래프
    plt.figure(figsize=(7, 5))
    plt.plot(intensities, contrasts, marker="o", color="orange")
    plt.title("Intensity vs Contrast Change (%)")
    plt.xlabel("Intensity")
    plt.ylabel("Contrast Change (%)")
    plt.grid(True)
    plt.savefig(os.path.join(save_dir, f"plot_contrast{suffix}.png"), dpi=200) # 경로 수정

    # KL Divergence 그래프
    plt.figure(figsize=(7, 5))
    plt.plot(intensities, kl_vals, marker="o", color="green")
    plt.title("Intensity vs KL Divergence (ON || OFF)")
    plt.xlabel("Intensity")
    plt.ylabel("KL Divergence")
    plt.grid(True)
    plt.savefig(os.path.join(save_dir, f"plot_kl{suffix}.png"), dpi=200) # 경로 수정

    print(f"그래프 3종 생성 완료: {save_dir} 폴더에 *_2blind.png 로 저장되었습니다.")


# -------------------------------------------------------
# 6. M_SHIFT 튜닝 (alpha_g, alpha_b grid search)
# -------------------------------------------------------
def tune_shift(image_path: str, cb_type: str = "deutan", intensity: float = 1.0):
    alpha_candidates = [0.3, 0.5, 0.7, 0.9]

    print("alpha_g  alpha_b   ΔE2000   Contrast(%)   KL(ON||OFF)")
    print("--------------------------------------------------------")

    for ag in alpha_candidates:
        for ab in alpha_candidates:
            res = evaluate_image(
                image_path,
                cb_type=cb_type,
                intensity=intensity,
                alpha_g=ag,
                alpha_b=ab,
            )

            print(
                f"{ag:6.2f}  {ab:6.2f}  "
                f"{res['mean_deltaE']:7.3f}  "
                f"{res['contrast_improve_pct']:10.2f}  "
                f"{res['kl_on_off']:10.3f}"
            )


# -------------------------------------------------------
# 7. 콘솔 예쁘게 출력
# -------------------------------------------------------
def pretty_print_result(image_path: str, result: dict):
    cb_kor = "제1 적록색맹" if result["cb_type"] == "protan" else "제2 적록색맹"

    print("==============================================")
    print(f"[이미지] {image_path}")
    print(f"[색각 유형] {cb_kor} ({result['cb_type']})")
    print(f"[보정 강도] intensity = {result['intensity']}")
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


# -------------------------------------------------------
# 8. main 실행부
# -------------------------------------------------------
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Daltonization 정량 평가 + 그래프 자동 생성"
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
        help="보정 강도",
    )
    parser.add_argument(
        "--graph",
        action="store_true",
        help="Intensity 그래프 생성",
    )
    parser.add_argument(
        "--tune_shift",
        action="store_true",
        help="M_SHIFT(alpha_g, alpha_b) grid search로 튜닝",
    )

    args = parser.parse_args()

    # --- 단일 intensity 평가 ---
    res = evaluate_image(
        args.image_path,
        cb_type=args.type,
        intensity=args.intensity,
    )
    pretty_print_result(args.image_path, res)

    # --- 그래프 생성 ---
    if args.graph:
        plot_graphs(args.image_path, cb_type=args.type)

    # --- M_SHIFT 튜닝 ---
    if args.tune_shift:
        tune_shift(args.image_path, cb_type=args.type, intensity=args.intensity)