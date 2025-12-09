# video_thread.py (DXcam + RGB 안전 버전)
from PyQt6.QtCore import QThread, pyqtSignal
import numpy as np
import dxcam
from algorithm import apply_daltonization


class VideoThread(QThread):
    change_pixmap_signal = pyqtSignal(object, object)  # (left_img, right_img)

    def __init__(self):
        super().__init__()
        self._run_flag = True
        self.mode = False          # False: OFF(문제 상황), True: ON(보정 후)
        self.cb_type = "protan"    # "protan" or "deutan"
        self.camera = None

    def run(self):
        # ✅ 1) 모니터 0, RGB 포맷으로 받기 (BGRA 아님)
        self.camera = dxcam.create(output_idx=0, output_color="RGB")

        if self.camera is None:
            print("DXcam 카메라 생성 실패")
            return

        # grab() 버전이 더 단순하고, 블랙 프레임 이슈도 적음
        while self._run_flag:
            frame = self.camera.grab()

            if frame is None:
                continue

            # 혹시 모를 타입/값 보정
            frame_rgb = frame.astype(np.uint8)

            # 완전히 까만 프레임(버그)면 그냥 스킵
            if np.mean(frame_rgb) == 0:
                continue

            # 왼쪽: 원본 화면
            left_img = frame_rgb

            # 오른쪽: 색각 시뮬 + 보정
            sim_off, sim_on = apply_daltonization(frame_rgb, cb_type=self.cb_type)

            right_img = sim_on if self.mode else sim_off

            self.change_pixmap_signal.emit(left_img, right_img)

        # 루프 종료 시 자원 정리
        if self.camera is not None:
            try:
                self.camera.stop()
            except Exception:
                pass
            try:
                self.camera.release()
            except Exception:
                pass
            self.camera = None

    def stop(self):
        self._run_flag = False
        self.wait()
