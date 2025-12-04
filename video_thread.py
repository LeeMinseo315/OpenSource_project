# video_thread.py
from PyQt6.QtCore import QThread, pyqtSignal
import cv2
from algorithm import apply_daltonization

class VideoThread(QThread):
    change_pixmap_signal = pyqtSignal(object, object)  # (left_img, right_img)

    def __init__(self):
        super().__init__()
        self._run_flag = True
        self.mode = False          # False: OFF(문제 상황), True: ON(보정 후)
        self.cb_type = "protan"    # "protan" or "deutan"

    def run(self):
        cap = cv2.VideoCapture(0)
        while self._run_flag:
            ret, frame = cap.read()
            if not ret:
                continue

            # OpenCV는 BGR이므로 RGB로 변환
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

            # 정상 시각(왼쪽)은 그냥 원본
            left_img = frame_rgb

            # 색각 시뮬레이션 + 보정
            sim_off, sim_on = apply_daltonization(frame_rgb, cb_type=self.cb_type)

            # 오른쪽은 모드에 따라 OFF/ON 선택
            right_img = sim_on if self.mode else sim_off

            self.change_pixmap_signal.emit(left_img, right_img)

        cap.release()

    def stop(self):
        self._run_flag = False
        self.wait()
