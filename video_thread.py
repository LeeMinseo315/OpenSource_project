# video_thread.py
import cv2
from PyQt6.QtCore import QThread, pyqtSignal
from algorithm import apply_daltonization


class VideoThread(QThread):
    # (왼쪽 이미지, 오른쪽 이미지) 2개를 보냄
    change_pixmap_signal = pyqtSignal(object, object)

    def __init__(self):
        super().__init__()
        self._run_flag = True
        self.mode = False       # False: 시뮬레이션 보기 / True: 보정 보기
        self.cb_type = "protan" # 기본값 (MainWindow에서 실제 값으로 덮어씀)

    def run(self):
        cap = cv2.VideoCapture(0)
        # 해상도 설정
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

        while self._run_flag:
            ret, cv_img = cap.read()
            if ret:
                cv_img = cv2.flip(cv_img, 1)  # 좌우반전
                original_img = cv2.cvtColor(cv_img, cv2.COLOR_BGR2RGB)

                # 알고리즘 수행 (시뮬레이션, 보정본 둘 다 받아옴)
                # 슬라이더가 없으므로 강도는 1.0(최대)으로 고정
                sim_img, corr_img = apply_daltonization(
                    original_img,
                    cb_type=self.cb_type,
                    intensity=1.0
                )

                # 모드가 OFF면 -> 오른쪽 화면에 '색맹 시뮬레이션'
                # 모드가 ON이면 -> 오른쪽 화면에 '보정된 화면'
                if self.mode:
                    right_img = corr_img
                else:
                    right_img = sim_img

                # UI로 전송
                self.change_pixmap_signal.emit(original_img, right_img)

        cap.release()

    def stop(self):
        self._run_flag = False
        self.wait()
