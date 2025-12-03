import sys
from PyQt6.QtWidgets import QApplication, QMainWindow, QLabel, QWidget, QVBoxLayout, QHBoxLayout, QPushButton
from PyQt6.QtCore import Qt
from PyQt6.QtGui import QImage, QPixmap, QColor, QPalette
from video_thread import VideoThread

class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        # 창 제목 변경
        self.setWindowTitle("Daltonization Project (Final Simulation View)")
        self.resize(1200, 600)

        main_layout = QVBoxLayout()
        video_layout = QHBoxLayout()
        control_layout = QHBoxLayout()

        # --- 라벨 2개 생성 ---
        self.label_left = QLabel("Normal Vision\n(정상인 시각)")
        # 초기 상태 텍스트
        self.label_right = QLabel("Simulation [OFF]\n(보정 전 색맹 시각 - 문제 상황)")
        
        # 스타일 설정
        for label in [self.label_left, self.label_right]:
            label.setAlignment(Qt.AlignmentFlag.AlignCenter)
            label.setStyleSheet("border: 2px solid #555; font-size: 16px; font-weight: bold; color: white;")
        
        video_layout.addWidget(self.label_left)
        video_layout.addWidget(self.label_right)

        # --- 컨트롤 UI ---
        self.btn_toggle = QPushButton("솔루션 적용하기 (Apply Solution)")
        self.btn_toggle.setCheckable(True)
        self.btn_toggle.setMinimumHeight(60)
        self.btn_toggle.setStyleSheet("font-size: 18px; font-weight: bold; background-color: #444; color: white; border-radius: 10px;")
        self.btn_toggle.clicked.connect(self.toggle_mode)
        
        control_layout.addWidget(self.btn_toggle)

        main_layout.addLayout(video_layout)
        main_layout.addLayout(control_layout)

        widget = QWidget()
        widget.setLayout(main_layout)
        self.setCentralWidget(widget)

        self.thread = VideoThread()
        self.thread.change_pixmap_signal.connect(self.update_image)
        self.thread.start()

    def update_image(self, left_img, right_img):
        self.label_left.setPixmap(self.convert_cv_qt(left_img))
        self.label_right.setPixmap(self.convert_cv_qt(right_img))

    def convert_cv_qt(self, cv_img):
        h, w, ch = cv_img.shape
        bytes_per_line = ch * w
        convert_to_Qt_format = QImage(cv_img.data, w, h, bytes_per_line, QImage.Format.Format_RGB888)
        p = convert_to_Qt_format.scaled(580, 450, Qt.AspectRatioMode.KeepAspectRatio)
        return QPixmap.fromImage(p)

    def toggle_mode(self):
        self.thread.mode = self.btn_toggle.isChecked()
        
        if self.thread.mode:
            # ON 상태: 최종 결과(시뮬레이션) 보여주기
            self.btn_toggle.setText("솔루션 해제하기 (Reset)")
            self.btn_toggle.setStyleSheet("background-color: #4CAF50; color: white; font-size: 18px; font-weight: bold; border-radius: 10px;")
            # 라벨 텍스트 핵심 변경!
            self.label_right.setText("Simulation [ON]\n(보정 후 색맹 시각 - 최종 결과)")
            self.label_right.setStyleSheet("border: 3px solid #4CAF50; font-size: 16px; font-weight: bold; color: #4CAF50;")
        else:
            # OFF 상태: 문제 상황 보여주기
            self.btn_toggle.setText("솔루션 적용하기 (Apply Solution)")
            self.btn_toggle.setStyleSheet("background-color: #444; color: white; font-size: 18px; font-weight: bold; border-radius: 10px;")
            self.label_right.setText("Simulation [OFF]\n(보정 전 색맹 시각 - 문제 상황)")
            self.label_right.setStyleSheet("border: 2px solid #FF5722; font-size: 16px; font-weight: bold; color: #FF5722;")

    def closeEvent(self, event):
        self.thread.stop()
        event.accept()

# (다크모드 함수는 그대로 유지)
def set_dark_theme(app):
    # ... (이전 코드와 동일) ...
    app.setStyle("Fusion")
    palette = QPalette()
    palette.setColor(QPalette.ColorRole.Window, QColor(53, 53, 53))
    palette.setColor(QPalette.ColorRole.WindowText, Qt.GlobalColor.white)
    palette.setColor(QPalette.ColorRole.Base, QColor(25, 25, 25))
    palette.setColor(QPalette.ColorRole.AlternateBase, QColor(53, 53, 53))
    palette.setColor(QPalette.ColorRole.ToolTipBase, Qt.GlobalColor.white)
    palette.setColor(QPalette.ColorRole.ToolTipText, Qt.GlobalColor.white)
    palette.setColor(QPalette.ColorRole.Text, Qt.GlobalColor.white)
    palette.setColor(QPalette.ColorRole.Button, QColor(53, 53, 53))
    palette.setColor(QPalette.ColorRole.ButtonText, Qt.GlobalColor.white)
    palette.setColor(QPalette.ColorRole.BrightText, Qt.GlobalColor.red)
    palette.setColor(QPalette.ColorRole.Link, QColor(42, 130, 218))
    palette.setColor(QPalette.ColorRole.Highlight, QColor(42, 130, 218))
    palette.setColor(QPalette.ColorRole.HighlightedText, Qt.GlobalColor.black)
    app.setPalette(palette)

if __name__ == "__main__":
    app = QApplication(sys.argv)
    set_dark_theme(app)
    window = MainWindow()
    window.show()
    sys.exit(app.exec())