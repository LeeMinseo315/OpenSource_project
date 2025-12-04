import sys
from PyQt6.QtWidgets import QApplication, QMainWindow, QLabel, QWidget, QVBoxLayout, QHBoxLayout, QPushButton, QComboBox
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
        control_layout.setAlignment(Qt.AlignmentFlag.AlignCenter)

        self.status_label = QLabel("현재 상태: Simulation OFF")
        self.status_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.status_label.setStyleSheet(
            "font-size: 16px; font-weight: bold; color: #00BCD4; padding: 8px;"
        )

        # --- 라벨 2개 생성 ---
        self.label_left = QLabel("Normal Vision")
        # 초기 상태 텍스트
        self.label_right = QLabel("Simulation [OFF]")

        # 스타일 설정
        for label in [self.label_left, self.label_right]:
            label.setAlignment(Qt.AlignmentFlag.AlignCenter)
            label.setStyleSheet("border: 2px solid #555; font-size: 16px; font-weight: bold; color: white;")

        video_layout.addWidget(self.label_left)
        video_layout.addWidget(self.label_right)

        self.combo_cb_type = QComboBox()
        self.combo_cb_type.addItem("제1 적록색맹 (Protanopia)", userData="protan")
        self.combo_cb_type.addItem("제2 적록색맹 (Deuteranopia)", userData="deutan")
        self.combo_cb_type.setCurrentIndex(0)
        self.combo_cb_type.currentIndexChanged.connect(self.on_cb_type_changed)

        control_layout.addWidget(self.combo_cb_type)
        control_layout.addSpacing(20)

        # --- 컨트롤 UI ---
        self.btn_toggle = QPushButton("솔루션 적용하기 (Apply Solution)")
        self.btn_toggle.setCheckable(True)
        self.btn_toggle.setMinimumHeight(60)
        self.btn_toggle.clicked.connect(self.toggle_mode)

        control_layout.addWidget(self.btn_toggle)

        main_layout.addLayout(video_layout)
        main_layout.addLayout(control_layout)
        main_layout.addWidget(self.status_label)

        widget = QWidget()
        widget.setLayout(main_layout)
        self.setCentralWidget(widget)

        self.thread = VideoThread()
        self.thread.cb_type = self.combo_cb_type.currentData()
        self.thread.change_pixmap_signal.connect(self.update_image)
        self.thread.start()

        self.update_status_text()

    def update_image(self, left_img, right_img):
        self.label_left.setPixmap(self.convert_cv_qt(left_img))
        self.label_right.setPixmap(self.convert_cv_qt(right_img))

    def update_status_text(self):
        cb_type = self.thread.cb_type
        cb_text = "제1 적록색맹" if cb_type == "protan" else "제2 적록색맹"
        mode_text = "Simulation ON" if self.thread.mode else "Simulation OFF"

        self.status_label.setText(f"현재 상태: {cb_text} · {mode_text}")

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
            self.btn_toggle.setStyleSheet(
                "background-color: #4CAF50; color: white; font-size: 18px;")
        else:
            # OFF 상태: 문제 상황 보여주기
            self.btn_toggle.setText("솔루션 적용하기 (Apply Solution)")
            self.btn_toggle.setStyleSheet(
                "background-color: #444; color: white; font-size: 18px;")

        self.update_status_text()

    def on_cb_type_changed(self, index: int):
        # VideoThread에 색각 유형 전달
        self.thread.cb_type = self.combo_cb_type.itemData(index)
        self.update_status_text()

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