"""视线追踪应用入口 — 整合主界面、校准和追踪功能。"""

import os
import sys
import numpy as np
from PySide6.QtWidgets import QApplication
from PySide6.QtCore import Qt, QTimer

from engine import GazeEngine
from calibrator import run_calibration
from widgets import MainWindow, OverlayWindow, CaptureWindow


class App:
    """应用控制器：管理引擎、窗口和校准/追踪状态切换。"""

    def __init__(self):
        self.app = QApplication(sys.argv)
        screen = self.app.primaryScreen().geometry()
        self.sw = screen.width()
        self.sh = screen.height()

        self.engine = GazeEngine(self.sw, self.sh)
        self.main_window = MainWindow()
        self.overlay = None
        self.capture = None
        self.tracking_active = False
        self.overlay_visible = True
        self._frame = 0

        # 连接信号
        self.main_window.start_clicked.connect(self._toggle_tracking)
        self.main_window.calibrate_clicked.connect(self._run_calibration)
        self.main_window.hide_clicked.connect(self._toggle_overlay)
        self.main_window.smoothing_changed.connect(self.engine.set_smoothing)

        calib_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "calibration.npz")

        # 启动摄像头
        self.engine.start_camera()
        if not self.engine.is_camera_ok():
            self.main_window.status_label.setText("摄像头不可用")
            self.main_window.start_btn.setEnabled(False)

        # 无校准文件则自动进入校准
        if not self.engine.has_calibration(calib_path):
            self.main_window.status_label.setText("需要校准...")
            self._run_calibration()

        self.engine.load_calibration(calib_path)

        # 定时刷新主界面状态
        self._status_timer = QTimer()
        self._status_timer.timeout.connect(self._update_ui)
        self._status_timer.start(50)

        self.main_window.show()

    # ── 追踪控制 ──────────────────────────────────────────────────

    def _toggle_tracking(self):
        if self.tracking_active:
            self._stop_tracking()
        else:
            self._start_tracking()

    def _start_tracking(self):
        screen_geo = self.app.primaryScreen().geometry()
        self.overlay = OverlayWindow(screen_geo)
        self.capture = CaptureWindow(screen_geo)
        self.engine.reset_position()
        self.tracking_active = True
        self.overlay_visible = False
        self.main_window.set_tracking_active(True)
        self.engine.gaze_updated.connect(self._on_gaze)
        QApplication.processEvents()
        self._toggle_overlay()  # 强制显示确保渲染

    def _stop_tracking(self):
        self.engine.gaze_updated.disconnect(self._on_gaze)
        if self.overlay:
            self.overlay.close()
            self.overlay = None
        if self.capture:
            self.capture.close()
            self.capture = None
        self.tracking_active = False
        self.main_window.set_tracking_active(False)

    def _toggle_overlay(self):
        if self.overlay:
            self.overlay_visible = not self.overlay_visible
            self.overlay.setVisible(self.overlay_visible)

    # ── 校准流程 ──────────────────────────────────────────────────

    def _run_calibration(self):
        was_tracking = self.tracking_active
        if was_tracking:
            self._stop_tracking()
        self.engine.pause()              # 暂停 tick，保持摄像头存活
        self.main_window.hide()
        run_calibration(self.engine)
        self.engine.resume()             # 恢复 tick
        calib_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "calibration.npz")
        if self.engine.has_calibration(calib_path):
            self.engine.load_calibration(calib_path)
        self.main_window.show()
        if was_tracking:
            self._start_tracking()

    # ── 视线数据回调 ──────────────────────────────────────────────

    def _on_gaze(self, x, y, vx, vy, tracking):
        self._frame += 1
        pulse = np.sin(self._frame * 0.07)
        if self.overlay:
            self.overlay.update_state(x, y, vx, vy, pulse, tracking)
        if self.capture:
            self.capture.update_state(x, y, vx, vy, pulse, tracking)

    def _update_ui(self):
        if self.tracking_active:
            self.main_window.update_status(True, self.engine.gaze_x, self.engine.gaze_y)
        else:
            self.main_window.update_status(False, 0, 0)

    def run(self):
        self.app.exec()


if __name__ == "__main__":
    App().run()
