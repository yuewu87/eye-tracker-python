"""视线追踪应用入口 — 整合主界面、校准和追踪功能。"""

import os
import sys
import numpy as np
from PySide6.QtWidgets import QApplication, QSystemTrayIcon, QMenu
from PySide6.QtCore import Qt, QTimer
from PySide6.QtGui import QIcon, QPixmap, QColor

from engine import GazeEngine
from calibrator import run_calibration, run_center_calibration
from widgets import MainWindow, OverlayWindow, CaptureWindow


class App:
    def __init__(self, use_ir=False):
        self.app = QApplication(sys.argv)
        self.app.setQuitOnLastWindowClosed(False)

        screen = self.app.primaryScreen().geometry()
        self.sw = screen.width()
        self.sh = screen.height()

        self.use_ir = use_ir
        self.engine = GazeEngine(self.sw, self.sh, use_ir=use_ir)
        self.main_window = MainWindow()
        self.overlay = None
        self.capture = None
        self.tracking_active = False
        self.overlay_visible = True
        self._frame = 0

        # 系统托盘
        self._setup_tray()

        # 主窗口关闭 → 退出程序
        self.main_window.closeEvent = self._on_main_close

        # 连接信号
        self.main_window.start_clicked.connect(self._toggle_tracking)
        self.main_window.calibrate_clicked.connect(self._run_calibration)
        self.main_window.center_calibrate_clicked.connect(self._run_center_calibration)
        self.main_window.hide_clicked.connect(self._toggle_overlay)
        self.main_window.smoothing_changed.connect(self.engine.set_smoothing)
        self.main_window.hide_panel_clicked.connect(self._hide_panel)
        self.main_window.mode_clicked.connect(self._toggle_mode)

        self._calib_path = lambda: os.path.join(
            os.path.dirname(os.path.abspath(__file__)),
            "calibration_ir.npz" if self.use_ir else "calibration.npz")

        self.engine.start_camera()
        if not self.engine.is_camera_ok():
            self.main_window.status_label.setText("摄像头不可用")
            self.main_window.start_btn.setEnabled(False)

        if not self.engine.has_calibration(self._calib_path()):
            self.main_window.status_label.setText("需要校准...")
            self._run_calibration()

        if self.engine.has_calibration(self._calib_path()):
            self.engine.load_calibration(self._calib_path())
        else:
            self.main_window.status_label.setText("校准失败，请重试")

        self._status_timer = QTimer()
        self._status_timer.timeout.connect(self._update_ui)
        self._status_timer.start(50)

        self.main_window.show()
        self._update_mode_button()

    # ── 模式切换 ──────────────────────────────────────────────────

    def _toggle_mode(self):
        was_tracking = self.tracking_active
        if was_tracking:
            self._stop_tracking()
        self.engine.gaze_updated.disconnect()
        self.engine.stop_camera()
        self.use_ir = not self.use_ir
        print(f"[i] 切换到 {'IR' if self.use_ir else 'RGB'} 模式")
        self.engine = GazeEngine(self.sw, self.sh, use_ir=self.use_ir)
        self.engine.start_camera()
        self.main_window.smoothing_changed.connect(self.engine.set_smoothing)
        if self.engine.has_calibration(self._calib_path()):
            self.engine.load_calibration(self._calib_path())
        else:
            self.main_window.status_label.setText("需要校准...")
            self._run_calibration()
            if self.engine.has_calibration(self._calib_path()):
                self.engine.load_calibration(self._calib_path())
        self._update_mode_button()
        if was_tracking:
            self._start_tracking()

    def _update_mode_button(self):
        if self.use_ir:
            self.main_window.mode_btn.setText("IR 模式")
            self.main_window.mode_btn.setStyleSheet("""
                QPushButton { padding: 8px; font-size: 12px; border-radius: 4px;
                border: 1px solid #a55; background: transparent; color: #a55; }
                QPushButton:hover { border-color: #f88; color: #f88; }
            """)
        else:
            self.main_window.mode_btn.setText("RGB 模式")
            self.main_window.mode_btn.setStyleSheet("""
                QPushButton { padding: 8px; font-size: 12px; border-radius: 4px;
                border: 1px solid #5af; background: transparent; color: #5af; }
                QPushButton:hover { border-color: #8cf; color: #8cf; }
            """)

    # ── 系统托盘 ──────────────────────────────────────────────────

    def _setup_tray(self):
        self.tray = QSystemTrayIcon()
        # 托盘图标
        pixmap = QPixmap(32, 32)
        pixmap.fill(QColor(0, 200, 255))
        self.tray.setIcon(QIcon(pixmap))
        self.tray.setToolTip("Eye Tracker")

        menu = QMenu()
        menu.addAction("显示面板", self._show_panel)
        menu.addAction("退出", self._quit)
        self.tray.setContextMenu(menu)
        self.tray.activated.connect(lambda reason: self._show_panel()
                                    if reason == QSystemTrayIcon.DoubleClick else None)
        self.tray.show()

    def _hide_panel(self):
        self.main_window.hide()

    def _show_panel(self):
        self.main_window.show()
        self.main_window.raise_()

    def _on_main_close(self, event):
        self.main_window.hide()  # 关闭 = 隐藏到托盘
        event.ignore()

    def _quit(self):
        if self.tracking_active:
            self._stop_tracking()
        self.engine.stop_camera()
        self.tray.hide()
        self.app.quit()

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
        self.capture.lower()  # 置底，不挡游戏
        self.overlay.hide()
        self.overlay.show()
        self.engine.reset_position()
        self.tracking_active = True
        self.overlay_visible = True
        self.main_window.set_tracking_active(True)
        self.engine.gaze_updated.connect(self._on_gaze)

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
        self.engine.pause()
        self.main_window.hide()
        run_calibration(self.engine)
        self.engine.resume()
        if self.engine.has_calibration(self._calib_path()):
            self.engine.load_calibration(self._calib_path())
        self.main_window.show()
        if was_tracking:
            self._start_tracking()

    def _run_center_calibration(self):
        was_tracking = self.tracking_active
        if was_tracking:
            self._stop_tracking()
        self.engine.pause()
        self.main_window.hide()
        run_center_calibration(self.engine)
        self.engine.resume()
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
        sys.exit(self.app.exec())


if __name__ == "__main__":
    use_ir = "--ir" in sys.argv
    App(use_ir=use_ir).run()
