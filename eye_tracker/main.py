"""Eye Tracker V2 — single-app entry point with integrated calibration."""

import os
import sys
import numpy as np
from PySide6.QtWidgets import QApplication
from PySide6.QtCore import Qt, QTimer

from engine import GazeEngine
from calibrator import run_calibration
from widgets import MainWindow, OverlayWindow, CaptureWindow


class App:
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

        # Wire signals
        self.main_window.start_clicked.connect(self._toggle_tracking)
        self.main_window.calibrate_clicked.connect(self._run_calibration)
        self.main_window.hide_clicked.connect(self._toggle_overlay)
        self.main_window.smoothing_changed.connect(self.engine.set_smoothing)

        calib_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "calibration.npz")

        # Start camera
        self.engine.start_camera()
        if not self.engine.is_camera_ok():
            self.main_window.status_label.setText("摄像头不可用")
            self.main_window.start_btn.setEnabled(False)

        # Auto-calibrate if needed
        if not self.engine.has_calibration(calib_path):
            self.main_window.status_label.setText("需要校准...")
            self._run_calibration()

        self.engine.load_calibration(calib_path)

        # Update timer for main window status
        self._status_timer = QTimer()
        self._status_timer.timeout.connect(self._update_ui)
        self._status_timer.start(50)

        self.main_window.show()

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

    def _run_calibration(self):
        was_tracking = self.tracking_active
        if was_tracking:
            self._stop_tracking()
        self.engine.pause()
        self.main_window.hide()
        run_calibration(self.engine)
        self.engine.resume()
        calib_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "calibration.npz")
        if self.engine.has_calibration(calib_path):
            self.engine.load_calibration(calib_path)
        self.main_window.show()
        if was_tracking:
            self._start_tracking()

    def _on_gaze(self, x, y, vx, vy, tracking):
        self._frame += 1
        pulse = np.sin(self._frame * 0.07)
        if self.overlay:
            self.overlay.update_state(x, y, vx, vy, pulse, tracking)
        if self.capture:
            self.capture.update_state(x, y, vx, vy, pulse, tracking)

    def _update_ui(self):
        if self.tracking_active and hasattr(self.engine, '_gaze_x'):
            self.main_window.update_status(
                True, self.engine._gaze_x, self.engine._gaze_y)
        else:
            self.main_window.update_status(False, 0, 0)

    def run(self):
        self.app.exec()


if __name__ == "__main__":
    App().run()
