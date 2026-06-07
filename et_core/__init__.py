"""et_core — 视线追踪核心库。

用法:
    from et_core import EyeTracker

    tracker = EyeTracker()
    tracker.start()
    result = tracker.update()  # GazeResult
    tracker.stop()
"""

import time
import numpy as np

from et_core.types import GazeResult
from et_core.engine import CameraProcessor
from et_core.filter import KalmanFilter, IIRFilter
from et_core.predictor import GazePredictor
from et_core.monitor_detect import MonitorDetector


class EyeTracker:
    """视线追踪顶层接口。"""

    def __init__(self,
                 screen_w: int = None,
                 screen_h: int = None,
                 monitors: list = None,
                 camera_id: int = 0,
                 calib_path: str = None,
                 monitor_calib_path: str = None):
        import ctypes

        # 屏幕尺寸
        if screen_w is None or screen_h is None:
            user32 = ctypes.windll.user32
            self.screen_w = user32.GetSystemMetrics(0)
            self.screen_h = user32.GetSystemMetrics(1)
        else:
            self.screen_w = screen_w
            self.screen_h = screen_h

        # 多显示器几何
        if monitors is None:
            self._monitors = []
            monitor_idx = 0

            def _enum_callback(hMonitor, hdc, rect, param):
                r = rect.contents
                self._monitors.append((r[0], r[1],
                                       r[2] - r[0], r[3] - r[1]))
                return True

            MonitorEnumProc = ctypes.WINFUNCTYPE(
                ctypes.c_bool, ctypes.c_void_p, ctypes.c_void_p,
                ctypes.POINTER(ctypes.c_long * 4), ctypes.c_void_p
            )
            ctypes.windll.user32.EnumDisplayMonitors(
                None, None, MonitorEnumProc(_enum_callback), 0
            )
        else:
            self._monitors = monitors

        # 组件
        self._camera = CameraProcessor(camera_id=camera_id)
        self._predictor = GazePredictor(self.screen_w, self.screen_h)
        self._kf = KalmanFilter()
        self._iir = IIRFilter(alpha=0.7)
        self._monitor_detector = MonitorDetector(hysteresis_frames=8)

        # 校准路径（默认在 et_core 目录下）
        import os as _os
        _pkg_dir = _os.path.dirname(_os.path.abspath(__file__))
        self._calib_path = calib_path or _os.path.join(_pkg_dir, "calibration.npz")
        self._monitor_calib_path = monitor_calib_path or _os.path.join(_pkg_dir, "monitor_calib.npz")

        # 状态
        self._gaze_x = self.screen_w / 2.0
        self._gaze_y = self.screen_h / 2.0
        self._prev_x = self.screen_w / 2.0
        self._prev_y = self.screen_h / 2.0
        self._tracking = False
        self._monitor_index = None

    # ── 属性 ──────────────────────────────────────────

    @property
    def tracking(self) -> bool:
        return self._tracking

    @property
    def gaze_x(self) -> float:
        return self._gaze_x

    @property
    def gaze_y(self) -> float:
        return self._gaze_y

    @property
    def monitor_index(self) -> int | None:
        return self._monitor_index

    @property
    def monitors(self) -> list:
        return self._monitors

    @property
    def camera(self):
        return self._camera

    @property
    def predictor(self):
        return self._predictor

    @property
    def monitor_detector(self):
        return self._monitor_detector

    # ── 生命周期 ──────────────────────────────────────

    def start(self):
        if not self._camera.open():
            raise RuntimeError("无法打开摄像头")
        self._load_calibration()
        self._load_monitor_calibration()

    def stop(self):
        self._camera.close()

    def _load_calibration(self):
        import os
        if os.path.exists(self._calib_path):
            try:
                self._predictor.load(self._calib_path)
                print(f"[i] 加载校准: {self._calib_path}")
            except Exception as e:
                print(f"[!] 校准加载失败: {e}")

    def _load_monitor_calibration(self):
        import os
        if os.path.exists(self._monitor_calib_path):
            ok = self._monitor_detector.load(self._monitor_calib_path)
            if ok:
                print(f"[i] 加载显示器校准: {self._monitor_calib_path}")

    def set_smoothing(self, factor: float):
        self._kf.set_smoothness(max(0.0, min(1.0, factor)))

    # ── 帧更新 ────────────────────────────────────────

    def update(self) -> GazeResult:
        feats = self._camera.process_frame()
        vx, vy = 0.0, 0.0
        monitor_index = None

        if feats is None:
            self._tracking = False
            return GazeResult(
                x=self._gaze_x, y=self._gaze_y,
                vx=0.0, vy=0.0, tracking=False,
                monitor_index=monitor_index
            )

        self._tracking = True

        if self._predictor.is_loaded:
            px, py = self._predictor.predict(feats)
        else:
            px = float(np.clip(feats[0] + 0.5, 0, 1)) * self.screen_w
            py = float(np.clip(feats[2] + 0.5, 0, 1)) * self.screen_h

        self._gaze_x, self._gaze_y = self._kf.update(np.array([px, py]))

        # 显示器分类
        iris_h = (feats[0] + feats[2]) / 2
        smoothed = self._iir.update(iris_h)
        monitor_index = self._monitor_detector.classify(smoothed)
        self._monitor_index = monitor_index

        vx = self._gaze_x - self._prev_x
        vy = self._gaze_y - self._prev_y
        self._prev_x = self._gaze_x
        self._prev_y = self._gaze_y

        return GazeResult(
            x=self._gaze_x, y=self._gaze_y,
            vx=vx, vy=vy, tracking=True,
            monitor_index=monitor_index
        )

    # ── 校准入口 ──────────────────────────────────────

    @staticmethod
    def _ensure_qt_app():
        from PySide6.QtWidgets import QApplication
        if QApplication.instance() is None:
            return QApplication()

    def run_calibration(self):
        """阻塞运行 7 点校准。需要 PySide6。"""
        self._ensure_qt_app()
        from et_core.calibration.ui import CalibrationWindow
        from PySide6.QtCore import QEventLoop

        window = CalibrationWindow(self._camera)
        loop = QEventLoop()
        window.calibration_done.connect(loop.quit)
        loop.exec()
        self._load_calibration()

    def run_center_calibration(self):
        """阻塞运行中心校准。需要 PySide6。"""
        self._ensure_qt_app()
        from et_core.calibration.ui import CenterCalibWindow
        from PySide6.QtCore import QEventLoop

        window = CenterCalibWindow(self._camera, self._predictor)
        loop = QEventLoop()
        window.calibration_done.connect(loop.quit)
        loop.exec()

    def run_monitor_calibration(self):
        """阻塞运行多显示器校准。需要 PySide6。"""
        self._ensure_qt_app()
        from et_core.calibration.ui import MonitorCalibWindow
        from PySide6.QtCore import QEventLoop

        window = MonitorCalibWindow(self._camera, self._monitors)
        loop = QEventLoop()
        window.calibration_done.connect(loop.quit)
        loop.exec()
        offsets = window.get_offsets()
        if offsets:
            self._monitor_detector.calibrate(offsets)
            self._monitor_detector.save(self._monitor_calib_path)
