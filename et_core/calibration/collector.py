"""校准数据采集器 — 回调驱动，零 GUI。"""

import numpy as np


class CalibrationCollector:
    """管理单个注视点的采样：计时、静默过渡、样本收集。

    用法:
        collector = CalibrationCollector(samples_needed=120, settle_seconds=0.5)
        collector.start_point(screen_x, screen_y)
        while not collector.is_done:
            features = camera_processor.process_frame()
            if features is not None:
                collector.feed_frame(features)
        samples = collector.get_samples()
    """

    def __init__(self, samples_needed=120, settle_seconds=0.5, dt=0.033):
        self.samples_needed = samples_needed
        self.settle_seconds = settle_seconds
        self.dt = dt
        self._target = None
        self._samples = []
        self._timer = 0.0
        self._collected = 0
        self._active = False

    @property
    def is_done(self) -> bool:
        return self._active and self._collected >= self.samples_needed

    @property
    def progress(self) -> float:
        return self._collected / max(self.samples_needed, 1)

    def start_point(self, screen_x: float, screen_y: float):
        self._target = np.array([screen_x, screen_y])
        self._samples = []
        self._timer = 0.0
        self._collected = 0
        self._active = True

    def feed_frame(self, features: np.ndarray):
        if not self._active or self.is_done or self._target is None:
            return
        self._timer += self.dt
        if self._timer >= self.settle_seconds:
            self._samples.append((features, self._target.copy()))
            self._collected += 1

    def get_samples(self) -> list:
        """返回 [(features, target), ...]"""
        return self._samples
