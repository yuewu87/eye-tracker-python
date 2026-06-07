"""多显示器检测 — 虹膜水平偏移分类。"""

import numpy as np
import os


class MonitorDetector:
    """基于虹膜水平偏移最近邻 + 迟滞的多屏分类器。"""

    def __init__(self, hysteresis_frames=8):
        self._offsets = None       # list[float]，每屏参考偏移
        self._hysteresis = hysteresis_frames
        self._candidate = None      # 当前候选屏幕索引
        self._candidate_count = 0

    @property
    def is_calibrated(self) -> bool:
        return self._offsets is not None and len(self._offsets) > 0

    def calibrate(self, offsets: list[float]):
        """存储每块屏幕的虹膜水平偏移参考值。"""
        self._offsets = sorted(offsets)

    def classify(self, iris_h_offset: float) -> int | None:
        """最近邻 + 迟滞：返回当前屏幕索引（0-based）。"""
        if not self.is_calibrated:
            return None
        dists = [abs(iris_h_offset - ref) for ref in self._offsets]
        nearest = int(np.argmin(dists))

        if nearest == self._candidate:
            self._candidate_count += 1
        else:
            self._candidate = nearest
            self._candidate_count = 1

        if self._candidate_count >= self._hysteresis:
            return self._candidate
        return None

    def classify_immediate(self, iris_h_offset: float) -> int | None:
        """无迟滞分类，直接返回最近屏（用于调试）。"""
        if not self.is_calibrated:
            return None
        dists = [abs(iris_h_offset - ref) for ref in self._offsets]
        return int(np.argmin(dists))

    def save(self, path: str):
        np.savez(path, offsets=np.array(self._offsets, dtype=np.float64))

    def load(self, path: str):
        if os.path.exists(path):
            calib = np.load(path)
            self._offsets = list(calib["offsets"])
            return True
        return False
