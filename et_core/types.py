"""et_core 公共数据类型。"""

import time
from dataclasses import dataclass


@dataclass
class GazeResult:
    """单帧视线追踪结果。"""
    x: float
    y: float
    vx: float
    vy: float
    tracking: bool
    monitor_index: int | None = None
    timestamp: float = 0.0

    def __post_init__(self):
        if self.timestamp == 0.0:
            self.timestamp = time.perf_counter()
