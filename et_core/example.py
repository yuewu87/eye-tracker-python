"""et_core 使用示例 — 自动校准 + 打印视线坐标和屏幕索引。"""

import os
import time
from et_core import EyeTracker

tracker = EyeTracker()
tracker.start()
print(f"屏幕: {tracker.screen_w}x{tracker.screen_h}")
print(f"显示器: {tracker.monitors}")

# 无校准文件则自动运行
if not tracker.predictor.is_loaded:
    print("[i] 未找到校准文件，运行 7 点校准...")
    tracker.run_calibration()

# 无显示器校准则自动运行
if not tracker.monitor_detector.is_calibrated and len(tracker.monitors) > 1:
    print("[i] 未找到显示器校准，运行多屏校准...")
    tracker.run_monitor_calibration()

try:
    while True:
        result = tracker.update()
        if result.tracking:
            print(f"\r  Gaze: ({result.x:6.1f}, {result.y:6.1f})  "
                  f"v=({result.vx:5.1f}, {result.vy:5.1f})  "
                  f"monitor={result.monitor_index}  ", end="")
        else:
            print(f"\r  未检测到人脸...", end="")
        time.sleep(0.04)
except KeyboardInterrupt:
    print("\n停止")
finally:
    tracker.stop()
