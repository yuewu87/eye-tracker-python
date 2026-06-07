"""et_core 使用示例 — 打印视线坐标和屏幕索引。"""

import time
from et_core import EyeTracker

tracker = EyeTracker()
tracker.start()
print(f"屏幕: {tracker.screen_w}x{tracker.screen_h}")
print(f"显示器: {tracker.monitors}")

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
