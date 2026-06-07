"""et_core 使用示例 — 自动校准 + 打印视线坐标和屏幕索引。"""

import time
from et_core import EyeTracker

tracker = EyeTracker()
tracker.start()

# 显示器标签
labels = {}
for i, (mx, my, mw, mh) in enumerate(tracker.monitors):
    orient = "竖" if mh > mw else "横"
    labels[i] = f"屏{i+1}({mw}x{mh}{orient})"
print(f"主屏: {tracker.screen_w}x{tracker.screen_h}")
print(f"显示器: {list(labels.values())}")

if not tracker.monitor_detector.is_calibrated and len(tracker.monitors) > 1:
    print("[i] 未找到显示器校准，运行多屏校准...")
    tracker.run_monitor_calibration()

try:
    while True:
        result = tracker.update()
        if result.tracking:
            monitor_str = labels.get(result.monitor_index, "?")
            print(f"\r  {monitor_str}  "
                  f"({result.x:6.1f}, {result.y:6.1f})  "
                  f"v=({result.vx:5.1f}, {result.vy:5.1f})  ", end="")
        else:
            print(f"\r  未检测到人脸...", end="")
        time.sleep(0.04)
except KeyboardInterrupt:
    print("\n停止")
finally:
    tracker.stop()
