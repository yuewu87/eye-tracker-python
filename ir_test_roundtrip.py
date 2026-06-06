"""测试 IR 桥接全流程：启动 C# bridge → TCP 收帧 → OpenCV 显示"""
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "eye_tracker"))

from ir_source import IRSource, start_ir_bridge
import cv2

print("启动 IR Bridge...")
proc = start_ir_bridge()

try:
    src = IRSource()
    print("按 Esc 退出。你应该看到红外画面。")
    while True:
        ret, frame = src.read()
        if ret and frame.mean() > 20:  # 过滤暗帧（IR LED 未点亮）
            cv2.imshow("IR Camera - TCP Bridge", frame)
        if cv2.waitKey(1) == 27:
            break
finally:
    src.release()
    proc.terminate()
    cv2.destroyAllWindows()
    print("退出")
