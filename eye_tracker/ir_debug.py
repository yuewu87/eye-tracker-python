"""IR 模式诊断：显示摄像头画面 + MediaPipe 关键点 + 特征值 + 预测坐标"""
import sys, os, cv2, time, math
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from engine import (GazeEngine, extract_features, RIGHT_IRIS, LEFT_IRIS,
                     R_EYE_OUTER, R_EYE_INNER, L_EYE_INNER, L_EYE_OUTER)
from PySide6.QtWidgets import QApplication

app = QApplication(sys.argv)
screen = app.primaryScreen().geometry()
engine = GazeEngine(screen.width(), screen.height(), use_ir=True)

# 测试：不归一化眼距，直接用原始虹膜偏移
import numpy as np
_orig_extract = extract_features
def ir_extract(lm):
    f = _orig_extract(lm)  # [dr_x, dr_y, dl_x, dl_y, nose_dx, nose_dy, eye_dist_log, yaw, pitch, roll]
    # 把眼距归一化的虹膜偏移 乘以 眼距还原为原始偏移，再乘以放大系数
    eye_dist = f[6]  # log1p(eye_dist * 100)
    real_dist = (np.exp(eye_dist) - 1) / 100  # 还原眼距
    scale = 5.0 / max(real_dist, 0.01)  # 放大系数
    f[0] *= scale  # dr_x
    f[1] *= scale  # dr_y
    f[2] *= scale  # dl_x
    f[3] *= scale  # dl_y
    return f

import engine as eng
eng.extract_features = ir_extract
print("IR 特征放大已启用")
engine.start_camera()
engine.timer.stop()

# 加载校准（如果有）
calib_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "calibration_ir.npz")
if engine.has_calibration(calib_path):
    engine.load_calibration(calib_path)
    print(f"已加载 IR 校准，特征均值: {engine.x_mean}")

print("IR 诊断 — 按 Esc 退出")
print("粉色=虹膜  绿色=眼角  黄字=特征值  红字=预测坐标")

frame_count, bright_count = 0, 0
last_pred = (0, 0)
while True:
    frame, results = engine.read_camera()
    if frame is None:
        time.sleep(0.01)
        continue

    frame_count += 1
    h, w = frame.shape[:2]
    if frame.mean() > 20:
        bright_count += 1

    if results and results.multi_face_landmarks:
        lm = results.multi_face_landmarks[0].landmark

        for i in RIGHT_IRIS + LEFT_IRIS:
            px, py = int(lm[i].x * w), int(lm[i].y * h)
            cv2.circle(frame, (px, py), 2, (255, 100, 255), -1)
        for i in [R_EYE_OUTER, R_EYE_INNER, L_EYE_INNER, L_EYE_OUTER]:
            px, py = int(lm[i].x * w), int(lm[i].y * h)
            cv2.circle(frame, (px, py), 4, (0, 255, 0), 2)

        feats = eng.extract_features(results.multi_face_landmarks[0])
        cv2.putText(frame, f"R iris: ({feats[0]:.3f},{feats[1]:.3f})", (10, h - 55),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 1)
        cv2.putText(frame, f"L iris: ({feats[2]:.3f},{feats[3]:.3f})", (10, h - 35),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 1)
        ypr = f"yaw/pitch/roll: ({feats[7]:.2f},{feats[8]:.2f},{feats[9]:.2f})"
        cv2.putText(frame, ypr, (10, h - 15),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 1)

        if engine._has_calib:
            px, py = engine.predict(feats)
            last_pred = (px, py)
            cv2.putText(frame, f"Pred: ({int(px)},{int(py)})", (10, 50),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)

    cv2.putText(frame, f"F:{frame_count} B:{bright_count} mean:{frame.mean():.0f}", (10, 20),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)

    big = cv2.resize(frame, None, fx=3, fy=3, interpolation=cv2.INTER_NEAREST)
    cv2.imshow("IR Debug (3x)", big)

    if cv2.waitKey(1) == 27:
        break

engine.stop_camera()
cv2.destroyAllWindows()
print(f"共 {frame_count} 帧")
