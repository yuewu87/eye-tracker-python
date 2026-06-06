"""视线追踪引擎 — 摄像头、MediaPipe、多项式回归、Kalman 滤波。"""

import math
import os
import sys

os.environ["GLOG_minloglevel"] = "3"
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

import cv2
import mediapipe as mp
import numpy as np
from sklearn.preprocessing import PolynomialFeatures
from PySide6.QtCore import QObject, Signal, QTimer

# ═══════════════════════════════════════════════════════════════════
# MediaPipe 关键点索引
# ═══════════════════════════════════════════════════════════════════

RIGHT_IRIS  = [468, 469, 470, 471, 472]
LEFT_IRIS   = [473, 474, 475, 476, 477]
R_EYE_OUTER, R_EYE_INNER = 33, 133
L_EYE_INNER, L_EYE_OUTER = 362, 263

_FaceMesh = mp.solutions.face_mesh.FaceMesh


def extract_features(face_landmarks):
    """提取 5 维特征：虹膜偏移 + 眼距。最简单的映射。

    [右虹膜 dx, dy, 左虹膜 dx, dy, 眼距]
    """
    lm = face_landmarks.landmark

    ri = np.mean([[lm[i].x, lm[i].y] for i in RIGHT_IRIS], axis=0)
    li = np.mean([[lm[i].x, lm[i].y] for i in LEFT_IRIS], axis=0)

    re = np.array([(lm[R_EYE_OUTER].x + lm[R_EYE_INNER].x) / 2,
                    (lm[R_EYE_OUTER].y + lm[R_EYE_INNER].y) / 2])
    le = np.array([(lm[L_EYE_INNER].x + lm[L_EYE_OUTER].x) / 2,
                    (lm[L_EYE_INNER].y + lm[L_EYE_OUTER].y) / 2])

    eye_dist = float(np.linalg.norm(re - le))
    dr = ri - re
    dl = li - le

    return np.array([dr[0], dr[1], dl[0], dl[1], eye_dist], dtype=np.float32)


# ═══════════════════════════════════════════════════════════════════
# Kalman 滤波器
# ═══════════════════════════════════════════════════════════════════

class KalmanFilter:
    def __init__(self, dt=1/25):
        self.dt = dt
        self.x = np.zeros(4)
        self.P = np.eye(4) * 500
        self.F = np.array([[1, 0, dt, 0],
                           [0, 1, 0, dt],
                           [0, 0, 1,  0],
                           [0, 0, 0,  1]])
        self.H = np.array([[1, 0, 0, 0],
                           [0, 1, 0, 0]])
        self.Q = np.diag([0.5, 0.5, 2.0, 2.0])
        self.R = np.eye(2) * 40
        self.initialized = False

    def update(self, z: np.ndarray):
        if not self.initialized:
            self.x[:2] = z
            self.initialized = True
            return z
        x_pred = self.F @ self.x
        P_pred = self.F @ self.P @ self.F.T + self.Q
        y_innov = z - self.H @ x_pred
        S = self.H @ P_pred @ self.H.T + self.R
        K = P_pred @ self.H.T @ np.linalg.inv(S)
        self.x = x_pred + K @ y_innov
        self.P = (np.eye(4) - K @ self.H) @ P_pred
        return self.x[:2].copy()

    def set_smoothness(self, factor: float):
        noise = 5 + factor * 200
        self.R = np.eye(2) * noise

    def reset(self):
        self.initialized = False
        self.P = np.eye(4) * 500


# ═══════════════════════════════════════════════════════════════════
# 追踪引擎
# ═══════════════════════════════════════════════════════════════════

class GazeEngine(QObject):
    gaze_updated = Signal(float, float, float, float, bool)

    def __init__(self, screen_w: int, screen_h: int, use_ir=False):
        super().__init__()
        self.screen_w = screen_w
        self.screen_h = screen_h
        self.use_ir = use_ir

        self.gaze_x = screen_w / 2.0
        self.gaze_y = screen_h / 2.0
        self.prev_x = screen_w / 2.0
        self.prev_y = screen_h / 2.0
        self.tracking = False

        self.cap = None
        self.ir_proc = None
        self.face_mesh = None

        self.model = None
        self.x_mean = None
        self.x_std = None
        self._poly = None
        self.scale_x = 1.0
        self.scale_y = 1.0
        self._has_calib = False

        self.bias_x = 0.0
        self.bias_y = 0.0

        self._frame_w = 1920
        self._frame_h = 1080
        self._eye_roi = None  # 上帧眼角像素坐标，用于人脸裁剪

        self.kf = KalmanFilter()
        self.timer = QTimer()
        self.timer.timeout.connect(self._tick)

    # ── 摄像头 ──────────────────────────────────────────────────

    def start_camera(self):
        if self.use_ir:
            from ir_source import IRSource, start_ir_bridge
            print("[i] 启动 IR Bridge...")
            self.ir_proc = start_ir_bridge()
            self.cap = IRSource()
            print(f"[i] IR 摄像头: {self.cap.width}x{self.cap.height}")
        else:
            self.cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
            if not self.cap.isOpened():
                print("[!] 无法打开摄像头")
                return
            self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)
            self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)

        self.face_mesh = _FaceMesh(
            static_image_mode=False, max_num_faces=1,
            refine_landmarks=True,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5,
        )
        self.timer.start(40)  # 25 fps

    def pause(self):
        self.timer.stop()

    def resume(self):
        self.timer.start(40)

    def stop_camera(self):
        self.timer.stop()
        if self.cap is not None:
            if not self.use_ir:
                if self.cap.isOpened():
                    self.cap.release()
            else:
                self.cap.release()
            self.cap = None
        if self.ir_proc is not None:
            self.ir_proc.terminate()
            self.ir_proc = None
        if self.face_mesh is not None:
            self.face_mesh.close()
            self.face_mesh = None

    def is_camera_ok(self) -> bool:
        if self.cap is None:
            return False
        if self.use_ir:
            return self.cap.is_opened()
        return self.cap.isOpened()

    # ── 校准 ────────────────────────────────────────────────────

    @staticmethod
    def has_calibration(path: str) -> bool:
        return os.path.exists(path)

    def load_calibration(self, path: str):
        calib = np.load(path, allow_pickle=True)
        self.x_mean = calib["x_mean"]
        self.x_std = calib["x_std"]
        n_feat = len(self.x_mean)
        if n_feat != 5:
            print(f"[!] 校准特征维度 {n_feat} != 5，请重新校准")
            self._has_calib = False
            return
        self.model = calib["model"].item()
        self.scale_x = self.screen_w / float(calib["screen_w"])
        self.scale_y = self.screen_h / float(calib["screen_h"])

        if "poly_degree" in calib:
            degree = int(calib["poly_degree"])
            n_in = int(calib["poly_features_in"])
            self._poly = PolynomialFeatures(degree=degree, include_bias=False)
            self._poly.fit(np.zeros((1, n_in)))
        else:
            self._poly = None

        self._has_calib = True

    def predict(self, features: np.ndarray):
        x_norm = ((features - self.x_mean) / self.x_std).reshape(1, -1)
        if self._poly is not None:
            x_norm = self._poly.transform(x_norm)
        pred = self.model.predict(x_norm)[0]
        pred[0] = pred[0] * self.scale_x + self.bias_x
        pred[1] = pred[1] * self.scale_y + self.bias_y
        return (float(np.clip(pred[0], 0, self.screen_w)),
                float(np.clip(pred[1], 0, self.screen_h)))

    # ── 单帧读取 ────────────────────────────────────────────────

    def read_camera(self):
        if not self.is_camera_ok():
            return None, None
        ret, frame = self.cap.read()
        if not ret:
            return None, None
        frame = cv2.flip(frame, 1)
        h, w = frame.shape[:2]
        self._frame_w = w
        self._frame_h = h

        # 人脸裁剪归一化：上帧眼角 → 裁剪当前帧 → 放大到标准眼距
        if self._eye_roi is not None:
            ex1, ey1, ex2, ey2 = self._eye_roi
            cx, cy = (ex1 + ex2) // 2, (ey1 + ey2) // 2
            size = max(abs(ex2 - ex1) * 3, abs(ey2 - ey1) * 3, 120)
            x1 = max(0, cx - size)
            y1 = max(0, cy - size)
            x2 = min(w, cx + size)
            y2 = min(h, cy + size)
            if x2 > x1 and y2 > y1:
                crop = frame[y1:y2, x1:x2]
                crop_eye_dist = np.linalg.norm([ex2 - ex1, ey2 - ey1])
                scale = 200.0 / max(crop_eye_dist, 1.0)
                crop = cv2.resize(crop, None, fx=scale, fy=scale)
                # 保存裁剪参数供 _tick 中 eye_roi 反算
                self._cr_w, self._cr_h = crop.shape[1], crop.shape[0]
                self._cr_scale = scale
                self._cr_x1, self._cr_y1 = x1, y1
                rgb = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
                rgb.flags.writeable = False
                results = self.face_mesh.process(rgb)
                # 不再映射回原图 —— landmarks 保持在归一化人脸坐标中
                # 特征提取时眼距始终 ≈200px，头距不变
                return frame, results

        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        rgb.flags.writeable = False
        results = self.face_mesh.process(rgb)
        return frame, results

    # ── 平滑 ────────────────────────────────────────────────────

    def set_smoothing(self, factor: float):
        self.kf.set_smoothness(max(0.0, min(1.0, factor)))

    def reset_position(self):
        self.gaze_x = self.screen_w / 2.0
        self.gaze_y = self.screen_h / 2.0
        self.prev_x = self.gaze_x
        self.prev_y = self.gaze_y
        self.kf.reset()

    # ── tick ─────────────────────────────────────────────────────

    def _tick(self):
        frame, results = self.read_camera()
        if results is None:
            self.tracking = False
            self.gaze_updated.emit(self.gaze_x, self.gaze_y, 0.0, 0.0, False)
            return

        if results.multi_face_landmarks:
            lm = results.multi_face_landmarks[0].landmark
            feats = extract_features(results.multi_face_landmarks[0])

            # 眼角坐标：从裁剪空间映射回原图
            if self._eye_roi is not None:
                s = self._cr_scale
                r_ox = (lm[R_EYE_OUTER].x * self._cr_w / s + self._cr_x1)
                r_oy = (lm[R_EYE_OUTER].y * self._cr_h / s + self._cr_y1)
                l_ox = (lm[L_EYE_OUTER].x * self._cr_w / s + self._cr_x1)
                l_oy = (lm[L_EYE_OUTER].y * self._cr_h / s + self._cr_y1)
                self._eye_roi = (int(r_ox), int(r_oy), int(l_ox), int(l_oy))
            else:
                self._eye_roi = (int(lm[R_EYE_OUTER].x * self._frame_w),
                                  int(lm[R_EYE_OUTER].y * self._frame_h),
                                  int(lm[L_EYE_OUTER].x * self._frame_w),
                                  int(lm[L_EYE_OUTER].y * self._frame_h))

            if self._has_calib:
                px, py = self.predict(feats)
            else:
                px = float(np.clip(feats[0] + 0.5, 0, 1)) * self.screen_w
                py = float(np.clip(feats[2] + 0.5, 0, 1)) * self.screen_h

            z = np.array([px, py])
            self.gaze_x, self.gaze_y = self.kf.update(z)
            self.tracking = True
        else:
            self.tracking = False

        vx = self.gaze_x - self.prev_x
        vy = self.gaze_y - self.prev_y
        self.prev_x = self.gaze_x
        self.prev_y = self.gaze_y

        self.gaze_updated.emit(self.gaze_x, self.gaze_y, vx, vy, self.tracking)
