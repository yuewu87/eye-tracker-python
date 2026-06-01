"""视线追踪引擎 — 3D 头部姿态 + 眼部 ROI 放大 + GBR 模型 + Kalman 滤波。"""

import math
import os
import sys

os.environ["GLOG_minloglevel"] = "3"
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

import cv2
import mediapipe as mp
import numpy as np
from PySide6.QtCore import QObject, Signal, QTimer

# ═══════════════════════════════════════════════════════════════════
# MediaPipe 关键点索引
# ═══════════════════════════════════════════════════════════════════

RIGHT_IRIS  = [468, 469, 470, 471, 472]
LEFT_IRIS   = [473, 474, 475, 476, 477]
R_EYE_OUTER, R_EYE_INNER = 33, 133
L_EYE_INNER, L_EYE_OUTER = 362, 263

_FaceMesh = mp.solutions.face_mesh.FaceMesh

# solvePnP 6 点参考模型 (mm)，用于 3D 头部姿态估计
_PNP_LANDMARKS = [1, 152, 33, 263, 61, 291]
_PNP_3D = np.array([
    [0.0, 0.0, 0.0],           # 鼻尖
    [0.0, -63.6, -12.8],        # 下巴
    [-33.3, 32.5, -30.6],       # 左眼外角
    [33.3, 32.5, -30.6],        # 右眼外角
    [-28.1, -27.5, -23.8],      # 左嘴角
    [28.1, -27.5, -23.8],       # 右嘴角
], dtype=np.float64)

# 相机内参（近似值，单位：像素）
_FX = 1920.0
_FY = 1920.0
_CAMERA_MATRIX = np.array([[_FX, 0, 960], [0, _FY, 540], [0, 0, 1]], dtype=np.float64)
_DIST_COEFFS = np.zeros((4, 1), dtype=np.float64)


def _compute_head_pose(face_landmarks):
    """通过 solvePnP 计算 3D 头部旋转角 (yaw, pitch, roll)，单位：弧度。"""
    lm = face_landmarks.landmark
    img_pts = np.array([[lm[i].x * 1920, lm[i].y * 1080] for i in _PNP_LANDMARKS], dtype=np.float64)
    success, rvec, tvec = cv2.solvePnP(_PNP_3D, img_pts, _CAMERA_MATRIX, _DIST_COEFFS,
                                        flags=cv2.SOLVEPNP_ITERATIVE)
    if not success:
        return 0.0, 0.0, 0.0
    rmat, _ = cv2.Rodrigues(rvec)
    # 从旋转矩阵提取欧拉角
    sy = math.sqrt(rmat[0, 0]**2 + rmat[1, 0]**2)
    singular = sy < 1e-6
    if not singular:
        pitch = math.atan2(-rmat[2, 0], sy)
        yaw   = math.atan2(rmat[1, 0], rmat[0, 0])
        roll  = math.atan2(rmat[2, 1], rmat[2, 2])
    else:
        pitch = math.atan2(-rmat[2, 0], sy)
        yaw   = math.atan2(-rmat[1, 2], rmat[1, 1])
        roll  = 0.0
    return yaw, pitch, roll


def _refine_iris(frame, eye_landmarks):
    """在原始分辨率帧上裁剪眼部区域，用阈值法精确定位瞳孔中心。

    返回 (cx_norm, cy_norm) — 归一化 [0,1] 坐标，或 None。
    """
    h, w = frame.shape[:2]
    xs = [int(lm.x * w) for lm in eye_landmarks]
    ys = [int(lm.y * h) for lm in eye_landmarks]
    x1, x2 = max(0, min(xs) - 8), min(w, max(xs) + 8)
    y1, y2 = max(0, min(ys) - 8), min(h, max(ys) + 8)

    if x2 <= x1 or y2 <= y1:
        return None

    roi = frame[y1:y2, x1:x2]
    gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    # 自适应阈值提取瞳孔（瞳孔是暗区）
    _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    if not contours:
        return None
    # 最大轮廓为瞳孔
    c = max(contours, key=cv2.contourArea)
    M = cv2.moments(c)
    if M["m00"] < 5:
        return None
    cx_roi = M["m10"] / M["m00"]
    cy_roi = M["m01"] / M["m00"]
    # 转换回全图归一化坐标
    return (x1 + cx_roi) / w, (y1 + cy_roi) / h


def extract_features(face_landmarks, bgr_frame=None):
    """提取 10 维视线特征（眼距归一化 + 3D 头部姿态）。

    返回 float32[10]:
      [右虹膜 dx, dy, 左虹膜 dx, dy, 鼻尖 dx, 鼻尖 dy, 对数眼距, yaw, pitch, roll]
    """
    lm = face_landmarks.landmark

    # 虹膜中心（优先用 ROI 精修结果）
    if bgr_frame is not None:
        ri_raw = _refine_iris(bgr_frame, [lm[i] for i in RIGHT_IRIS])
        li_raw = _refine_iris(bgr_frame, [lm[i] for i in LEFT_IRIS])
    else:
        ri_raw, li_raw = None, None

    if ri_raw is None:
        ri_raw = (np.mean([lm[i].x for i in RIGHT_IRIS]),
                  np.mean([lm[i].y for i in RIGHT_IRIS]))
    if li_raw is None:
        li_raw = (np.mean([lm[i].x for i in LEFT_IRIS]),
                  np.mean([lm[i].y for i in LEFT_IRIS]))

    ri = np.array(ri_raw)
    li = np.array(li_raw)

    re = np.array([(lm[R_EYE_OUTER].x + lm[R_EYE_INNER].x) / 2,
                    (lm[R_EYE_OUTER].y + lm[R_EYE_INNER].y) / 2])
    le = np.array([(lm[L_EYE_INNER].x + lm[L_EYE_OUTER].x) / 2,
                    (lm[L_EYE_INNER].y + lm[L_EYE_OUTER].y) / 2])

    face_cx = (re[0] + le[0]) / 2
    face_cy = (re[1] + le[1]) / 2
    eye_dist = float(np.linalg.norm(re - le))

    nose = np.array([lm[1].x, lm[1].y])
    nose_dx = nose[0] - face_cx
    nose_dy = nose[1] - face_cy

    if eye_dist < 1e-6:
        eye_dist = 1.0

    dr = (ri - re) / eye_dist
    dl = (li - le) / eye_dist
    nose_dx /= eye_dist
    nose_dy /= eye_dist
    eye_dist_log = math.log1p(eye_dist * 100)

    yaw, pitch, roll = _compute_head_pose(face_landmarks)

    return np.array([dr[0], dr[1], dl[0], dl[1],
                     nose_dx, nose_dy, eye_dist_log,
                     yaw, pitch, roll], dtype=np.float32)


# ═══════════════════════════════════════════════════════════════════
# Kalman 滤波器
# ═══════════════════════════════════════════════════════════════════

class KalmanFilter:
    def __init__(self, dt=1/30):
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
        self.R = np.eye(2) * 30
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

    def __init__(self, screen_w: int, screen_h: int):
        super().__init__()
        self.screen_w = screen_w
        self.screen_h = screen_h

        self.gaze_x = screen_w / 2.0
        self.gaze_y = screen_h / 2.0
        self.prev_x = screen_w / 2.0
        self.prev_y = screen_h / 2.0
        self.tracking = False

        self.cap = None
        self.face_mesh = None

        self.model = None
        self.x_mean = None
        self.x_std = None
        self.scale_x = 1.0
        self.scale_y = 1.0
        self._has_calib = False

        self.bias_x = 0.0
        self.bias_y = 0.0

        self.kf = KalmanFilter()
        self.timer = QTimer()
        self.timer.timeout.connect(self._tick)

    # ── 摄像头 ──────────────────────────────────────────────────

    def start_camera(self):
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
        self.timer.start(33)

    def pause(self):
        self.timer.stop()

    def resume(self):
        self.timer.start(33)

    def stop_camera(self):
        self.timer.stop()
        if self.cap is not None:
            if self.cap.isOpened():
                self.cap.release()
            self.cap = None
        if self.face_mesh is not None:
            self.face_mesh.close()
            self.face_mesh = None

    def is_camera_ok(self) -> bool:
        return self.cap is not None and self.cap.isOpened()

    # ── 校准 ────────────────────────────────────────────────────

    @staticmethod
    def has_calibration(path: str) -> bool:
        return os.path.exists(path)

    def load_calibration(self, path: str):
        calib = np.load(path, allow_pickle=True)
        self.x_mean = calib["x_mean"]
        self.x_std = calib["x_std"]
        n_feat = len(self.x_mean)
        if n_feat not in (7, 10):
            print(f"[!] 校准特征维度 {n_feat} 不兼容")
            self._has_calib = False
            return
        self.model = calib["model"].item()
        self.scale_x = self.screen_w / float(calib["screen_w"])
        self.scale_y = self.screen_h / float(calib["screen_h"])
        self._has_calib = True

    def predict(self, features: np.ndarray):
        x_norm = ((features - self.x_mean) / self.x_std).reshape(1, -1)
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
            # 传入 BGR 帧用于眼部 ROI 瞳孔精修
            feats = extract_features(results.multi_face_landmarks[0], frame)

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
