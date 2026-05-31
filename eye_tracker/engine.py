"""视线追踪引擎 — 摄像头、MediaPipe、GBR 模型、Kalman 滤波。"""

import math
import os
import sys

# 必须在 import mediapipe 之前设置以屏蔽日志
os.environ["GLOG_minloglevel"] = "3"
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

import cv2
import mediapipe as mp
import numpy as np
from PySide6.QtCore import QObject, Signal, QTimer

# ═══════════════════════════════════════════════════════════════════
# MediaPipe 虹膜关键点索引
# ═══════════════════════════════════════════════════════════════════

RIGHT_IRIS  = [468, 469, 470, 471, 472]
LEFT_IRIS   = [473, 474, 475, 476, 477]
R_EYE_OUTER, R_EYE_INNER = 33, 133
L_EYE_INNER, L_EYE_OUTER = 362, 263

_FaceMesh = mp.solutions.face_mesh.FaceMesh


def extract_features(face_landmarks):
    """从 MediaPipe 人脸关键点提取 7 维视线特征（眼距归一化）。

    返回 float32[7]:
      [右虹膜 dx, dy, 左虹膜 dx, dy, 鼻尖 dx, 鼻尖 dy, 对数眼距]
    """
    lm = face_landmarks.landmark

    ri = np.mean([[lm[i].x, lm[i].y] for i in RIGHT_IRIS], axis=0)
    li = np.mean([[lm[i].x, lm[i].y] for i in LEFT_IRIS], axis=0)

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

    return np.array([dr[0], dr[1], dl[0], dl[1],
                     nose_dx, nose_dy, eye_dist_log], dtype=np.float32)


# ═══════════════════════════════════════════════════════════════════
# Kalman 滤波器 — 4 状态 (x, y, vx, vy) 匀速模型
# ═══════════════════════════════════════════════════════════════════

class KalmanFilter:
    """4 状态 Kalman 滤波器：位置 + 速度，比 EMA 更平滑且延迟更小。"""

    def __init__(self, dt=1/30):
        self.dt = dt
        # 状态: [x, y, vx, vy]
        self.x = np.zeros(4)
        self.P = np.eye(4) * 500
        # 状态转移：匀速运动
        self.F = np.array([[1, 0, dt, 0],
                           [0, 1, 0, dt],
                           [0, 0, 1,  0],
                           [0, 0, 0,  1]])
        # 观测矩阵：只观测位置
        self.H = np.array([[1, 0, 0, 0],
                           [0, 1, 0, 0]])
        # 过程噪声
        self.Q = np.diag([0.5, 0.5, 2.0, 2.0])
        # 测量噪声
        self.R = np.eye(2) * 30
        self.initialized = False

    def update(self, z: np.ndarray):
        """输入测量值 [x, y]，返回滤波后 [x, y]。"""
        if not self.initialized:
            self.x[:2] = z
            self.initialized = True
            return z

        # 预测
        x_pred = self.F @ self.x
        P_pred = self.F @ self.P @ self.F.T + self.Q
        # 更新
        y_innov = z - self.H @ x_pred
        S = self.H @ P_pred @ self.H.T + self.R
        K = P_pred @ self.H.T @ np.linalg.inv(S)
        self.x = x_pred + K @ y_innov
        self.P = (np.eye(4) - K @ self.H) @ P_pred
        return self.x[:2].copy()

    def set_smoothness(self, factor: float):
        """调整平滑度：factor 越大越平滑（增大测量噪声 R）。"""
        noise = 5 + factor * 200  # factor 0..1 → noise 5..205
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

        # 校准模型（GBR）
        self.model = None
        self.x_mean = None
        self.x_std = None
        self.scale_x = 1.0
        self.scale_y = 1.0
        self._has_calib = False

        # 中心校准偏置修正
        self.bias_x = 0.0
        self.bias_y = 0.0

        # Kalman 滤波器
        self.kf = KalmanFilter()

        self.timer = QTimer()
        self.timer.timeout.connect(self._tick)

    # ── 摄像头管理 ──────────────────────────────────────────────

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

    # ── 校准加载 ────────────────────────────────────────────────

    @staticmethod
    def has_calibration(path: str) -> bool:
        return os.path.exists(path)

    def load_calibration(self, path: str):
        calib = np.load(path, allow_pickle=True)
        self.x_mean = calib["x_mean"]
        self.x_std = calib["x_std"]
        if len(self.x_mean) != 7:
            print(f"[!] 校准文件特征维度不匹配 ({len(self.x_mean)} != 7)")
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

    # ── 平滑参数 ─────────────────────────────────────────────────

    def set_smoothing(self, factor: float):
        """调整 Kalman 平滑度（0=灵敏, 1=极平滑）。"""
        self.kf.set_smoothness(max(0.0, min(1.0, factor)))

    def reset_position(self):
        self.gaze_x = self.screen_w / 2.0
        self.gaze_y = self.screen_h / 2.0
        self.prev_x = self.gaze_x
        self.prev_y = self.gaze_y
        self.kf.reset()

    # ── 内部 tick ────────────────────────────────────────────────

    def _tick(self):
        _, results = self.read_camera()
        if results is None:
            self.tracking = False
            self.gaze_updated.emit(self.gaze_x, self.gaze_y, 0.0, 0.0, False)
            return

        if results.multi_face_landmarks:
            feats = extract_features(results.multi_face_landmarks[0])

            if self._has_calib:
                px, py = self.predict(feats)
            else:
                px = float(np.clip(feats[0] + 0.5, 0, 1)) * self.screen_w
                py = float(np.clip(feats[2] + 0.5, 0, 1)) * self.screen_h

            # Kalman 滤波（无需死区，Kalman 自带去抖）
            z = np.array([px, py])
            self.gaze_x, self.gaze_y = self.kf.update(z)
            self.tracking = True
        else:
            self.tracking = False

        # 速度 = 帧间差分（滤波后位置），用于拖尾渲染
        vx = self.gaze_x - self.prev_x
        vy = self.gaze_y - self.prev_y
        self.prev_x = self.gaze_x
        self.prev_y = self.gaze_y

        self.gaze_updated.emit(self.gaze_x, self.gaze_y, vx, vy, self.tracking)
