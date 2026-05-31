"""视线追踪引擎 — 封装摄像头、MediaPipe 人脸网格、校准模型和平滑滤波。"""

import math
import os
import cv2
import mediapipe as mp
import numpy as np
from sklearn.preprocessing import PolynomialFeatures
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
    """从 MediaPipe 人脸关键点提取 7 维视线特征。

    返回 float32[7]:
      [右眼虹膜偏移 x, y, 左眼虹膜偏移 x, y, 鼻尖偏移 x, 鼻尖偏移 y, 眼距]
    """
    lm = face_landmarks.landmark

    # 虹膜中心
    ri = np.mean([[lm[i].x, lm[i].y] for i in RIGHT_IRIS], axis=0)
    li = np.mean([[lm[i].x, lm[i].y] for i in LEFT_IRIS], axis=0)

    # 眼睛中心（内外眼角中点）
    re = np.array([(lm[R_EYE_OUTER].x + lm[R_EYE_INNER].x) / 2,
                    (lm[R_EYE_OUTER].y + lm[R_EYE_INNER].y) / 2])
    le = np.array([(lm[L_EYE_INNER].x + lm[L_EYE_OUTER].x) / 2,
                    (lm[L_EYE_INNER].y + lm[L_EYE_OUTER].y) / 2])

    # 脸部参考点：双眼连线中点
    face_cx = (re[0] + le[0]) / 2
    face_cy = (re[1] + le[1]) / 2
    eye_dist = float(np.linalg.norm(re - le))

    # 鼻尖（关键点 1）用于估计头部朝向
    nose = np.array([lm[1].x, lm[1].y])
    nose_dx = nose[0] - face_cx
    nose_dy = nose[1] - face_cy

    # 虹膜偏移 + 鼻尖偏移 — 除以眼距实现距离不变
    if eye_dist < 1e-6:
        eye_dist = 1.0

    dr = (ri - re) / eye_dist
    dl = (li - le) / eye_dist
    nose_dx /= eye_dist
    nose_dy /= eye_dist
    eye_dist_log = math.log1p(eye_dist * 100)  # 对数压缩保留距离信息

    return np.array([dr[0], dr[1], dl[0], dl[1],
                     nose_dx, nose_dy, eye_dist_log], dtype=np.float32)


# ═══════════════════════════════════════════════════════════════════
# 追踪引擎
# ═══════════════════════════════════════════════════════════════════

class GazeEngine(QObject):
    """封装摄像头采集、人脸检测、校准预测和 EMA 平滑的 QObject。

    每帧通过 gaze_updated 信号发出 (x, y, vx, vy, tracking)。
    """

    gaze_updated = Signal(float, float, float, float, bool)

    def __init__(self, screen_w: int, screen_h: int):
        super().__init__()
        self.screen_w = screen_w
        self.screen_h = screen_h

        # 平滑后的视线位置（初始在屏幕中央）
        self.gaze_x = screen_w / 2.0
        self.gaze_y = screen_h / 2.0
        self.prev_gaze_x = screen_w / 2.0
        self.prev_gaze_y = screen_h / 2.0

        # EMA 平滑系数（0.01=极平滑, 0.5=极灵敏）
        self.alpha = 0.08

        # 当前帧是否检测到人脸
        self.tracking = False

        # 摄像头和 MediaPipe 实例
        self.cap = None
        self.face_mesh = None

        # 校准参数
        self.coef = None
        self.intercept = None
        self.x_mean = None
        self.x_std = None
        self.scale_x = 1.0
        self.scale_y = 1.0
        self._poly = None       # 多项式特征转换器
        self._has_calib = False

        # 定时器
        self.timer = QTimer()
        self.timer.timeout.connect(self._tick)

    # ── 摄像头管理 ──────────────────────────────────────────────

    def start_camera(self):
        """打开默认摄像头并初始化 MediaPipe FaceMesh。"""
        self.cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
        if not self.cap.isOpened():
            print("[!] 无法打开摄像头")
            return
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

        self.face_mesh = _FaceMesh(
            static_image_mode=False,
            max_num_faces=1,
            refine_landmarks=True,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5,
        )

        self.timer.start(33)  # ~30 fps

    def pause(self):
        """暂停追踪 tick，保持摄像头和 face_mesh 存活（供校准使用）。"""
        self.timer.stop()

    def resume(self):
        """恢复追踪 tick。"""
        self.timer.start(33)

    def stop_camera(self):
        """停止 tick 并释放摄像头和 MediaPipe 资源。"""
        self.timer.stop()
        if self.cap is not None:
            if self.cap.isOpened():
                self.cap.release()
            self.cap = None
        if self.face_mesh is not None:
            self.face_mesh.close()
            self.face_mesh = None

    def is_camera_ok(self) -> bool:
        """摄像头是否正常打开。"""
        return self.cap is not None and self.cap.isOpened()

    # ── 校准文件加载 ────────────────────────────────────────────

    @staticmethod
    def has_calibration(path: str) -> bool:
        """校准文件是否存在。"""
        return os.path.exists(path)

    def load_calibration(self, path: str):
        """加载 calibration.npz 并计算屏幕缩放比例。"""
        calib = np.load(path)
        self.coef = calib["coef"]
        self.intercept = calib["intercept"]
        self.x_mean = calib["x_mean"]
        self.x_std = calib["x_std"]

        # 校验特征维度
        n_features = len(self.x_mean)
        if n_features != 7:
            print(f"[!] 校准文件特征维度不匹配 ({n_features} != 7)，请重新校准")
            self._has_calib = False
            return

        calib_w = float(calib["screen_w"])
        calib_h = float(calib["screen_h"])
        self.scale_x = self.screen_w / calib_w
        self.scale_y = self.screen_h / calib_h

        # 重建多项式特征转换器
        if "poly_degree" in calib:
            degree = int(calib["poly_degree"])
            n_in = int(calib["poly_features_in"])
            self._poly = PolynomialFeatures(degree=degree, include_bias=False)
            self._poly.fit(np.zeros((1, n_in)))  # 初始化内部参数
        else:
            self._poly = None

        self._has_calib = True

    def predict(self, features: np.ndarray):
        """将特征向量映射为屏幕坐标 (x, y)。"""
        x_norm = (features - self.x_mean) / self.x_std

        # 多项式特征扩展（与校准时一致）
        if self._poly is not None:
            x_norm = self._poly.fit_transform(x_norm.reshape(1, -1)).flatten()

        pred = self.coef @ x_norm + self.intercept
        pred[0] *= self.scale_x
        pred[1] *= self.scale_y
        return (float(np.clip(pred[0], 0, self.screen_w)),
                float(np.clip(pred[1], 0, self.screen_h)))

    # ── 单帧读取（校准模块也使用此方法） ────────────────────────

    def read_camera(self):
        """读取并处理一帧画面，返回 (bgr_frame, results_or_None)。"""
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

    def set_smoothing(self, alpha: float):
        """设置 EMA 平滑系数，自动钳制到 [0.01, 0.5]。"""
        self.alpha = max(0.01, min(0.5, alpha))

    def reset_position(self):
        """将视线位置重置到屏幕中央。"""
        self.gaze_x = self.screen_w / 2.0
        self.gaze_y = self.screen_h / 2.0
        self.prev_gaze_x = self.gaze_x
        self.prev_gaze_y = self.gaze_y

    # ── 内部 tick ────────────────────────────────────────────────

    def _tick(self):
        """定时回调：读帧 → 提取特征 → 预测 → 平滑 → 发送信号。"""
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
                # 无校准时使用原始虹膜偏移作为归一化视线
                px = float(np.clip(feats[0] + 0.5, 0, 1)) * self.screen_w
                py = float(np.clip(feats[2] + 0.5, 0, 1)) * self.screen_h

            # 死区：忽略亚像素抖动（3px）
            dx_raw = px - self.gaze_x
            dy_raw = py - self.gaze_y
            if abs(dx_raw) < 3.0 and abs(dy_raw) < 3.0:
                px = self.gaze_x
                py = self.gaze_y

            # EMA 指数平滑
            self.gaze_x = self.alpha * px + (1.0 - self.alpha) * self.gaze_x
            self.gaze_y = self.alpha * py + (1.0 - self.alpha) * self.gaze_y
            self.tracking = True
        else:
            self.tracking = False

        # 计算速度（像素/帧）
        vx = self.gaze_x - self.prev_gaze_x
        vy = self.gaze_y - self.prev_gaze_y
        self.prev_gaze_x = self.gaze_x
        self.prev_gaze_y = self.gaze_y

        self.gaze_updated.emit(self.gaze_x, self.gaze_y, vx, vy, self.tracking)
