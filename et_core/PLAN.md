# et_core 视线追踪核心库 — 实现计划

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 从零构建 `et_core/` 纯 Python 库 — 面部关键点检测、视线坐标预测、多显示器分类。零 GUI 核心，校准 UI 可选。

**Architecture:** 8 个模块自底向上构建。`engine.py` 处理摄像头+特征提取，`predictor.py`+`filter.py` 做坐标预测和平滑，`monitor_detect.py` 做屏幕分类。全部由 `EyeTracker.__init__` 编排，外部只需 `tracker.update()` 驱动一帧。

**Tech Stack:** numpy, opencv-python-headless, mediapipe, scikit-learn（核心）；PySide6（可选校准 UI）

---

### Task 1: types.py — 数据类型

**Files:**
- Create: `et_core/types.py`

- [ ] **Step 1: 创建 types.py**

```python
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
```

- [ ] **Step 2: Commit**

```bash
git add et_core/types.py
git commit -m "feat(et_core): add GazeResult dataclass"
```

---

### Task 2: engine.py — CameraProcessor + 特征提取

**Files:**
- Create: `et_core/engine.py`

- [ ] **Step 1: 创建 engine.py**

```python
"""摄像头读取、MediaPipe FaceMesh、人脸裁剪归一化、特征提取。"""

import math
import os

os.environ["GLOG_minloglevel"] = "3"
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

import cv2
import mediapipe as mp
import numpy as np

# MediaPipe 关键点索引
RIGHT_IRIS  = [468, 469, 470, 471, 472]
LEFT_IRIS   = [473, 474, 475, 476, 477]
R_EYE_OUTER, R_EYE_INNER = 33, 133
L_EYE_INNER, L_EYE_OUTER = 362, 263

_FaceMesh = mp.solutions.face_mesh.FaceMesh


def extract_features(face_landmarks):
    """5 维特征：虹膜偏移 + 对数眼距。[右 dx, dy, 左 dx, dy, log1p(ed*100)]"""
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
    return np.array([dr[0], dr[1], dl[0], dl[1],
                     math.log1p(eye_dist * 100)], dtype=np.float32)


class CameraProcessor:
    """摄像头 + MediaPipe + 人脸裁剪归一化。纯类，无 Qt 依赖。"""

    def __init__(self, camera_id=0):
        self.cap = None
        self.face_mesh = None
        self._camera_id = camera_id
        self._frame_w = 1920
        self._frame_h = 1080
        self._eye_roi = None
        self._cr_w = self._cr_h = 0
        self._cr_scale = 1.0
        self._cr_x1 = self._cr_y1 = 0

    def open(self) -> bool:
        self.cap = cv2.VideoCapture(self._camera_id, cv2.CAP_DSHOW)
        if not self.cap.isOpened():
            return False
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)
        self.face_mesh = _FaceMesh(
            static_image_mode=False, max_num_faces=1,
            refine_landmarks=True,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5,
        )
        return True

    def close(self):
        if self.cap is not None:
            if self.cap.isOpened():
                self.cap.release()
            self.cap = None
        if self.face_mesh is not None:
            self.face_mesh.close()
            self.face_mesh = None

    def is_opened(self) -> bool:
        return self.cap is not None and self.cap.isOpened()

    def read_camera(self):
        """读取一帧，返回 (frame, mp_results)。"""
        if not self.is_opened():
            return None, None
        ret, frame = self.cap.read()
        if not ret:
            return None, None
        frame = cv2.flip(frame, 1)
        h, w = frame.shape[:2]
        self._frame_w = w
        self._frame_h = h

        if self._eye_roi is None:
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            rgb.flags.writeable = False
            results = self.face_mesh.process(rgb)
            if results and results.multi_face_landmarks:
                lm = results.multi_face_landmarks[0].landmark
                self._eye_roi = (int(lm[R_EYE_OUTER].x * w), int(lm[R_EYE_OUTER].y * h),
                                  int(lm[L_EYE_OUTER].x * w), int(lm[L_EYE_OUTER].y * h))
                results = None

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
                self._cr_w, self._cr_h = crop.shape[1], crop.shape[0]
                self._cr_scale = scale
                self._cr_x1, self._cr_y1 = x1, y1
                rgb = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
                rgb.flags.writeable = False
                results = self.face_mesh.process(rgb)
                return frame, results

        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        rgb.flags.writeable = False
        results = self.face_mesh.process(rgb)
        return frame, results

    def update_eye_roi(self, face_landmarks):
        """用当前帧 landmarks 更新眼角坐标，供下帧裁剪。"""
        lm = face_landmarks.landmark
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

    def process_frame(self):
        """单帧处理：读取 → 裁剪 → 特征提取。返回 features(5,) 或 None。"""
        _, results = self.read_camera()
        if results is None or not results.multi_face_landmarks:
            return None
        feats = extract_features(results.multi_face_landmarks[0])
        self.update_eye_roi(results.multi_face_landmarks[0])
        return feats
```

- [ ] **Step 2: Commit**

```bash
git add et_core/engine.py
git commit -m "feat(et_core): add CameraProcessor and extract_features"
```

---

### Task 3: filter.py — Kalman + IIR

**Files:**
- Create: `et_core/filter.py`

- [ ] **Step 1: 创建 filter.py**

```python
"""Kalman 滤波（2D 位置） + 一阶 IIR（1D 信号）。"""

import numpy as np


class KalmanFilter:
    """4 状态 Kalman 滤波器：位置 + 速度，恒速模型。"""

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

    def predict_only(self):
        """只预测不更新，用于测量被丢弃时。"""
        if not self.initialized:
            return self.x[:2].copy()
        self.x = self.F @ self.x
        self.P = self.F @ self.P @ self.F.T + self.Q
        return self.x[:2].copy()

    def set_smoothness(self, factor: float):
        noise = 5 + factor * 200
        self.R = np.eye(2) * noise

    def reset(self):
        self.initialized = False
        self.P = np.eye(4) * 500


class IIRFilter:
    """一阶 IIR：out = alpha * input + (1 - alpha) * prev。"""

    def __init__(self, alpha=0.7):
        self.alpha = alpha
        self._value = None

    def update(self, value: float) -> float:
        if self._value is None:
            self._value = value
        else:
            self._value = self.alpha * value + (1 - self.alpha) * self._value
        return self._value

    def reset(self):
        self._value = None
```

- [ ] **Step 2: Commit**

```bash
git add et_core/filter.py
git commit -m "feat(et_core): add KalmanFilter and IIRFilter"
```

---

### Task 4: predictor.py — 视线坐标预测

**Files:**
- Create: `et_core/predictor.py`

- [ ] **Step 1: 创建 predictor.py**

```python
"""多项式回归视线预测器。加载 calibration.npz，预测像素坐标。"""

import numpy as np
from sklearn.preprocessing import PolynomialFeatures


class GazePredictor:
    """Poly(deg=2) + RidgeCV 视线预测器。"""

    def __init__(self, screen_w: int, screen_h: int):
        self.screen_w = screen_w
        self.screen_h = screen_h
        self.model = None
        self.x_mean = None
        self.x_std = None
        self._poly = None
        self.scale_x = 1.0
        self.scale_y = 1.0
        self.bias_x = 0.0
        self.bias_y = 0.0
        self._loaded = False

    @property
    def is_loaded(self) -> bool:
        return self._loaded

    def load(self, path: str):
        calib = np.load(path, allow_pickle=True)
        self.x_mean = calib["x_mean"]
        self.x_std = calib["x_std"]
        n_feat = len(self.x_mean)
        if n_feat != 5:
            raise ValueError(f"校准特征维度 {n_feat} != 5，请重新校准")
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

        self._loaded = True

    def predict(self, features: np.ndarray):
        x_norm = ((features - self.x_mean) / self.x_std).reshape(1, -1)
        if self._poly is not None:
            x_norm = self._poly.transform(x_norm)
        pred = self.model.predict(x_norm)[0]
        pred[0] = pred[0] * self.scale_x + self.bias_x
        pred[1] = pred[1] * self.scale_y + self.bias_y
        return (float(np.clip(pred[0], 0, self.screen_w)),
                float(np.clip(pred[1], 0, self.screen_h)))
```

- [ ] **Step 2: Commit**

```bash
git add et_core/predictor.py
git commit -m "feat(et_core): add GazePredictor"
```

---

### Task 5: monitor_detect.py — 多显示器检测

**Files:**
- Create: `et_core/monitor_detect.py`

- [ ] **Step 1: 创建 monitor_detect.py**

```python
"""多显示器检测 — 虹膜水平偏移分类。"""

import numpy as np
import os


class MonitorDetector:
    """基于虹膜水平偏移最近邻 + 迟滞的多屏分类器。"""

    def __init__(self, hysteresis_frames=8):
        self._offsets = None       # list[float]，每屏参考偏移
        self._hysteresis = hysteresis_frames
        self._candidate = None      # 当前候选屏幕索引
        self._candidate_count = 0

    @property
    def is_calibrated(self) -> bool:
        return self._offsets is not None and len(self._offsets) > 0

    def calibrate(self, offsets: list[float]):
        """存储每块屏幕的虹膜水平偏移参考值。"""
        self._offsets = sorted(offsets)

    def classify(self, iris_h_offset: float) -> int | None:
        """最近邻 + 迟滞：返回当前屏幕索引（0-based）。"""
        if not self.is_calibrated:
            return None
        dists = [abs(iris_h_offset - ref) for ref in self._offsets]
        nearest = int(np.argmin(dists))

        if nearest == self._candidate:
            self._candidate_count += 1
        else:
            self._candidate = nearest
            self._candidate_count = 1

        if self._candidate_count >= self._hysteresis:
            return self._candidate
        return None

    def classify_immediate(self, iris_h_offset: float) -> int | None:
        """无迟滞分类，直接返回最近屏（用于调试）。"""
        if not self.is_calibrated:
            return None
        dists = [abs(iris_h_offset - ref) for ref in self._offsets]
        return int(np.argmin(dists))

    def save(self, path: str):
        np.savez(path, offsets=np.array(self._offsets, dtype=np.float64))

    def load(self, path: str):
        if os.path.exists(path):
            calib = np.load(path)
            self._offsets = list(calib["offsets"])
            return True
        return False
```

- [ ] **Step 2: Commit**

```bash
git add et_core/monitor_detect.py
git commit -m "feat(et_core): add MonitorDetector with hysteresis"
```

---

### Task 6: calibration/__init__.py + collector.py

**Files:**
- Create: `et_core/calibration/__init__.py`
- Create: `et_core/calibration/collector.py`

- [ ] **Step 1: 创建 calibration/__init__.py**

```python
"""校准子模块 — 数据采集、模型训练、可选 PySide6 UI。"""

from et_core.calibration.collector import CalibrationCollector
```

- [ ] **Step 2: 创建 calibration/collector.py**

```python
"""校准数据采集器 — 回调驱动，零 GUI。"""

import numpy as np


class CalibrationCollector:
    """管理单个注视点的采样：计时、静默过渡、样本收集。

    用法:
        collector = CalibrationCollector(samples_needed=120, settle_seconds=0.5)
        collector.start_point(screen_x, screen_y)
        while not collector.is_done:
            features = camera_processor.process_frame()
            if features is not None:
                collector.feed_frame(features)
        samples = collector.get_samples()
    """

    def __init__(self, samples_needed=120, settle_seconds=0.5, dt=0.033):
        self.samples_needed = samples_needed
        self.settle_seconds = settle_seconds
        self.dt = dt
        self._target = None
        self._samples = []
        self._timer = 0.0
        self._collected = 0
        self._active = False

    @property
    def is_done(self) -> bool:
        return self._active and self._collected >= self.samples_needed

    @property
    def progress(self) -> float:
        return self._collected / max(self.samples_needed, 1)

    def start_point(self, screen_x: float, screen_y: float):
        self._target = np.array([screen_x, screen_y])
        self._samples = []
        self._timer = 0.0
        self._collected = 0
        self._active = True

    def feed_frame(self, features: np.ndarray):
        if not self._active or self.is_done or self._target is None:
            return
        self._timer += self.dt
        if self._timer >= self.settle_seconds:
            self._samples.append((features, self._target.copy()))
            self._collected += 1

    def get_samples(self) -> list:
        """返回 [(features, target), ...]"""
        return self._samples
```

- [ ] **Step 3: Commit**

```bash
git add et_core/calibration/__init__.py et_core/calibration/collector.py
git commit -m "feat(et_core): add CalibrationCollector"
```

---

### Task 7: calibration/trainer.py — 训练 + 保存/加载

**Files:**
- Create: `et_core/calibration/trainer.py`

- [ ] **Step 1: 创建 calibration/trainer.py**

```python
"""校准模型训练、保存、加载、评估。"""

import os
import numpy as np
from sklearn.linear_model import RidgeCV
from sklearn.preprocessing import PolynomialFeatures


def train(samples: list, degree=2):
    """训练 Poly+RidgeCV 模型。

    samples: [(features(5,), target(2,)), ...]
    Returns: (model, x_mean, x_std, poly)
    """
    X = np.array([s[0] for s in samples], dtype=np.float64)
    y = np.array([s[1] for s in samples], dtype=np.float64)
    x_mean = X.mean(axis=0)
    x_std = X.std(axis=0) + 1e-6
    X_norm = (X - x_mean) / x_std

    poly = PolynomialFeatures(degree=degree, include_bias=False)
    X_poly = poly.fit_transform(X_norm)
    n_feat = X_poly.shape[1]

    model = RidgeCV(alphas=[0.01, 0.1, 0.5, 1.0, 5.0, 10.0, 50.0])
    model.fit(X_poly, y)
    print(f"[i] Poly(deg={degree}): 5→{n_feat} 维, best α={model.alpha_}")
    return model, x_mean, x_std, poly


def save(path: str, model, x_mean, x_std, screen_w: int, screen_h: int,
         poly: PolynomialFeatures):
    np.savez(path,
             x_mean=x_mean.astype(np.float64),
             x_std=x_std.astype(np.float64),
             screen_w=screen_w,
             screen_h=screen_h,
             model=model,
             poly_degree=2,
             poly_features_in=5)
    print(f"[OK] 校准参数已保存: {path}")
    print(f"     样本数: {x_mean.shape[0]}, 屏幕: {screen_w}x{screen_h}")


def evaluate(samples: list, model, x_mean, x_std, poly: PolynomialFeatures,
             screen_w: int, screen_h: int, samples_per_point: int,
             point_labels: list[str] = None):
    """评估校准误差。"""
    X_all = np.array([s[0] for s in samples], dtype=np.float64)
    y_all = np.array([s[1] for s in samples], dtype=np.float64)
    Xn = (X_all - x_mean) / x_std
    Xn_poly = poly.transform(Xn)
    y_pred = model.predict(Xn_poly)
    errors = np.sqrt(((y_pred - y_all) ** 2).sum(axis=1))
    print(f"[i] 校准误差: 平均={errors.mean():.1f}px 最大={errors.max():.1f}px")

    if point_labels and samples_per_point > 0:
        for i, label in enumerate(point_labels):
            start = i * samples_per_point
            end = start + samples_per_point
            if start < len(errors):
                e = errors[start:end].mean()
                print(f"     {label}: 平均误差 {e:.0f}px")

    return errors.mean(), errors.max()
```

- [ ] **Step 2: 更新 calibration/__init__.py**

编辑 `et_core/calibration/__init__.py`，在现有内容后追加：

```python
from et_core.calibration.trainer import train, save, evaluate
```

- [ ] **Step 3: Commit**

```bash
git add et_core/calibration/trainer.py et_core/calibration/__init__.py
git commit -m "feat(et_core): add calibration trainer, save, evaluate"
```

---

### Task 8: calibration/ui.py — 可选 PySide6 校准窗口

**Files:**
- Create: `et_core/calibration/ui.py`

- [ ] **Step 1: 创建 calibration/ui.py**

完整复用现有 `eye_tracker/calibrator.py`，适配新接口。内容较多，直接复制现有 calibrator.py 并做以下调整：
1. import 路径改为 `from et_core.engine import extract_features, RIGHT_IRIS, ...`
2. 用 `CalibrationCollector` 替代内嵌的采集逻辑
3. 保留 CalibrationWindow（7点）和 CenterCalibWindow（单点）
4. 新增 MonitorCalibWindow（每屏中心 1 点）
5. 三个窗口都接受 `camera_processor: CameraProcessor` 而非 `engine: GazeEngine`

```python
"""校准 UI — 可选子模块，依赖 PySide6。"""

import os
import numpy as np
from sklearn.preprocessing import PolynomialFeatures
from PySide6.QtWidgets import QApplication, QWidget
from PySide6.QtCore import Qt, QTimer, QPoint, Signal, QEventLoop
from PySide6.QtGui import QPainter, QColor, QFont, QPen

from et_core.engine import (
    extract_features, RIGHT_IRIS, LEFT_IRIS,
    R_EYE_OUTER, R_EYE_INNER, L_EYE_INNER, L_EYE_OUTER,
)
from et_core.calibration.collector import CalibrationCollector
from et_core.calibration.trainer import train, save, evaluate

# 7 点校准：覆盖全屏
CALIB_POINTS = [
    (0.08, 0.08), (0.50, 0.08), (0.92, 0.08),
    (0.08, 0.92), (0.92, 0.92),
    (0.50, 0.50), (0.50, 0.92),
]
CALIB_LABELS = [
    "点1 (8%,8%)", "点2 (50%,8%)", "点3 (92%,8%)",
    "点4 (8%,92%)", "点5 (92%,92%)",
    "点6 (50%,50%)", "点7 (50%,92%)",
]
SAMPLES_RGB = 120
SAMPLES_IR = 30
PREP_SECONDS = 1.0


class CalibrationWindow(QWidget):
    """全屏 7 点校准窗口，阻塞执行。"""

    calibration_done = Signal()

    def __init__(self, camera_processor, use_ir=False):
        super().__init__()
        self.camera = camera_processor
        self.use_ir = use_ir
        print("[i] 正在初始化校准...")

        screen = QApplication.primaryScreen().geometry()
        print(f"[i] 屏幕: {screen.width()}x{screen.height()}")

        self.setCursor(Qt.BlankCursor)
        self.setStyleSheet("background: #1a1a1a;")
        self.setWindowFlags(Qt.FramelessWindowHint | Qt.WindowStaysOnTopHint)
        self.setGeometry(screen)

        self.current_idx = 0
        self.phase = "prep"
        self.phase_timer = 0.0
        self.collector = CalibrationCollector(
            samples_needed=SAMPLES_IR if use_ir else SAMPLES_RGB
        )
        self.all_samples = []
        self._iris_buf = []

        self.timer = QTimer()
        self.timer.timeout.connect(self._tick)
        self.timer.setInterval(33)

        print("[i] 显示校准窗口...")
        self.showFullScreen()
        self.timer.start()

    # ── 绘制 ──────────────────────────────────────────

    def paintEvent(self, event):
        try:
            p = QPainter(self)
            p.setRenderHint(QPainter.Antialiasing)
            w, h = self.width(), self.height()
            p.fillRect(self.rect(), QColor(26, 26, 26))

            if self.current_idx >= len(CALIB_POINTS):
                p.setPen(QColor(255, 255, 255))
                p.setFont(QFont("Arial", 28, QFont.Bold))
                p.drawText(self.rect(), Qt.AlignCenter, "校准完成！\n\n按 Esc 关闭此窗口")
                p.end()
                return

            px, py = CALIB_POINTS[self.current_idx]
            cx = int(px * w)
            cy = int(py * h)

            t = self.phase_timer
            if self.phase == "prep":
                pulse = 0.5 + 0.5 * np.sin(t * 4)
                length = 30.0 + 15.0 * pulse
                gap = 6.0 + 4.0 * pulse
                color = QColor(200, 160, 255, 80 + int(80 * pulse))
            else:
                pulse = 0.5 + 0.5 * np.sin(t * 3)
                length = 34.0 + 8.0 * pulse
                gap = 6.0 + 3.0 * pulse
                color = QColor(120, 200, 255, 140 + int(80 * pulse))
                bar_w, bar_h = 300, 6
                bx = (w - bar_w) // 2
                by_ = h - 50
                p.fillRect(bx, by_, bar_w, bar_h, QColor(60, 60, 60))
                p.fillRect(bx, by_, int(bar_w * self.collector.progress), bar_h, QColor(0, 200, 255))

            pen = QPen(color, 3)
            pen.setCapStyle(Qt.RoundCap)
            p.setPen(pen)
            p.drawLine(cx, int(cy - gap), cx, int(cy - length))
            p.drawLine(cx, int(cy + gap), cx, int(cy + length))
            p.drawLine(int(cx - gap), cy, int(cx - length), cy)
            p.drawLine(int(cx + gap), cy, int(cx + length), cy)
            p.setPen(Qt.NoPen)
            p.setBrush(QColor(255, 255, 255, 200))
            p.drawEllipse(QPoint(cx, cy), 2, 2)

            if self.phase == "prep":
                sec = max(1, int(np.ceil(PREP_SECONDS - self.phase_timer)))
                p.setPen(QColor(255, 255, 255, 100))
                p.setFont(QFont("Arial", 36))
                p.drawText(self.rect(), Qt.AlignCenter, str(sec))

            p.setPen(QColor(120, 120, 120))
            p.setFont(QFont("Arial", 13))
            labels = {"prep": "准备注视", "collect": "采集中..."}
            p.drawText(20, 30, f"[{self.current_idx + 1}/{len(CALIB_POINTS)}]  {labels[self.phase]}")
            p.end()
        except Exception as e:
            print(f"[!] paint 错误: {e}")

    def keyPressEvent(self, event):
        if event.key() == Qt.Key_Escape:
            self.timer.stop()
            self.close()
            self.calibration_done.emit()

    # ── 主循环 ────────────────────────────────────────

    def _tick(self):
        try:
            self._tick_impl()
        except Exception as e:
            print(f"[!] tick 错误: {e}")

    def _tick_impl(self):
        dt = 0.033
        if self.current_idx >= len(CALIB_POINTS):
            return
        self.phase_timer += dt

        if self.phase == "prep" and self.phase_timer >= PREP_SECONDS:
            px, py = CALIB_POINTS[self.current_idx]
            self.collector.start_point(px * self.width(), py * self.height())
            self.phase = "collect"
            self.phase_timer = 0

        elif self.phase == "collect":
            feats = self.camera.process_frame()
            if feats is not None:
                self.collector.feed_frame(feats)

            if self.collector.is_done:
                point_samples = self.collector.get_samples()
                self._filter_outliers(point_samples)
                self.all_samples.extend(point_samples)
                self.current_idx += 1
                self.phase = "prep"
                self.phase_timer = 0
                if self.current_idx >= len(CALIB_POINTS):
                    self._finish()
        self.repaint()

    def _filter_outliers(self, samples):
        if len(samples) < 20:
            return
        feats = np.array([s[0] for s in samples])
        mean = feats.mean(axis=0)
        std = feats.std(axis=0) + 1e-6
        z = np.abs((feats - mean) / std).max(axis=1)
        keep = z < 2.5
        n_removed = (~keep).sum()
        if n_removed > 0:
            samples[:] = [samples[i] for i in range(len(samples)) if keep[i]]
            print(f"  [i] 点{self.current_idx + 1}: 剔除 {n_removed} 个离群帧")

    def _finish(self):
        self.timer.stop()

        if len(self.all_samples) < 30:
            print("[!] 样本不足，请重新校准")
            self.close()
            self.calibration_done.emit()
            return

        model, x_mean, x_std, poly = train(self.all_samples, degree=2)

        screen = QApplication.primaryScreen().geometry()
        fname = "calibration_ir.npz" if self.use_ir else "calibration.npz"
        save_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), fname)
        save(save_path, model, x_mean, x_std, screen.width(), screen.height(), poly)

        evaluate(self.all_samples, model, x_mean, x_std, poly,
                 screen.width(), screen.height(), SAMPLES_IR if self.use_ir else SAMPLES_RGB,
                 CALIB_LABELS)

        self.close()
        self.calibration_done.emit()


class CenterCalibWindow(QWidget):
    """单点中心校准：注视屏幕中央 2.5 秒。"""

    calibration_done = Signal()

    def __init__(self, camera_processor, predictor):
        super().__init__()
        self.camera = camera_processor
        self.predictor = predictor
        screen = QApplication.primaryScreen().geometry()
        self.setCursor(Qt.BlankCursor)
        self.setStyleSheet("background: #1a1a1a;")
        self.setWindowFlags(Qt.FramelessWindowHint | Qt.WindowStaysOnTopHint)
        self.setGeometry(screen)
        self.samples = []
        self.timer_count = int(2.5 * 30)
        self.frame = 0
        self._finished = False

        self.timer = QTimer()
        self.timer.timeout.connect(self._tick)
        self.timer.setInterval(33)
        self.showFullScreen()
        self.timer.start()

    def paintEvent(self, event):
        p = QPainter(self)
        p.setRenderHint(QPainter.Antialiasing)
        w, h = self.width(), self.height()
        p.fillRect(self.rect(), QColor(26, 26, 26))
        cx, cy = w // 2, h // 2
        pulse = 0.5 + 0.5 * np.sin(self.frame * 0.15)
        length = 40.0 + 12.0 * pulse
        gap = 7.0 + 4.0 * pulse
        alpha = 120 + int(80 * pulse)
        pen = QPen(QColor(180, 140, 240, alpha), 3)
        pen.setCapStyle(Qt.RoundCap)
        p.setPen(pen)
        p.drawLine(cx, int(cy - gap), cx, int(cy - length))
        p.drawLine(cx, int(cy + gap), cx, int(cy + length))
        p.drawLine(int(cx - gap), cy, int(cx - length), cy)
        p.drawLine(int(cx + gap), cy, int(cx + length), cy)
        p.setPen(Qt.NoPen)
        p.setBrush(QColor(255, 255, 255, 200))
        p.drawEllipse(QPoint(cx, cy), 2, 2)
        p.setPen(QColor(255, 255, 255, 120))
        p.setFont(QFont("Arial", 18))
        p.drawText(self.rect(), Qt.AlignCenter, f"注视中心十字\n{max(0, self.timer_count - self.frame)//30 + 1} 秒")
        p.end()

    def keyPressEvent(self, event):
        if event.key() == Qt.Key_Escape:
            self.timer.stop()
            self.close()
            self.calibration_done.emit()

    def _tick(self):
        if self._finished:
            return
        self.frame += 1
        feats = self.camera.process_frame()
        if feats is not None:
            self.samples.append(feats)
        self.repaint()
        if self.frame >= self.timer_count:
            self._finish()

    def _finish(self):
        self.timer.stop()
        self._finished = True
        if len(self.samples) < 10:
            self.close()
            self.calibration_done.emit()
            return

        X = np.array(self.samples, dtype=np.float32)
        mean_feats = X.mean(axis=0)
        px, py = self.predictor.predict(mean_feats)

        offset_x = self.predictor.screen_w / 2 - px
        offset_y = self.predictor.screen_h / 2 - py

        print(f"[i] 中心偏移: ({offset_x:.1f}, {offset_y:.1f}) px")
        self.predictor.bias_x = offset_x
        self.predictor.bias_y = offset_y

        self.close()
        self.calibration_done.emit()


class MonitorCalibWindow(QWidget):
    """多显示器校准：依次注视每屏中心。"""

    calibration_done = Signal()

    def __init__(self, camera_processor, monitors: list):
        """
        monitors: [(x, y, w, h), ...] 每块屏幕的几何
        """
        super().__init__()
        self.camera = camera_processor
        self.monitors = monitors
        print(f"[i] 显示器校准 — {len(monitors)} 块屏幕")

        screen = QApplication.primaryScreen().geometry()
        self.setCursor(Qt.BlankCursor)
        self.setStyleSheet("background: #1a1a1a;")
        self.setWindowFlags(Qt.FramelessWindowHint | Qt.WindowStaysOnTopHint)
        self.setGeometry(screen)

        self.current_idx = 0
        self.phase = "prep"
        self.phase_timer = 0.0
        self.collector = CalibrationCollector(samples_needed=60, settle_seconds=0.5)
        self._offsets = []  # 每屏虹膜水平偏移均值

        self.timer = QTimer()
        self.timer.timeout.connect(self._tick)
        self.timer.setInterval(33)
        self.showFullScreen()
        self.timer.start()

    def paintEvent(self, event):
        p = QPainter(self)
        p.setRenderHint(QPainter.Antialiasing)
        w, h = self.width(), self.height()
        p.fillRect(self.rect(), QColor(26, 26, 26))

        if self.current_idx >= len(self.monitors):
            p.setPen(QColor(255, 255, 255))
            p.setFont(QFont("Arial", 28, QFont.Bold))
            p.drawText(self.rect(), Qt.AlignCenter, "显示器校准完成！")
            p.end()
            return

        # 十字准星画在当前屏中心（相对于主屏坐标）
        mx, my, mw, mh = self.monitors[self.current_idx]
        cx = int(mx + mw // 2)
        cy = int(my + mh // 2)

        t = self.phase_timer
        pulse = 0.5 + 0.5 * np.sin(t * (3 if self.phase == "collect" else 4))
        length = 34.0 + 8.0 * pulse
        gap = 6.0 + 3.0 * pulse
        color = QColor(120, 255, 160, 120 + int(80 * pulse))

        pen = QPen(color, 3)
        pen.setCapStyle(Qt.RoundCap)
        p.setPen(pen)
        p.drawLine(cx, int(cy - gap), cx, int(cy - length))
        p.drawLine(cx, int(cy + gap), cx, int(cy + length))
        p.drawLine(int(cx - gap), cy, int(cx - length), cy)
        p.drawLine(int(cx + gap), cy, int(cx + length), cy)
        p.setPen(Qt.NoPen)
        p.setBrush(QColor(255, 255, 255, 200))
        p.drawEllipse(QPoint(cx, cy), 3, 3)

        if self.phase == "prep":
            sec = max(1, int(np.ceil(PREP_SECONDS - self.phase_timer)))
            p.setPen(QColor(255, 255, 255, 100))
            p.setFont(QFont("Arial", 36))
            p.drawText(self.rect(), Qt.AlignCenter, f"屏幕 {self.current_idx + 1}\n{sec}")

        p.setPen(QColor(120, 120, 120))
        p.setFont(QFont("Arial", 13))
        labels = {"prep": "准备注视", "collect": "采集中..."}
        p.drawText(20, 30, f"[{self.current_idx + 1}/{len(self.monitors)}]  {labels[self.phase]}")
        p.end()

    def keyPressEvent(self, event):
        if event.key() == Qt.Key_Escape:
            self.timer.stop()
            self.close()
            self.calibration_done.emit()

    def _tick(self):
        dt = 0.033
        if self.current_idx >= len(self.monitors):
            return
        self.phase_timer += dt

        if self.phase == "prep" and self.phase_timer >= PREP_SECONDS:
            self.collector.start_point(0, 0)
            self.phase = "collect"
            self.phase_timer = 0

        elif self.phase == "collect":
            feats = self.camera.process_frame()
            if feats is not None:
                self.collector.feed_frame(feats)

            if self.collector.is_done:
                samples = self.collector.get_samples()
                offsets = [(s[0][0] + s[0][2]) / 2 for s in samples]
                avg_offset = float(np.mean(offsets))
                self._offsets.append(avg_offset)
                print(f"  [i] 屏幕 {self.current_idx + 1}: 偏移={avg_offset:.4f}")
                self.current_idx += 1
                self.phase = "prep"
                self.phase_timer = 0
                if self.current_idx >= len(self.monitors):
                    self._finish()
        self.repaint()

    def _finish(self):
        self.timer.stop()
        self.calibration_done.emit()

    def get_offsets(self) -> list[float]:
        return self._offsets
```

- [ ] **Step 2: 更新 calibration/__init__.py**

```bash
# 追加到 et_core/calibration/__init__.py:
from et_core.calibration.ui import CalibrationWindow, CenterCalibWindow, MonitorCalibWindow
```

- [ ] **Step 3: Commit**

```bash
git add et_core/calibration/ui.py et_core/calibration/__init__.py
git commit -m "feat(et_core): add optional PySide6 calibration UI"
```

---

### Task 9: __init__.py — EyeTracker 顶层 API

**Files:**
- Create: `et_core/__init__.py`

- [ ] **Step 1: 创建 __init__.py**

```python
"""et_core — 视线追踪核心库。

用法:
    from et_core import EyeTracker

    tracker = EyeTracker()
    tracker.start()
    result = tracker.update()  # GazeResult
    tracker.stop()
"""

import time
import numpy as np

from et_core.types import GazeResult
from et_core.engine import CameraProcessor
from et_core.filter import KalmanFilter, IIRFilter
from et_core.predictor import GazePredictor
from et_core.monitor_detect import MonitorDetector


class EyeTracker:
    """视线追踪顶层接口。"""

    def __init__(self,
                 screen_w: int = None,
                 screen_h: int = None,
                 monitors: list = None,
                 camera_id: int = 0,
                 calib_path: str = None,
                 monitor_calib_path: str = None):
        import ctypes
        from ctypes import wintypes

        # 屏幕尺寸
        if screen_w is None or screen_h is None:
            user32 = ctypes.windll.user32
            self.screen_w = user32.GetSystemMetrics(0)
            self.screen_h = user32.GetSystemMetrics(1)
        else:
            self.screen_w = screen_w
            self.screen_h = screen_h

        # 多显示器几何
        if monitors is None:
            self._monitors = []
            monitor_idx = 0
            def _enum_callback(hMonitor, hdc, rect, param):
                nonlocal monitor_idx
                r = rect.contents
                self._monitors.append((r.left, r.top,
                                       r.right - r.left, r.bottom - r.top))
                monitor_idx += 1
                return True
            MonitorEnumProc = ctypes.WINFUNCTYPE(
                ctypes.c_bool, ctypes.c_void_p, ctypes.c_void_p,
                ctypes.POINTER(ctypes.c_long * 4), ctypes.c_void_p
            )
            ctypes.windll.user32.EnumDisplayMonitors(
                None, None, MonitorEnumProc(_enum_callback), 0
            )
        else:
            self._monitors = monitors

        # 组件
        self._camera = CameraProcessor(camera_id=camera_id)
        self._predictor = GazePredictor(self.screen_w, self.screen_h)
        self._kf = KalmanFilter()
        self._iir = IIRFilter(alpha=0.7)
        self._monitor_detector = MonitorDetector(hysteresis_frames=8)

        # 校准路径
        self._calib_path = calib_path or "calibration.npz"
        self._monitor_calib_path = monitor_calib_path or "monitor_calib.npz"

        # 状态
        self._gaze_x = self.screen_w / 2.0
        self._gaze_y = self.screen_h / 2.0
        self._prev_x = self.screen_w / 2.0
        self._prev_y = self.screen_h / 2.0
        self._tracking = False
        self._running = False

    # ── 属性 ──────────────────────────────────────────

    @property
    def tracking(self) -> bool:
        return self._tracking

    @property
    def gaze_x(self) -> float:
        return self._gaze_x

    @property
    def gaze_y(self) -> float:
        return self._gaze_y

    @property
    def monitor_index(self) -> int | None:
        return getattr(self, '_monitor_index', None)

    @property
    def monitors(self) -> list:
        return self._monitors

    @property
    def camera(self):
        return self._camera

    @property
    def predictor(self):
        return self._predictor

    @property
    def monitor_detector(self):
        return self._monitor_detector

    # ── 生命周期 ──────────────────────────────────────

    def start(self):
        if not self._camera.open():
            raise RuntimeError("无法打开摄像头")

        if self._predictor.load(self._calib_path) if self._calib_path else False:
            pass
        else:
            self._load_calibration()

        self._monitor_detector.load(self._monitor_calib_path)

        self._running = True

    def stop(self):
        self._running = False
        self._camera.close()

    def _load_calibration(self):
        import os
        if os.path.exists(self._calib_path):
            try:
                self._predictor.load(self._calib_path)
            except Exception:
                pass

    def set_smoothing(self, factor: float):
        self._kf.set_smoothness(max(0.0, min(1.0, factor)))

    # ── 帧更新 ────────────────────────────────────────

    def update(self) -> GazeResult:
        feats = self._camera.process_frame()
        vx, vy = 0.0, 0.0
        monitor_index = None

        if feats is None:
            self._tracking = False
            return GazeResult(
                x=self._gaze_x, y=self._gaze_y,
                vx=0.0, vy=0.0, tracking=False,
                monitor_index=monitor_index
            )

        self._tracking = True

        if self._predictor.is_loaded:
            px, py = self._predictor.predict(feats)
        else:
            px = float(np.clip(feats[0] + 0.5, 0, 1)) * self.screen_w
            py = float(np.clip(feats[2] + 0.5, 0, 1)) * self.screen_h

        self._gaze_x, self._gaze_y = self._kf.update(np.array([px, py]))

        # 显示器分类
        iris_h = (feats[0] + feats[2]) / 2
        smoothed = self._iir.update(iris_h)
        monitor_index = self._monitor_detector.classify(smoothed)
        self._monitor_index = monitor_index

        vx = self._gaze_x - self._prev_x
        vy = self._gaze_y - self._prev_y
        self._prev_x = self._gaze_x
        self._prev_y = self._gaze_y

        return GazeResult(
            x=self._gaze_x, y=self._gaze_y,
            vx=vx, vy=vy, tracking=True,
            monitor_index=monitor_index
        )

    # ── 校准入口 ──────────────────────────────────────

    def run_calibration(self):
        """阻塞运行 7 点校准。需要 PySide6。"""
        from et_core.calibration.ui import CalibrationWindow
        from PySide6.QtCore import QEventLoop

        window = CalibrationWindow(self._camera)
        loop = QEventLoop()
        window.calibration_done.connect(loop.quit)
        loop.exec()
        self._load_calibration()

    def run_center_calibration(self):
        """阻塞运行中心校准。需要 PySide6。"""
        from et_core.calibration.ui import CenterCalibWindow
        from PySide6.QtCore import QEventLoop

        window = CenterCalibWindow(self._camera, self._predictor)
        loop = QEventLoop()
        window.calibration_done.connect(loop.quit)
        loop.exec()

    def run_monitor_calibration(self):
        """阻塞运行多显示器校准。需要 PySide6。"""
        from et_core.calibration.ui import MonitorCalibWindow
        from PySide6.QtCore import QEventLoop

        window = MonitorCalibWindow(self._camera, self._monitors)
        loop = QEventLoop()
        window.calibration_done.connect(loop.quit)
        loop.exec()
        offsets = window.get_offsets()
        if offsets:
            self._monitor_detector.calibrate(offsets)
            self._monitor_detector.save(self._monitor_calib_path)
```

- [ ] **Step 2: Commit**

```bash
git add et_core/__init__.py
git commit -m "feat(et_core): add EyeTracker top-level API"
```

---

### Task 10: 集成验证 — 创建示例脚本

**Files:**
- Create: `et_core/example.py`

- [ ] **Step 1: 创建 example.py**

```python
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
```

- [ ] **Step 2: Commit**

```bash
git add et_core/example.py
git commit -m "feat(et_core): add usage example"
```

---

### 完成检查

所有模块创建完毕：

```
et_core/
├── __init__.py           # EyeTracker
├── types.py              # GazeResult
├── engine.py             # CameraProcessor + extract_features
├── predictor.py          # GazePredictor
├── filter.py             # KalmanFilter + IIRFilter
├── monitor_detect.py     # MonitorDetector
├── calibration/
│   ├── __init__.py       # exports
│   ├── collector.py      # CalibrationCollector
│   ├── trainer.py        # train, save, evaluate
│   └── ui.py             # CalibrationWindow, CenterCalibWindow, MonitorCalibWindow
├── example.py            # 使用示例
├── SPEC.md               # 设计规格
└── PLAN.md               # 本文档
```

验证：`python et_core/example.py` 应能启动摄像头并打印视线坐标和显示器索引。
