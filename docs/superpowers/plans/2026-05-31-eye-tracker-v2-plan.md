# Eye Tracker V2 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Merge calibrator + tracker into a single PySide6 app with dark-themed main panel, fluid-deforming ring aperture, and integrated calibration.

**Architecture:** Four modules — `engine.py` (QObject emitting gaze signals), `widgets.py` (all QWidget subclasses + draw_glow), `calibrator.py` (refactored to accept engine), `main.py` (entry + MainWindow + wiring).

**Tech Stack:** Python 3.10, PySide6, OpenCV, MediaPipe, scikit-learn, numpy

---

### Task 1: Create `engine.py` — GazeEngine QObject

**Files:**
- Create: `eye_tracker/engine.py`

The engine owns camera, MediaPipe, and calibration model. It emits a signal with gaze data every tick.

- [ ] **Step 1: Write engine.py**

```python
"""Gaze tracking engine — camera, MediaPipe, calibration, smoothing."""

import os
import numpy as np
import cv2
import mediapipe as mp
from PySide6.QtCore import QObject, QTimer, Signal

FaceMesh = mp.solutions.face_mesh.FaceMesh
RIGHT_IRIS  = [468, 469, 470, 471, 472]
LEFT_IRIS   = [473, 474, 475, 476, 477]
R_EYE_OUTER, R_EYE_INNER = 33, 133
L_EYE_INNER, L_EYE_OUTER = 362, 263


def extract_features(face_landmarks):
    lm = face_landmarks.landmark
    ri = np.mean([[lm[i].x, lm[i].y] for i in RIGHT_IRIS], axis=0)
    re = np.array([(lm[R_EYE_OUTER].x + lm[R_EYE_INNER].x) / 2,
                    (lm[R_EYE_OUTER].y + lm[R_EYE_INNER].y) / 2])
    li = np.mean([[lm[i].x, lm[i].y] for i in LEFT_IRIS], axis=0)
    le = np.array([(lm[L_EYE_INNER].x + lm[L_EYE_OUTER].x) / 2,
                    (lm[L_EYE_INNER].y + lm[L_EYE_OUTER].y) / 2])
    return np.concatenate([ri - re, li - le]).astype(np.float32)


class GazeEngine(QObject):
    """Owns camera, MediaPipe, and calibration model. Emits gaze updates."""

    gaze_updated = Signal(float, float, float, float, bool)
    # x, y, vx, vy, tracking

    def __init__(self, screen_w, screen_h):
        super().__init__()
        self.sw = screen_w
        self.sh = screen_h
        self.alpha = 0.12          # EMA smoothing factor

        self._gaze_x = screen_w / 2
        self._gaze_y = screen_h / 2
        self._prev_x = self._gaze_x
        self._prev_y = self._gaze_y
        self._frame = 0
        self._coef = None
        self._intercept = None
        self._x_mean = None
        self._x_std = None
        self._scale_x = 1.0
        self._scale_y = 1.0
        self._cap = None
        self._face_mesh = None
        self._timer = None

    # ── lifecycle ────────────────────────────────────────────────

    def start_camera(self):
        self._cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
        self._cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        self._cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        self._face_mesh = FaceMesh(
            static_image_mode=False, max_num_faces=1,
            refine_landmarks=True,
            min_detection_confidence=0.5, min_tracking_confidence=0.5,
        )
        self._timer = QTimer()
        self._timer.timeout.connect(self._tick)
        self._timer.start(33)  # ~30 Hz

    def stop_camera(self):
        if self._timer:
            self._timer.stop()
        if self._cap:
            self._cap.release()
        if self._face_mesh:
            self._face_mesh.close()

    def is_camera_ok(self):
        return self._cap is not None and self._cap.isOpened()

    # ── calibration ──────────────────────────────────────────────

    def load_calibration(self, path):
        calib = np.load(path)
        self._coef = calib["coef"]
        self._intercept = calib["intercept"]
        self._x_mean = calib["x_mean"]
        self._x_std = calib["x_std"]
        self._scale_x = self.sw / float(calib["screen_w"])
        self._scale_y = self.sh / float(calib["screen_h"])

    def has_calibration(self, path):
        return os.path.exists(path)

    def predict(self, features):
        """Map iris features → screen coordinates."""
        if self._coef is None:
            return self._gaze_x, self._gaze_y
        x_norm = (features - self._x_mean) / self._x_std
        pred = self._coef @ x_norm + self._intercept
        pred[0] *= self._scale_x
        pred[1] *= self._scale_y
        return float(pred[0]), float(pred[1])

    # ── camera access (used by calibrator) ───────────────────────

    def read_camera(self):
        """Read one frame, return (bgr_frame, results_or_none)."""
        if not self._cap or not self._face_mesh:
            return None, None
        ret, frame = self._cap.read()
        if not ret:
            return None, None
        frame = cv2.flip(frame, 1)
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        rgb.flags.writeable = False
        results = self._face_mesh.process(rgb)
        return frame, results

    # ── smoothing control ────────────────────────────────────────

    def set_smoothing(self, alpha):
        self.alpha = max(0.01, min(0.5, alpha))

    def reset_position(self):
        self._gaze_x = self.sw / 2
        self._gaze_y = self.sh / 2
        self._prev_x = self._gaze_x
        self._prev_y = self._gaze_y

    # ── internal tick ────────────────────────────────────────────

    def _tick(self):
        self._frame += 1
        frame, results = self.read_camera()
        tracking = False
        vx, vy = 0.0, 0.0

        if results and results.multi_face_landmarks:
            feats = extract_features(results.multi_face_landmarks[0])
            px, py = self.predict(feats)

            # dead zone
            dx, dy = px - self._gaze_x, py - self._gaze_y
            if abs(dx) < 2.0 and abs(dy) < 2.0:
                px, py = self._gaze_x, self._gaze_y

            # EMA smoothing
            self._gaze_x = self.alpha * px + (1 - self.alpha) * self._gaze_x
            self._gaze_y = self.alpha * py + (1 - self.alpha) * self._gaze_y

            # velocity (px/frame)
            vx = self._gaze_x - self._prev_x
            vy = self._gaze_y - self._prev_y
            self._prev_x = self._gaze_x
            self._prev_y = self._gaze_y
            tracking = True

        self.gaze_updated.emit(self._gaze_x, self._gaze_y, vx, vy, tracking)
```

- [ ] **Step 2: Commit**

```bash
git add eye_tracker/engine.py
git commit -m "feat: add GazeEngine QObject for camera + MediaPipe + prediction"
```

---

### Task 2: Create `widgets.py` — draw_glow with fluid deformation

**Files:**
- Create: `eye_tracker/widgets.py`

Start with just the rendering function. Windows added in subsequent tasks.

- [ ] **Step 1: Write widgets.py with draw_glow**

```python
"""All QWidget subclasses and the aperture renderer."""

import math
import ctypes
from PySide6.QtWidgets import (QWidget, QPushButton, QLabel, QSlider,
                                QVBoxLayout, QHBoxLayout)
from PySide6.QtCore import Qt, QPointF
from PySide6.QtGui import (QPainter, QColor, QRadialGradient, QConicalGradient,
                            QBrush, QFont, QPen)

# Windows API flags for click-through overlay
GWL_EXSTYLE = -20
WS_EX_TRANSPARENT = 0x00000020
WS_EX_LAYERED = 0x00080000
WS_EX_NOACTIVATE = 0x08000000


def win_set_exstyle(hwnd, flags):
    ex = ctypes.windll.user32.GetWindowLongW(int(hwnd), GWL_EXSTYLE)
    ctypes.windll.user32.SetWindowLongW(int(hwnd), GWL_EXSTYLE, ex | flags)


def draw_glow(painter, x, y, vx, vy, pulse):
    """Hollow ring aperture with velocity-based fluid deformation.

    When moving, the ring stretches along the velocity vector like a droplet.
    """
    r = 42
    speed = math.hypot(vx, vy)

    # Compute deformation
    stretch = 1.0 + speed * 0.08
    if stretch > 1.8:
        stretch = 1.8
    inv_stretch = 1.0 / stretch
    angle = math.degrees(math.atan2(vy, vx)) if speed > 0.5 else 0.0

    painter.save()
    painter.translate(x, y)
    painter.rotate(angle)
    painter.scale(stretch, inv_stretch)

    # ── Outer soft glow ──────────────────────────────────────────
    g_out = QRadialGradient(QPointF(0, 0), r + 16)
    g_out.setColorAt(0.7, QColor(0, 200, 255, 25))
    g_out.setColorAt(0.85, QColor(0, 160, 255, 12))
    g_out.setColorAt(1.0, QColor(0, 0, 0, 0))
    painter.setBrush(QBrush(g_out))
    painter.setPen(Qt.NoPen)
    painter.drawEllipse(QPointF(0, 0), r + 16, r + 16)

    # ── Main ring — hollow, conical gradient ─────────────────────
    rot = pulse * 45
    grad = QConicalGradient(QPointF(0, 0), rot)
    grad.setColorAt(0.00, QColor(0, 240, 255, 230))
    grad.setColorAt(0.25, QColor(80, 180, 255, 200))
    grad.setColorAt(0.50, QColor(0, 255, 200, 240))
    grad.setColorAt(0.75, QColor(80, 180, 255, 200))
    grad.setColorAt(1.00, QColor(0, 240, 255, 230))

    pen = QPen(QBrush(grad), 6)
    pen.setCapStyle(Qt.RoundCap)
    painter.setPen(pen)
    painter.setBrush(Qt.NoBrush)
    painter.drawEllipse(QPointF(0, 0), r, r)

    # ── Inner highlight ring ─────────────────────────────────────
    pen2 = QPen(QColor(180, 255, 255, 90), 1.5)
    painter.setPen(pen2)
    painter.drawEllipse(QPointF(0, 0), r - 3, r - 3)

    # ── Orbiting bright droplets ─────────────────────────────────
    for i in range(3):
        a = math.radians(rot + i * 120)
        dx = r * math.cos(a)
        dy = r * math.sin(a)
        s = 3 + pulse * 1.5
        g_drop = QRadialGradient(QPointF(dx, dy), s)
        g_drop.setColorAt(0.0, QColor(255, 255, 255, 200))
        g_drop.setColorAt(1.0, QColor(0, 200, 255, 0))
        painter.setBrush(QBrush(g_drop))
        painter.setPen(Qt.NoPen)
        painter.drawEllipse(QPointF(dx, dy), s, s)

    painter.restore()
```

- [ ] **Step 2: Commit**

```bash
git add eye_tracker/widgets.py
git commit -m "feat: add draw_glow with velocity-based fluid deformation"
```

---

### Task 3: Add OverlayWindow + CaptureWindow to `widgets.py`

**Files:**
- Modify: `eye_tracker/widgets.py` — append to file

- [ ] **Step 1: Append OverlayWindow and CaptureWindow classes**

Add after the `draw_glow` function:

```python
# ═══════════════════════════════════════════════════════════════════
# Overlay window — transparent, click-through, always-on-top
# ═══════════════════════════════════════════════════════════════════

class OverlayWindow(QWidget):
    def __init__(self, geo):
        super().__init__()
        self.setGeometry(geo)
        self.setWindowFlags(
            Qt.FramelessWindowHint | Qt.WindowStaysOnTopHint |
            Qt.Tool | Qt.NoDropShadowWindowHint
        )
        self.setAttribute(Qt.WA_TranslucentBackground)
        self.setAttribute(Qt.WA_ShowWithoutActivating)
        self.setStyleSheet("background: transparent;")
        self._gx = geo.width() // 2
        self._gy = geo.height() // 2
        self._vx = 0.0
        self._vy = 0.0
        self._pulse = 0.0
        self._tracking = False

    def showEvent(self, event):
        super().showEvent(event)
        win_set_exstyle(self.winId(), WS_EX_TRANSPARENT | WS_EX_LAYERED | WS_EX_NOACTIVATE)

    def paintEvent(self, event):
        p = QPainter(self)
        p.setRenderHint(QPainter.Antialiasing)
        if self._tracking:
            draw_glow(p, self._gx, self._gy, self._vx, self._vy, self._pulse)

    def update_state(self, x, y, vx, vy, pulse, tracking):
        self._gx, self._gy = x, y
        self._vx, self._vy = vx, vy
        self._pulse = pulse
        self._tracking = tracking
        self.repaint()


# ═══════════════════════════════════════════════════════════════════
# Capture window — black background for OBS chroma-key
# ═══════════════════════════════════════════════════════════════════

class CaptureWindow(QWidget):
    def __init__(self, geo):
        super().__init__()
        self.setGeometry(geo)
        self.setWindowTitle("Eye Tracker - Capture")
        self.setWindowFlags(
            Qt.FramelessWindowHint | Qt.Tool | Qt.NoDropShadowWindowHint
        )
        self.setAutoFillBackground(True)
        self.setStyleSheet("background: #000000;")
        self._gx = geo.width() // 2
        self._gy = geo.height() // 2
        self._vx = 0.0
        self._vy = 0.0
        self._pulse = 0.0
        self._tracking = False

    def paintEvent(self, event):
        p = QPainter(self)
        p.setRenderHint(QPainter.Antialiasing)
        if self._tracking:
            draw_glow(p, self._gx, self._gy, self._vx, self._vy, self._pulse)

    def update_state(self, x, y, vx, vy, pulse, tracking):
        self._gx, self._gy = x, y
        self._vx, self._vy = vx, vy
        self._pulse = pulse
        self._tracking = tracking
        self.repaint()
```

- [ ] **Step 2: Commit**

```bash
git add eye_tracker/widgets.py
git commit -m "feat: add OverlayWindow and CaptureWindow with fluid aperture"
```

---

### Task 4: Add MainWindow to `widgets.py`

**Files:**
- Modify: `eye_tracker/widgets.py` — append to file

- [ ] **Step 1: Append MainWindow class**

Add after CaptureWindow:

```python
# ═══════════════════════════════════════════════════════════════════
# Main window — dark panel with controls
# ═══════════════════════════════════════════════════════════════════

class MainWindow(QWidget):
    """Dark-themed control panel for the eye tracker."""

    start_clicked = Signal()
    calibrate_clicked = Signal()
    hide_clicked = Signal()
    smoothing_changed = Signal(float)

    def __init__(self):
        super().__init__()
        self.setWindowTitle("Eye Tracker")
        self.setFixedSize(340, 280)
        self.setWindowFlags(Qt.WindowCloseButtonHint | Qt.WindowStaysOnTopHint)
        self.setStyleSheet("background: #1a1a1a;")

        layout = QVBoxLayout()
        layout.setContentsMargins(20, 16, 20, 16)
        layout.setSpacing(10)

        # Status row: LED + label
        status_row = QHBoxLayout()
        self.led = QLabel("●")
        self.led.setStyleSheet("color: #555; font-size: 18px;")
        self.led.setFixedWidth(24)
        self.status_label = QLabel("待机中")
        self.status_label.setStyleSheet("color: #999; font-size: 13px;")
        status_row.addWidget(self.led)
        status_row.addWidget(self.status_label)
        status_row.addStretch()
        layout.addLayout(status_row)

        # Gaze coordinates
        self.coord_label = QLabel("视线: (— , —)")
        self.coord_label.setStyleSheet("color: #666; font-size: 12px;")
        layout.addWidget(self.coord_label)

        # Smoothing slider
        smooth_row = QHBoxLayout()
        smooth_label = QLabel("平滑:")
        smooth_label.setStyleSheet("color: #888; font-size: 12px;")
        smooth_label.setFixedWidth(40)
        self.smooth_slider = QSlider(Qt.Horizontal)
        self.smooth_slider.setRange(2, 50)
        self.smooth_slider.setValue(12)
        self.smooth_slider.setStyleSheet("""
            QSlider::groove:horizontal {
                height: 4px; background: #333; border-radius: 2px;
            }
            QSlider::handle:horizontal {
                width: 14px; height: 14px; margin: -5px 0;
                background: #0af; border-radius: 7px;
            }
            QSlider::sub-page:horizontal {
                background: #0af; border-radius: 2px;
            }
        """)
        self.smooth_val = QLabel("0.12")
        self.smooth_val.setStyleSheet("color: #0af; font-size: 12px;")
        self.smooth_val.setFixedWidth(36)
        self.smooth_slider.valueChanged.connect(
            lambda v: self.smoothing_changed.emit(v / 100.0)
        )
        self.smooth_slider.valueChanged.connect(
            lambda v: self.smooth_val.setText(f"{v/100:.2f}")
        )
        smooth_row.addWidget(smooth_label)
        smooth_row.addWidget(self.smooth_slider)
        smooth_row.addWidget(self.smooth_val)
        layout.addLayout(smooth_row)

        layout.addSpacing(6)

        # Start button
        self.start_btn = QPushButton("开始追踪")
        self.start_btn.setStyleSheet("""
            QPushButton {
                padding: 12px; font-size: 15px; font-weight: bold;
                border-radius: 6px; border: 2px solid #0af;
                background: transparent; color: #0af;
            }
            QPushButton:hover {
                background: #0af; color: #111;
            }
            QPushButton:disabled {
                border-color: #444; color: #444;
            }
        """)
        self.start_btn.clicked.connect(self.start_clicked)
        layout.addWidget(self.start_btn)

        # Bottom row: hide + calibrate
        btn_row = QHBoxLayout()
        btn_row.setSpacing(10)

        self.hide_btn = QPushButton("隐藏")
        self.hide_btn.setEnabled(False)
        self.hide_btn.setStyleSheet("""
            QPushButton {
                padding: 10px; font-size: 13px; border-radius: 5px;
                border: 1px solid #666; background: transparent; color: #aaa;
            }
            QPushButton:hover { border-color: #0af; color: #0af; }
            QPushButton:disabled { border-color: #333; color: #444; }
        """)
        self.hide_btn.clicked.connect(self.hide_clicked)

        self.cal_btn = QPushButton("校准")
        self.cal_btn.setStyleSheet("""
            QPushButton {
                padding: 10px; font-size: 13px; border-radius: 5px;
                border: 1px solid #666; background: transparent; color: #aaa;
            }
            QPushButton:hover { border-color: #fa0; color: #fa0; }
        """)
        self.cal_btn.clicked.connect(self.calibrate_clicked)

        btn_row.addWidget(self.hide_btn)
        btn_row.addWidget(self.cal_btn)
        layout.addLayout(btn_row)

        self.setLayout(layout)

    def update_status(self, tracking, gaze_x, gaze_y):
        if tracking:
            self.led.setStyleSheet("color: #0f8; font-size: 18px;")
            self.status_label.setText("追踪中")
            self.status_label.setStyleSheet("color: #0f8; font-size: 13px;")
            self.coord_label.setText(f"视线: ({int(gaze_x)}, {int(gaze_y)})")
            self.coord_label.setStyleSheet("color: #aaa; font-size: 12px;")
        else:
            self.led.setStyleSheet("color: #555; font-size: 18px;")
            self.status_label.setText("待机中")
            self.status_label.setStyleSheet("color: #999; font-size: 13px;")
            self.coord_label.setText("视线: (— , —)")
            self.coord_label.setStyleSheet("color: #666; font-size: 12px;")

    def set_tracking_active(self, active):
        if active:
            self.start_btn.setText("停止追踪")
            self.start_btn.setStyleSheet("""
                QPushButton {
                    padding: 12px; font-size: 15px; font-weight: bold;
                    border-radius: 6px; border: 2px solid #f55;
                    background: transparent; color: #f55;
                }
                QPushButton:hover { background: #f55; color: #111; }
            """)
            self.hide_btn.setEnabled(True)
        else:
            self.start_btn.setText("开始追踪")
            self.start_btn.setStyleSheet("""
                QPushButton {
                    padding: 12px; font-size: 15px; font-weight: bold;
                    border-radius: 6px; border: 2px solid #0af;
                    background: transparent; color: #0af;
                }
                QPushButton:hover { background: #0af; color: #111; }
            """)
            self.hide_btn.setEnabled(False)
```

- [ ] **Step 2: Add Signal import to widgets.py top**

Edit the import line at the top of `widgets.py`:

Replace:
```python
from PySide6.QtCore import Qt, QPointF
```
With:
```python
from PySide6.QtCore import Qt, QPointF, Signal
```

- [ ] **Step 3: Commit**

```bash
git add eye_tracker/widgets.py
git commit -m "feat: add MainWindow dark-themed control panel"
```

---

### Task 5: Refactor `calibrator.py` to accept GazeEngine

**Files:**
- Modify: `eye_tracker/calibrator.py`

Rewrite calibrator to accept a GazeEngine instance and use its camera/face_mesh.

- [ ] **Step 1: Rewrite calibrator.py**

```python
"""Gaze calibration — fullscreen 5-point flow, now accepts GazeEngine."""

import os
import sys
import numpy as np
from sklearn.linear_model import Ridge
from PySide6.QtWidgets import QApplication, QWidget
from PySide6.QtCore import Qt, QTimer, QPoint
from PySide6.QtGui import QPainter, QColor, QFont

CALIB_POINTS = [
    (0.1, 0.1), (0.9, 0.1), (0.5, 0.5), (0.1, 0.9), (0.9, 0.9),
]
SAMPLES_PER_POINT = 50
SETTLE_SECONDS = 1.0
PREP_SECONDS = 2.0

RIGHT_IRIS  = [468, 469, 470, 471, 472]
LEFT_IRIS   = [473, 474, 475, 476, 477]
R_EYE_OUTER, R_EYE_INNER = 33, 133
L_EYE_INNER, L_EYE_OUTER = 362, 263


def _extract_features(face_landmarks):
    lm = face_landmarks.landmark
    ri = np.mean([[lm[i].x, lm[i].y] for i in RIGHT_IRIS], axis=0)
    re = np.array([(lm[R_EYE_OUTER].x + lm[R_EYE_INNER].x) / 2,
                    (lm[R_EYE_OUTER].y + lm[R_EYE_INNER].y) / 2])
    li = np.mean([[lm[i].x, lm[i].y] for i in LEFT_IRIS], axis=0)
    le = np.array([(lm[L_EYE_INNER].x + lm[L_EYE_OUTER].x) / 2,
                    (lm[L_EYE_INNER].y + lm[L_EYE_OUTER].y) / 2])
    return np.concatenate([ri - re, li - le]).astype(np.float32)


class CalibrationWindow(QWidget):
    def __init__(self, engine):
        super().__init__()
        self.engine = engine
        screen = QApplication.primaryScreen().geometry()

        self.setCursor(Qt.BlankCursor)
        self.setStyleSheet("background: #000;")
        self.setWindowFlags(Qt.FramelessWindowHint | Qt.WindowStaysOnTopHint)
        self.setGeometry(screen)

        self.current_idx = 0
        self.phase = "prep"
        self.phase_timer = 0.0
        self.collected = 0
        self.samples = []

        self.timer = QTimer()
        self.timer.timeout.connect(self._tick)
        self.timer.setInterval(33)

        self.showFullScreen()
        self.timer.start()

    def paintEvent(self, event):
        p = QPainter(self)
        p.setRenderHint(QPainter.Antialiasing)
        w, h = self.width(), self.height()
        p.fillRect(self.rect(), QColor(0, 0, 0))

        if self.current_idx >= len(CALIB_POINTS):
            p.setPen(QColor(255, 255, 255))
            p.setFont(QFont("Arial", 28, QFont.Bold))
            p.drawText(self.rect(), Qt.AlignCenter, "校准完成！\n按 Esc 关闭")
            p.end()
            return

        px, py = CALIB_POINTS[self.current_idx]
        cx, cy = int(px * w), int(py * h)

        if self.phase == "prep":
            radius = 14 + 5 * np.sin(self.phase_timer * 4)
            color = QColor(255, 180, 50)
        elif self.phase == "settle":
            radius = 22
            color = QColor(0, 255, 100)
        else:
            radius = 18
            color = QColor(0, 200, 255)
            bar_w, bar_h = 300, 6
            bx = (w - bar_w) // 2
            by_ = h - 50
            p.fillRect(bx, by_, bar_w, bar_h, QColor(60, 60, 60))
            p.fillRect(bx, by_, int(bar_w * self.collected / SAMPLES_PER_POINT), bar_h, QColor(0, 200, 255))

        p.setBrush(color)
        p.setPen(Qt.NoPen)
        p.drawEllipse(QPoint(cx, cy), radius, radius)

        if self.phase == "prep":
            sec = max(1, int(np.ceil(PREP_SECONDS - self.phase_timer)))
            p.setPen(QColor(255, 255, 255, 180))
            p.setFont(QFont("Arial", 64, QFont.Bold))
            p.drawText(self.rect(), Qt.AlignCenter, str(sec))

        p.setPen(QColor(120, 120, 120))
        p.setFont(QFont("Arial", 13))
        labels = {"prep": "准备注视", "settle": "保持注视...", "collect": "采集中..."}
        p.drawText(20, 30, f"[{self.current_idx + 1}/{len(CALIB_POINTS)}]  {labels[self.phase]}")
        p.end()

    def keyPressEvent(self, event):
        if event.key() == Qt.Key_Escape:
            self.timer.stop()
            QApplication.quit()

    def _tick(self):
        dt = 0.033
        if self.current_idx >= len(CALIB_POINTS):
            return
        self.phase_timer += dt
        if self.phase == "prep" and self.phase_timer >= PREP_SECONDS:
            self.phase = "settle"
            self.phase_timer = 0
        elif self.phase == "settle" and self.phase_timer >= SETTLE_SECONDS:
            self.phase = "collect"
            self.phase_timer = 0
            self.collected = 0
        elif self.phase == "collect":
            self._collect()
            if self.collected >= SAMPLES_PER_POINT:
                self.current_idx += 1
                self.phase = "prep"
                self.phase_timer = 0
                if self.current_idx >= len(CALIB_POINTS):
                    self._finish()
        self.repaint()

    def _collect(self):
        _, results = self.engine.read_camera()
        if results and results.multi_face_landmarks:
            feats = _extract_features(results.multi_face_landmarks[0])
            px, py = CALIB_POINTS[self.current_idx]
            target = np.array([px * self.width(), py * self.height()])
            self.samples.append((feats, target))
            self.collected += 1

    def _finish(self):
        self.timer.stop()
        if len(self.samples) < 30:
            print("[!] 样本不足，请重新校准", file=sys.stderr)
            QApplication.quit()
            return
        X = np.array([s[0] for s in self.samples], dtype=np.float32)
        y = np.array([s[1] for s in self.samples], dtype=np.float32)
        x_mean = X.mean(axis=0)
        x_std  = X.std(axis=0) + 1e-6
        X_norm = (X - x_mean) / x_std
        model = Ridge(alpha=0.5)
        model.fit(X_norm, y)
        screen = QApplication.primaryScreen().geometry()
        save_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "calibration.npz")
        np.savez(save_path,
                 coef=model.coef_.astype(np.float32),
                 intercept=model.intercept_.astype(np.float32),
                 x_mean=x_mean.astype(np.float32),
                 x_std=x_std.astype(np.float32),
                 screen_w=screen.width(), screen_h=screen.height())
        print(f"[OK] 校准参数已保存: {save_path}")
        QApplication.quit()


def run_calibration(engine):
    """Run fullscreen calibration using the given engine. Returns 0 on success."""
    app = QApplication.instance()
    if app is None:
        app = QApplication(sys.argv)
    window = CalibrationWindow(engine)
    app.exec()
    return 0
```

- [ ] **Step 2: Commit**

```bash
git add eye_tracker/calibrator.py
git commit -m "refactor: calibrator uses GazeEngine for camera + MediaPipe"
```

---

### Task 6: Create `main.py` — entry point

**Files:**
- Create: `eye_tracker/main.py`

- [ ] **Step 1: Write main.py**

```python
"""Eye Tracker V2 — single-app entry point with integrated calibration."""

import os
import sys
import numpy as np
from PySide6.QtWidgets import QApplication
from PySide6.QtCore import Qt, QTimer

from engine import GazeEngine
from calibrator import run_calibration
from widgets import MainWindow, OverlayWindow, CaptureWindow


class App:
    def __init__(self):
        self.app = QApplication(sys.argv)
        screen = self.app.primaryScreen().geometry()
        self.sw = screen.width()
        self.sh = screen.height()

        self.engine = GazeEngine(self.sw, self.sh)
        self.main_window = MainWindow()
        self.overlay = None
        self.capture = None
        self.tracking_active = False
        self.overlay_visible = True
        self._frame = 0

        # Wire signals
        self.main_window.start_clicked.connect(self._toggle_tracking)
        self.main_window.calibrate_clicked.connect(self._run_calibration)
        self.main_window.hide_clicked.connect(self._toggle_overlay)
        self.main_window.smoothing_changed.connect(self.engine.set_smoothing)

        calib_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "calibration.npz")

        # Start camera
        self.engine.start_camera()
        if not self.engine.is_camera_ok():
            self.main_window.status_label.setText("摄像头不可用")
            self.main_window.start_btn.setEnabled(False)

        # Auto-calibrate if needed
        if not self.engine.has_calibration(calib_path):
            self.main_window.status_label.setText("需要校准...")
            self._run_calibration()

        self.engine.load_calibration(calib_path)

        # Update timer for main window status
        self._status_timer = QTimer()
        self._status_timer.timeout.connect(self._update_ui)
        self._status_timer.start(50)

        self.main_window.show()

    def _toggle_tracking(self):
        if self.tracking_active:
            self._stop_tracking()
        else:
            self._start_tracking()

    def _start_tracking(self):
        screen_geo = self.app.primaryScreen().geometry()
        self.overlay = OverlayWindow(screen_geo)
        self.capture = CaptureWindow(screen_geo)
        self.engine.reset_position()
        self.tracking_active = True
        self.overlay_visible = True
        self.main_window.set_tracking_active(True)
        self.engine.gaze_updated.connect(self._on_gaze)

    def _stop_tracking(self):
        self.engine.gaze_updated.disconnect(self._on_gaze)
        if self.overlay:
            self.overlay.close()
            self.overlay = None
        if self.capture:
            self.capture.close()
            self.capture = None
        self.tracking_active = False
        self.overlay_visible = True
        self.main_window.set_tracking_active(False)

    def _toggle_overlay(self):
        if self.overlay:
            self.overlay_visible = not self.overlay_visible
            self.overlay.setVisible(self.overlay_visible)

    def _run_calibration(self):
        was_tracking = self.tracking_active
        if was_tracking:
            self._stop_tracking()
        self.engine.stop_camera()
        self.main_window.hide()
        run_calibration(self.engine)
        self.engine.start_camera()
        calib_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "calibration.npz")
        if self.engine.has_calibration(calib_path):
            self.engine.load_calibration(calib_path)
        self.main_window.show()
        if was_tracking:
            self._start_tracking()

    def _on_gaze(self, x, y, vx, vy, tracking):
        self._frame += 1
        pulse = np.sin(self._frame * 0.07)
        if self.overlay:
            self.overlay.update_state(x, y, vx, vy, pulse, tracking)
        if self.capture:
            self.capture.update_state(x, y, vx, vy, pulse, tracking)

    def _update_ui(self):
        if self.tracking_active and hasattr(self.engine, '_gaze_x'):
            self.main_window.update_status(
                True, self.engine._gaze_x, self.engine._gaze_y)
        else:
            self.main_window.update_status(False, 0, 0)

    def run(self):
        self.app.exec()


if __name__ == "__main__":
    App().run()
```

- [ ] **Step 2: Commit**

```bash
git add eye_tracker/main.py
git commit -m "feat: add main.py — unified app entry with integrated calibration"
```

---

### Task 7: Cleanup + Final Test

**Files:**
- Delete: `eye_tracker/tracker.py` (old standalone tracker)
- Keep: `eye_tracker/calibrator.py` (can still be run standalone if needed, but also used by main.py via `run_calibration(engine)`)

- [ ] **Step 1: Remove old tracker.py**

```bash
git rm eye_tracker/tracker.py
```

- [ ] **Step 2: Commit**

```bash
git commit -m "chore: remove old standalone tracker.py"
```

- [ ] **Step 3: Verify project structure**

```bash
ls eye_tracker/
# Expected: calibrator.py  engine.py  environment.yml  main.py  widgets.py
```

- [ ] **Step 4: Run the app**

```bash
python eye_tracker/main.py
```

Expected: Main window appears. If no calibration, auto-starts calibration. After calibration, tracking works. Buttons respond correctly.

---

## Self-Review

**Spec coverage check:**
- Single app entry ✅ (main.py)
- Dark-themed main panel ✅ (MainWindow in widgets.py)
- Three buttons: start, hide, calibrate ✅
- Auto-detect calibration ✅ (main.py checks calibration.npz)
- Fluid-deforming aperture ✅ (draw_glow with velocity stretch)
- Smoothing slider ✅ (MainWindow slider → engine.set_smoothing)
- Calibration uses engine's camera ✅ (calibrator.py refactored)
- Overlay + Capture windows ✅ (OverlayWindow, CaptureWindow)
- State machine (idle → tracking → calibrate) ✅ (App class methods)

**Placeholder scan:** No TBD, TODO, or vague descriptions. All code is complete.

**Type consistency:**
- `draw_glow(painter, x, y, vx, vy, pulse)` — consistent across all callers
- `engine.gaze_updated.emit(x, y, vx, vy, tracking)` — matches Signal signature
- `engine.read_camera()` → `(frame, results)` — used by calibrator
- `engine.load_calibration(path)` — called by App
- `MainWindow.update_status(tracking, x, y)` — called by App

All consistent.
