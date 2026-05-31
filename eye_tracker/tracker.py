"""Real-time gaze tracking with dual-window output for OBS streaming.

Overlay: fullscreen transparent overlay (user sees).
Capture: fullscreen black window for OBS Window Capture + Chroma Key.
Control: small visible window for status + toggle button.
"""

import os
import sys
import ctypes
from ctypes import wintypes
import numpy as np
import cv2
import mediapipe as mp
from PySide6.QtWidgets import (QApplication, QWidget, QPushButton,
                                QLabel, QVBoxLayout, QHBoxLayout)
from PySide6.QtCore import Qt, QTimer, QPointF
from PySide6.QtGui import (QPainter, QColor, QRadialGradient, QConicalGradient,
                            QBrush, QFont, QPen)

# ═══════════════════════════════════════════════════════════════════
# Windows API
# ═══════════════════════════════════════════════════════════════════

GWL_EXSTYLE = -20
WS_EX_TRANSPARENT = 0x00000020
WS_EX_LAYERED = 0x00080000
WS_EX_TOOLWINDOW = 0x00000080
WS_EX_NOACTIVATE = 0x08000000


def win_set_exstyle(hwnd, flags):
    ex = ctypes.windll.user32.GetWindowLongW(int(hwnd), GWL_EXSTYLE)
    ctypes.windll.user32.SetWindowLongW(int(hwnd), GWL_EXSTYLE, ex | flags)


# ═══════════════════════════════════════════════════════════════════
# MediaPipe
# ═══════════════════════════════════════════════════════════════════

FaceMesh = mp.solutions.face_mesh.FaceMesh
RIGHT_IRIS = [468, 469, 470, 471, 472]
LEFT_IRIS = [473, 474, 475, 476, 477]
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


# ═══════════════════════════════════════════════════════════════════
# Glow rendering
# ═══════════════════════════════════════════════════════════════════

def draw_glow(painter, x, y, pulse):
    """Hollow ring with fluid color flow — like an 'O' with animated gradient."""
    import math
    r = 42  # ring radius

    # ── Outer soft glow ──────────────────────────────────────────
    g_out = QRadialGradient(QPointF(x, y), r + 16)
    g_out.setColorAt(0.7, QColor(0, 200, 255, 30))
    g_out.setColorAt(0.85, QColor(0, 160, 255, 15))
    g_out.setColorAt(1.0, QColor(0, 0, 0, 0))
    painter.setBrush(QBrush(g_out))
    painter.setPen(Qt.NoPen)
    painter.drawEllipse(QPointF(x, y), r + 16, r + 16)

    # ── Main ring — hollow, rotating conical gradient ────────────
    angle = pulse * 45  # rotating gradient drives fluid feel
    grad = QConicalGradient(QPointF(x, y), angle)
    grad.setColorAt(0.00, QColor(0, 240, 255, 230))
    grad.setColorAt(0.25, QColor(80, 180, 255, 200))
    grad.setColorAt(0.50, QColor(0, 255, 200, 240))
    grad.setColorAt(0.75, QColor(80, 180, 255, 200))
    grad.setColorAt(1.00, QColor(0, 240, 255, 230))

    pen = QPen(QBrush(grad), 6)
    pen.setCapStyle(Qt.RoundCap)
    painter.setPen(pen)
    painter.setBrush(Qt.NoBrush)  # hollow!
    painter.drawEllipse(QPointF(x, y), r, r)

    # ── Inner edge highlight ─────────────────────────────────────
    pen2 = QPen(QColor(180, 255, 255, 90), 1.5)
    painter.setPen(pen2)
    painter.drawEllipse(QPointF(x, y), r - 3, r - 3)

    # ── Bright spot "droplets" orbiting the ring ─────────────────
    for i in range(3):
        a = math.radians(angle + i * 120)
        dx = r * math.cos(a)
        dy = r * math.sin(a)
        s = 3 + pulse * 1.5
        g_drop = QRadialGradient(QPointF(x + dx, y + dy), s)
        g_drop.setColorAt(0.0, QColor(255, 255, 255, 200))
        g_drop.setColorAt(1.0, QColor(0, 200, 255, 0))
        painter.setBrush(QBrush(g_drop))
        painter.setPen(Qt.NoPen)
        painter.drawEllipse(QPointF(x + dx, y + dy), s, s)


# ═══════════════════════════════════════════════════════════════════
# Overlay window (transparent, click-through)
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
        self._pulse = 0.0
        self._tracking = False

    def showEvent(self, event):
        super().showEvent(event)
        win_set_exstyle(self.winId(), WS_EX_TRANSPARENT | WS_EX_LAYERED | WS_EX_NOACTIVATE)

    def paintEvent(self, event):
        p = QPainter(self)
        p.setRenderHint(QPainter.Antialiasing)
        if self._tracking:
            draw_glow(p, self._gx, self._gy, self._pulse)

    def update_state(self, x, y, pulse, tracking):
        self._gx, self._gy = x, y
        self._pulse = pulse
        self._tracking = tracking
        self.repaint()


# ═══════════════════════════════════════════════════════════════════
# Capture window (black background, for OBS Chroma Key)
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
        self._pulse = 0.0
        self._tracking = False

    def paintEvent(self, event):
        p = QPainter(self)
        p.setRenderHint(QPainter.Antialiasing)
        if self._tracking:
            draw_glow(p, self._gx, self._gy, self._pulse)

    def update_state(self, x, y, pulse, tracking):
        self._gx, self._gy = x, y
        self._pulse = pulse
        self._tracking = tracking
        self.repaint()


# ═══════════════════════════════════════════════════════════════════
# Control panel (visible window with status + toggle)
# ═══════════════════════════════════════════════════════════════════

class ControlPanel(QWidget):
    def __init__(self, on_toggle, on_quit):
        super().__init__()
        self.setWindowTitle("Eye Tracker Control")
        self.setFixedSize(300, 180)
        self.setWindowFlags(
            Qt.WindowStaysOnTopHint | Qt.WindowCloseButtonHint
        )

        layout = QVBoxLayout()
        layout.setSpacing(10)

        self.status_label = QLabel("状态: 初始化中...")
        self.status_label.setStyleSheet("font-size: 14px; color: #ccc;")

        self.pos_label = QLabel("视线位置: (— , —)")
        self.pos_label.setStyleSheet("font-size: 12px; color: #888;")

        self.overlay_btn = QPushButton("Overlay: 显示中")
        self.overlay_btn.setStyleSheet("""
            QPushButton {
                padding: 8px; font-size: 14px; border-radius: 4px;
                background: #2a5; color: white;
            }
            QPushButton:hover { background: #3b6; }
        """)
        self.overlay_btn.clicked.connect(on_toggle)

        quit_btn = QPushButton("退出")
        quit_btn.setStyleSheet("""
            QPushButton {
                padding: 6px; font-size: 12px; border-radius: 4px;
                background: #555; color: #aaa;
            }
            QPushButton:hover { background: #c33; color: white; }
        """)
        quit_btn.clicked.connect(on_quit)

        layout.addWidget(self.status_label)
        layout.addWidget(self.pos_label)
        layout.addWidget(self.overlay_btn)
        layout.addWidget(quit_btn)
        self.setLayout(layout)

        self.setStyleSheet("background: #222;")

    def update_status(self, tracking, gaze_x, gaze_y, overlay_visible):
        if tracking:
            self.status_label.setText("状态: 追踪中 ✓")
            self.status_label.setStyleSheet("font-size: 14px; color: #5f5;")
        else:
            self.status_label.setText("状态: 未检测到人脸")
            self.status_label.setStyleSheet("font-size: 14px; color: #f55;")

        self.pos_label.setText(f"视线位置: ({int(gaze_x)}, {int(gaze_y)})")

        if overlay_visible:
            self.overlay_btn.setText("Overlay: 显示中")
            self.overlay_btn.setStyleSheet("""
                QPushButton { padding: 8px; font-size: 14px; border-radius: 4px;
                background: #2a5; color: white; }
                QPushButton:hover { background: #3b6; }
            """)
        else:
            self.overlay_btn.setText("Overlay: 已隐藏")
            self.overlay_btn.setStyleSheet("""
                QPushButton { padding: 8px; font-size: 14px; border-radius: 4px;
                background: #a33; color: white; }
                QPushButton:hover { background: #c44; }
            """)


# ═══════════════════════════════════════════════════════════════════
# Main tracker
# ═══════════════════════════════════════════════════════════════════

class GazeTracker:
    def __init__(self):
        screen_geo = QApplication.primaryScreen().geometry()
        sw, sh = screen_geo.width(), screen_geo.height()

        # Load calibration
        calib_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "calibration.npz")
        if not os.path.exists(calib_path):
            print("[!] 未找到校准文件，请先运行 calibrator.py")
            sys.exit(1)

        calib = np.load(calib_path)
        self.coef = calib["coef"]
        self.intercept = calib["intercept"]
        self.x_mean = calib["x_mean"]
        self.x_std = calib["x_std"]
        self.scale_x = sw / float(calib["screen_w"])
        self.scale_y = sh / float(calib["screen_h"])

        # Windows
        self.overlay = OverlayWindow(screen_geo)
        self.capture = CaptureWindow(screen_geo)
        self.overlay_visible = False  # start hidden; toggled below

        self.control = ControlPanel(self._toggle_overlay, self._quit)
        self.control.show()

        QApplication.processEvents()

        # Explicitly toggle ON after init to ensure proper render
        self._toggle_overlay()

        # Camera
        self.cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

        self.face_mesh = FaceMesh(
            static_image_mode=False,
            max_num_faces=1,
            refine_landmarks=True,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5,
        )

        # State
        self.gaze_x = sw / 2
        self.gaze_y = sh / 2
        self.frame = 0
        self.tracking = False

        # Timer
        self.timer = QTimer()
        self.timer.timeout.connect(self._tick)
        self.timer.start(33)

    def _tick(self):
        try:
            self._tick_impl()
        except KeyboardInterrupt:
            print("\n[i] 退出")
            QApplication.quit()
        except Exception as e:
            print(f"[!] 错误: {e}", file=sys.stderr)

    def _tick_impl(self):
        self.frame += 1
        pulse = np.sin(self.frame * 0.07)

        ret, frame = self.cap.read()
        if not ret:
            return

        frame = cv2.flip(frame, 1)
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        rgb.flags.writeable = False
        results = self.face_mesh.process(rgb)

        if results.multi_face_landmarks:
            feats = extract_features(results.multi_face_landmarks[0])
            x_norm = (feats - self.x_mean) / self.x_std
            pred = self.coef @ x_norm + self.intercept
            pred[0] *= self.scale_x
            pred[1] *= self.scale_y

            # Dead zone: ignore sub-pixel jitter
            dx, dy = pred[0] - self.gaze_x, pred[1] - self.gaze_y
            if abs(dx) < 2.0 and abs(dy) < 2.0:
                pred[0], pred[1] = self.gaze_x, self.gaze_y

            # Heavy EMA smoothing
            alpha = 0.12
            self.gaze_x = alpha * pred[0] + (1 - alpha) * self.gaze_x
            self.gaze_y = alpha * pred[1] + (1 - alpha) * self.gaze_y
            self.tracking = True
        else:
            self.tracking = False

        self.overlay.update_state(self.gaze_x, self.gaze_y, pulse, self.tracking)
        self.capture.update_state(self.gaze_x, self.gaze_y, pulse, self.tracking)
        self.control.update_status(self.tracking, self.gaze_x, self.gaze_y, self.overlay_visible)

    def _toggle_overlay(self):
        self.overlay_visible = not self.overlay_visible
        self.overlay.setVisible(self.overlay_visible)

    def _quit(self):
        self.timer.stop()
        self.cap.release()
        self.face_mesh.close()
        self.overlay.close()
        self.capture.close()
        self.control.close()
        QApplication.quit()


def main():
    app = QApplication(sys.argv)
    print("=" * 50)
    print("  Eye Tracker — 桌面视线追踪光圈")
    print("  控制面板 → 切换 Overlay 显示 / 退出")
    print("=" * 50)
    tracker = GazeTracker()
    app.exec()


if __name__ == "__main__":
    main()
