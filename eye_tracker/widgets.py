"""All QWidget subclasses and the aperture renderer."""

import math
import ctypes
from PySide6.QtWidgets import (QWidget, QPushButton, QLabel, QSlider,
                                QVBoxLayout, QHBoxLayout)
from PySide6.QtCore import Qt, QPointF, Signal
from PySide6.QtGui import (QPainter, QColor, QRadialGradient,
                            QBrush, QFont, QPen)

# ── Windows API helpers ────────────────────────────────────────────

GWL_EXSTYLE = -20
WS_EX_TRANSPARENT = 0x00000020
WS_EX_LAYERED = 0x00080000
WS_EX_NOACTIVATE = 0x08000000


def win_set_exstyle(hwnd, flags):
    ex = ctypes.windll.user32.GetWindowLongW(int(hwnd), GWL_EXSTYLE)
    ctypes.windll.user32.SetWindowLongW(int(hwnd), GWL_EXSTYLE, ex | flags)


# ═══════════════════════════════════════════════════════════════════
# Aperture renderer
# ═══════════════════════════════════════════════════════════════════

_R = 42  # ring radius


def draw_glow(painter, x, y, vx, vy, pulse):
    """Hollow ring + outer glow, stretches along velocity for fluid feel."""
    speed = math.hypot(vx, vy)
    stretch = 1.0 + speed * 0.08
    if stretch > 1.8:
        stretch = 1.8
    angle = math.degrees(math.atan2(vy, vx)) if speed > 0.5 else 0.0

    painter.save()
    painter.translate(x, y)
    painter.rotate(angle)
    painter.scale(stretch, 1.0 / stretch)

    # Outer glow
    g = QRadialGradient(QPointF(0, 0), _R + 20)
    g.setColorAt(0.70, QColor(0, 200, 255, 35))
    g.setColorAt(0.85, QColor(0, 160, 255, 15))
    g.setColorAt(1.00, QColor(0, 0, 0, 0))
    painter.setBrush(QBrush(g))
    painter.setPen(Qt.NoPen)
    painter.drawEllipse(QPointF(0, 0), _R + 20, _R + 20)

    # Hollow ring
    pen = QPen(QColor(0, 220, 255, 220), 5)
    pen.setCapStyle(Qt.RoundCap)
    painter.setPen(pen)
    painter.setBrush(Qt.NoBrush)
    painter.drawEllipse(QPointF(0, 0), _R, _R)

    painter.restore()


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
        self.update()


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
        self.update()


# ═══════════════════════════════════════════════════════════════════
# Main window — dark panel with controls
# ═══════════════════════════════════════════════════════════════════

class MainWindow(QWidget):
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
        self._prev_tracking = None

        layout = QVBoxLayout()
        layout.setContentsMargins(20, 16, 20, 16)
        layout.setSpacing(10)

        # Status
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

        self.coord_label = QLabel("视线: (— , —)")
        self.coord_label.setStyleSheet("color: #666; font-size: 12px;")
        layout.addWidget(self.coord_label)

        # Smoothing
        smooth_row = QHBoxLayout()
        smooth_label = QLabel("平滑:")
        smooth_label.setStyleSheet("color: #888; font-size: 12px;")
        smooth_label.setFixedWidth(40)
        self.smooth_slider = QSlider(Qt.Horizontal)
        self.smooth_slider.setRange(2, 50)
        self.smooth_slider.setValue(12)
        self.smooth_slider.setStyleSheet("""
            QSlider::groove:horizontal { height: 4px; background: #333; border-radius: 2px; }
            QSlider::handle:horizontal { width: 14px; height: 14px; margin: -5px 0; background: #0af; border-radius: 7px; }
            QSlider::sub-page:horizontal { background: #0af; border-radius: 2px; }
        """)
        self.smooth_val = QLabel("0.12")
        self.smooth_val.setStyleSheet("color: #0af; font-size: 12px;")
        self.smooth_val.setFixedWidth(36)
        self.smooth_slider.valueChanged.connect(lambda v: self.smoothing_changed.emit(v / 100.0))
        self.smooth_slider.valueChanged.connect(lambda v: self.smooth_val.setText(f"{v/100:.2f}"))
        smooth_row.addWidget(smooth_label)
        smooth_row.addWidget(self.smooth_slider)
        smooth_row.addWidget(self.smooth_val)
        layout.addLayout(smooth_row)
        layout.addSpacing(6)

        # Start button
        self.start_btn = QPushButton("开始追踪")
        self.start_btn.setStyleSheet("""
            QPushButton { padding: 12px; font-size: 15px; font-weight: bold; border-radius: 6px; border: 2px solid #0af; background: transparent; color: #0af; }
            QPushButton:hover { background: #0af; color: #111; }
            QPushButton:disabled { border-color: #444; color: #444; }
        """)
        self.start_btn.clicked.connect(self.start_clicked)
        layout.addWidget(self.start_btn)

        # Bottom buttons
        btn_row = QHBoxLayout()
        btn_row.setSpacing(10)
        self.hide_btn = QPushButton("隐藏")
        self.hide_btn.setEnabled(False)
        self.hide_btn.setStyleSheet("""
            QPushButton { padding: 10px; font-size: 13px; border-radius: 5px; border: 1px solid #666; background: transparent; color: #aaa; }
            QPushButton:hover { border-color: #0af; color: #0af; }
            QPushButton:disabled { border-color: #333; color: #444; }
        """)
        self.hide_btn.clicked.connect(self.hide_clicked)

        self.cal_btn = QPushButton("校准")
        self.cal_btn.setStyleSheet("""
            QPushButton { padding: 10px; font-size: 13px; border-radius: 5px; border: 1px solid #666; background: transparent; color: #aaa; }
            QPushButton:hover { border-color: #fa0; color: #fa0; }
        """)
        self.cal_btn.clicked.connect(self.calibrate_clicked)
        btn_row.addWidget(self.hide_btn)
        btn_row.addWidget(self.cal_btn)
        layout.addLayout(btn_row)
        self.setLayout(layout)

    def update_status(self, tracking, gaze_x, gaze_y):
        changed = (tracking != self._prev_tracking)
        if changed:
            self._prev_tracking = tracking
            if tracking:
                self.led.setStyleSheet("color: #0f8; font-size: 18px;")
                self.status_label.setText("追踪中")
                self.status_label.setStyleSheet("color: #0f8; font-size: 13px;")
            else:
                self.led.setStyleSheet("color: #555; font-size: 18px;")
                self.status_label.setText("待机中")
                self.status_label.setStyleSheet("color: #999; font-size: 13px;")
        if tracking:
            self.coord_label.setText(f"视线: ({int(gaze_x)}, {int(gaze_y)})")
            self.coord_label.setStyleSheet("color: #aaa; font-size: 12px;")
        elif changed:
            self.coord_label.setText("视线: (— , —)")
            self.coord_label.setStyleSheet("color: #666; font-size: 12px;")

    def set_tracking_active(self, active):
        if active:
            self.start_btn.setText("停止追踪")
            self.start_btn.setStyleSheet("""
                QPushButton { padding: 12px; font-size: 15px; font-weight: bold; border-radius: 6px; border: 2px solid #f55; background: transparent; color: #f55; }
                QPushButton:hover { background: #f55; color: #111; }
            """)
            self.hide_btn.setEnabled(True)
        else:
            self.start_btn.setText("开始追踪")
            self.start_btn.setStyleSheet("""
                QPushButton { padding: 12px; font-size: 15px; font-weight: bold; border-radius: 6px; border: 2px solid #0af; background: transparent; color: #0af; }
                QPushButton:hover { background: #0af; color: #111; }
            """)
            self.hide_btn.setEnabled(False)
