"""所有 QWidget 子类和光圈渲染器。"""

import math
import ctypes
from PySide6.QtWidgets import (QWidget, QPushButton, QLabel, QSlider,
                                QVBoxLayout, QHBoxLayout)
from PySide6.QtCore import Qt, QPointF, Signal
from PySide6.QtGui import (QPainter, QColor, QRadialGradient,
                            QBrush, QFont, QPen)

# ═══════════════════════════════════════════════════════════════════
# Windows API — 点击穿透和窗口样式
# ═══════════════════════════════════════════════════════════════════

GWL_EXSTYLE = -20
WS_EX_TRANSPARENT = 0x00000020
WS_EX_LAYERED = 0x00080000
WS_EX_NOACTIVATE = 0x08000000


def win_set_exstyle(hwnd, flags):
    """给窗口附加扩展样式（如点击穿透）。"""
    ex = ctypes.windll.user32.GetWindowLongW(int(hwnd), GWL_EXSTYLE)
    ctypes.windll.user32.SetWindowLongW(int(hwnd), GWL_EXSTYLE, ex | flags)


# ═══════════════════════════════════════════════════════════════════
# 光圈渲染
# ═══════════════════════════════════════════════════════════════════

_R = 56                # 光圈基础半径
_SPEED_THRESH = 15.0   # 速度阈值
_TAIL_LEN = 0.35       # 拖尾长度系数
_TAIL_SEG = 3          # 拖尾段数


def draw_glow(painter, x, y, vx, vy, pulse):
    """空心圆环光圈：白环 + 紫色外发光 + 移动时彗星拖尾。"""
    speed = math.hypot(vx, vy)

    painter.save()

    # ── 紫色外发光（仅向外扩散） ──────────────────────────────
    glow_r = _R * 2.0
    g = QRadialGradient(QPointF(x, y), _R, QPointF(x, y), glow_r)
    g.setColorAt(0.0, QColor(255, 255, 255, 0))      # 内圈无发光
    g.setColorAt(0.15, QColor(200, 140, 255, 50))     # 淡紫过渡
    g.setColorAt(0.4, QColor(140, 60, 220, 30))       # 紫色扩散
    g.setColorAt(1.0, QColor(0, 0, 0, 0))             # 边缘消失
    painter.setBrush(QBrush(g))
    painter.setPen(Qt.NoPen)
    painter.drawEllipse(QPointF(x, y), glow_r, glow_r)

    # ── 拖尾 — 大幅度移动时显示 ──────────────────────────────
    if speed > _SPEED_THRESH:
        tail_len = speed * _TAIL_LEN
        nx = -vx / speed  # 速度反方向
        ny = -vy / speed

        for i in range(_TAIL_SEG):
            t = (i + 1) / _TAIL_SEG
            tx = x + nx * tail_len * t
            ty = y + ny * tail_len * t
            seg_r = _R + _R * t * 0.6
            alpha = int(100 * (1.0 - t))
            pen = QPen(QColor(160, 100, 240, alpha), 1.5)
            pen.setCapStyle(Qt.RoundCap)
            painter.setPen(pen)
            painter.setBrush(Qt.NoBrush)
            painter.drawEllipse(QPointF(tx, ty), seg_r, seg_r)

    # ── 主环 — 白→紫渐变空心圆环 ────────────────────────────
    pen_main = QPen(QColor(255, 255, 255, 240), 2)
    pen_main.setCapStyle(Qt.RoundCap)
    painter.setPen(pen_main)
    painter.setBrush(Qt.NoBrush)
    painter.drawEllipse(QPointF(x, y), _R, _R)

    # 内圈淡紫高光环
    pen_inner = QPen(QColor(200, 160, 255, 120), 2)
    pen_inner.setCapStyle(Qt.RoundCap)
    painter.setPen(pen_inner)
    painter.drawEllipse(QPointF(x, y), _R - 3, _R - 3)

    painter.restore()


# ═══════════════════════════════════════════════════════════════════
# Overlay 窗口 — 全屏透明、点击穿透、始终置顶
# ═══════════════════════════════════════════════════════════════════

class OverlayWindow(QWidget):
    """用户可见的光圈覆盖层。"""

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
        # 设为点击穿透 + 不抢焦点
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
# Capture 窗口 — 全屏黑底，供 OBS 色度键抠图
# ═══════════════════════════════════════════════════════════════════

class CaptureWindow(QWidget):
    """OBS 捕捉用窗口：纯黑背景 + 光圈，色度键抠除黑色即可叠加。"""

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
# 主控制面板 — 暗色主题
# ═══════════════════════════════════════════════════════════════════

class MainWindow(QWidget):
    """暗色主题控制面板：状态灯、坐标显示、平滑滑块、控制按钮。"""

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

        # 状态行：指示灯 + 文字
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

        # 视线坐标
        self.coord_label = QLabel("视线: (— , —)")
        self.coord_label.setStyleSheet("color: #666; font-size: 12px;")
        layout.addWidget(self.coord_label)

        # 平滑滑块
        smooth_row = QHBoxLayout()
        smooth_label = QLabel("平滑:")
        smooth_label.setStyleSheet("color: #888; font-size: 12px;")
        smooth_label.setFixedWidth(40)
        self.smooth_slider = QSlider(Qt.Horizontal)
        self.smooth_slider.setRange(5, 100)
        self.smooth_slider.setValue(30)
        self.smooth_slider.setStyleSheet("""
            QSlider::groove:horizontal { height: 4px; background: #333; border-radius: 2px; }
            QSlider::handle:horizontal { width: 14px; height: 14px; margin: -5px 0; background: #0af; border-radius: 7px; }
            QSlider::sub-page:horizontal { background: #0af; border-radius: 2px; }
        """)
        self.smooth_val = QLabel("0.30")
        self.smooth_val.setStyleSheet("color: #0af; font-size: 12px;")
        self.smooth_val.setFixedWidth(36)
        self.smooth_slider.valueChanged.connect(lambda v: self.smoothing_changed.emit(v / 100.0))
        self.smooth_slider.valueChanged.connect(lambda v: self.smooth_val.setText(f"{v/100:.2f}"))
        smooth_row.addWidget(smooth_label)
        smooth_row.addWidget(self.smooth_slider)
        smooth_row.addWidget(self.smooth_val)
        layout.addLayout(smooth_row)
        layout.addSpacing(6)

        # 开始/停止追踪按钮
        self.start_btn = QPushButton("开始追踪")
        self.start_btn.setStyleSheet("""
            QPushButton { padding: 12px; font-size: 15px; font-weight: bold;
            border-radius: 6px; border: 2px solid #0af; background: transparent; color: #0af; }
            QPushButton:hover { background: #0af; color: #111; }
            QPushButton:disabled { border-color: #444; color: #444; }
        """)
        self.start_btn.clicked.connect(self.start_clicked)
        layout.addWidget(self.start_btn)

        # 底部按钮：隐藏 + 校准
        btn_row = QHBoxLayout()
        btn_row.setSpacing(10)
        self.hide_btn = QPushButton("隐藏")
        self.hide_btn.setEnabled(False)
        self.hide_btn.setStyleSheet("""
            QPushButton { padding: 10px; font-size: 13px; border-radius: 5px;
            border: 1px solid #666; background: transparent; color: #aaa; }
            QPushButton:hover { border-color: #0af; color: #0af; }
            QPushButton:disabled { border-color: #333; color: #444; }
        """)
        self.hide_btn.clicked.connect(self.hide_clicked)

        self.cal_btn = QPushButton("校准")
        self.cal_btn.setStyleSheet("""
            QPushButton { padding: 10px; font-size: 13px; border-radius: 5px;
            border: 1px solid #666; background: transparent; color: #aaa; }
            QPushButton:hover { border-color: #fa0; color: #fa0; }
        """)
        self.cal_btn.clicked.connect(self.calibrate_clicked)
        btn_row.addWidget(self.hide_btn)
        btn_row.addWidget(self.cal_btn)
        layout.addLayout(btn_row)
        self.setLayout(layout)

    def update_status(self, tracking, gaze_x, gaze_y):
        """更新状态显示，仅在状态变化时刷新样式表。"""
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
        """切换开始/停止按钮样式和隐藏按钮可用状态。"""
        if active:
            self.start_btn.setText("停止追踪")
            self.start_btn.setStyleSheet("""
                QPushButton { padding: 12px; font-size: 15px; font-weight: bold;
                border-radius: 6px; border: 2px solid #f55; background: transparent; color: #f55; }
                QPushButton:hover { background: #f55; color: #111; }
            """)
            self.hide_btn.setEnabled(True)
        else:
            self.start_btn.setText("开始追踪")
            self.start_btn.setStyleSheet("""
                QPushButton { padding: 12px; font-size: 15px; font-weight: bold;
                border-radius: 6px; border: 2px solid #0af; background: transparent; color: #0af; }
                QPushButton:hover { background: #0af; color: #111; }
            """)
            self.hide_btn.setEnabled(False)
