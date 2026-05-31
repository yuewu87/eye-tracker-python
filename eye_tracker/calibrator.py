"""视线校准模块 — 全屏 9 点校准流程，使用引擎共享的摄像头和特征提取。"""

import os
import sys
import numpy as np
from sklearn.ensemble import GradientBoostingRegressor
from PySide6.QtWidgets import QApplication, QWidget
from PySide6.QtCore import Qt, QTimer, QPoint, Signal, QEventLoop
from PySide6.QtGui import QPainter, QColor, QFont

from engine import extract_features

# 5 点校准（四角 + 中心），减少眼部疲劳
CALIB_POINTS = [
    (0.1, 0.1), (0.9, 0.1), (0.5, 0.5), (0.1, 0.9), (0.9, 0.9),
]

SAMPLES_PER_POINT = 60   # 每点采集帧数
SETTLE_SECONDS = 0.8     # 注视稳定时间
PREP_SECONDS = 1.0       # 倒计时准备时间


class CalibrationWindow(QWidget):
    """全屏校准窗口：依次显示 9 个注视点，采集虹膜特征训练 Ridge 回归。"""

    calibration_done = Signal()

    def __init__(self, engine):
        super().__init__()
        self.engine = engine
        print("[i] 正在初始化校准...")

        screen = QApplication.primaryScreen().geometry()
        print(f"[i] 屏幕: {screen.width()}x{screen.height()}")

        self.setCursor(Qt.BlankCursor)
        self.setStyleSheet("background: #1a1a1a;")
        self.setWindowFlags(Qt.FramelessWindowHint | Qt.WindowStaysOnTopHint)
        self.setGeometry(screen)

        self.current_idx = 0          # 当前校准点索引
        self.phase = "prep"           # prep → settle → collect
        self.phase_timer = 0.0        # 当前阶段计时器
        self.collected = 0            # 当前点已采集帧数
        self.samples = []             # [(features, target), ...]

        self.timer = QTimer()
        self.timer.timeout.connect(self._tick)
        self.timer.setInterval(33)    # ~30 fps

        print("[i] 显示校准窗口...")
        self.showFullScreen()
        self.timer.start()

    # ── 绘制 ──────────────────────────────────────────────────────

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

            if self.phase == "prep":
                radius = 30 + 8 * np.sin(self.phase_timer * 3)
                color = QColor(200, 160, 255, 120)
            elif self.phase == "settle":
                radius = 36
                color = QColor(180, 140, 240, 160)
            else:
                radius = 28
                color = QColor(160, 180, 255, 200)
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
                p.setPen(QColor(255, 255, 255, 100))
                p.setFont(QFont("Arial", 36))
                p.drawText(self.rect(), Qt.AlignCenter, str(sec))

            p.setPen(QColor(120, 120, 120))
            p.setFont(QFont("Arial", 13))
            labels = {"prep": "准备注视", "settle": "保持注视...", "collect": "采集中..."}
            p.drawText(20, 30, f"[{self.current_idx + 1}/{len(CALIB_POINTS)}]  {labels[self.phase]}")
            p.end()
        except Exception as e:
            print(f"[!] paint 错误: {e}", file=sys.stderr)
            import traceback
            traceback.print_exc()

    def keyPressEvent(self, event):
        if event.key() == Qt.Key_Escape:
            self.timer.stop()
            self.close()
            self.calibration_done.emit()

    # ── 主循环 ────────────────────────────────────────────────────

    def _tick(self):
        try:
            self._tick_impl()
        except Exception as e:
            print(f"[!] tick 错误: {e}", file=sys.stderr)
            import traceback
            traceback.print_exc()

    def _tick_impl(self):
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
            self._collect_frame()
            if self.collected >= SAMPLES_PER_POINT:
                self.current_idx += 1
                self.phase = "prep"
                self.phase_timer = 0
                if self.current_idx >= len(CALIB_POINTS):
                    self._finish()
        self.repaint()

    def _collect_frame(self):
        """从引擎读取一帧，提取特征并与注视点坐标配对存储。"""
        _, results = self.engine.read_camera()
        if results is None:
            return
        if not results.multi_face_landmarks:
            return
        feats = extract_features(results.multi_face_landmarks[0])
        if feats is None:
            return
        px, py = CALIB_POINTS[self.current_idx]
        target = np.array([px * self.width(), py * self.height()])
        self.samples.append((feats, target))
        self.collected += 1

    # ── 拟合与保存 ────────────────────────────────────────────────

    def _finish(self):
        self.timer.stop()

        if len(self.samples) < 30:
            print("[!] 样本不足，请重新校准", file=sys.stderr)
        else:
            X = np.array([s[0] for s in self.samples], dtype=np.float64)
            y = np.array([s[1] for s in self.samples], dtype=np.float64)
            self.x_mean = X.mean(axis=0)
            self.x_std  = X.std(axis=0) + 1e-6
            X_norm = (X - self.x_mean) / self.x_std

            # 梯度提升回归：树模型天然处理非线性映射
            model = GradientBoostingRegressor(
                n_estimators=100, max_depth=4, learning_rate=0.1,
                random_state=42)
            model.fit(X_norm, y)
            print(f"[i] GBR 训练完成, R²={model.score(X_norm, y):.3f}")

            screen = QApplication.primaryScreen().geometry()
            save_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "calibration.npz")
            np.savez(save_path,
                     x_mean=self.x_mean.astype(np.float64),
                     x_std=self.x_std.astype(np.float64),
                     screen_w=screen.width(),
                     screen_h=screen.height(),
                     model=model)
            print(f"[OK] 校准参数已保存: {save_path}")
            print(f"     样本数: {len(self.samples)}, 屏幕: {screen.width()}x{screen.height()}")

        self.close()
        self.calibration_done.emit()


def run_calibration(engine):
    """运行全屏校准，阻塞直到完成。返回 0。"""
    window = CalibrationWindow(engine)
    loop = QEventLoop()
    window.calibration_done.connect(loop.quit)
    loop.exec()
    return 0
