"""视线校准模块 — 全屏 7 点校准流程，使用引擎共享的摄像头和特征提取。"""

import os
import sys
import numpy as np
from sklearn.linear_model import RidgeCV
from sklearn.preprocessing import PolynomialFeatures
from PySide6.QtWidgets import QApplication, QWidget
from PySide6.QtCore import Qt, QTimer, QPoint, Signal, QEventLoop
from PySide6.QtGui import QPainter, QColor, QFont, QPen

from engine import extract_features

# 7 点校准：覆盖全屏 + 上行加强（补偿摄像头在下巴位置的向上视角盲区）
CALIB_POINTS = [
    (0.08, 0.05), (0.50, 0.05), (0.92, 0.05),   # 上行：顶部
    (0.08, 0.92), (0.92, 0.92),                   # 下行：底部
    (0.50, 0.50), (0.50, 0.92),                   # 中心 + 下中
]

SAMPLES_RGB = 120
SAMPLES_IR  = 30   # IR 帧率低，减半采样
SETTLE_SECONDS = 0.8
PREP_SECONDS = 1.0


class CalibrationWindow(QWidget):
    """全屏校准窗口：依次显示 7 个注视点，采集虹膜特征训练 Ridge 回归。"""

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

        self.current_idx = 0
        self.phase = "prep"
        self.phase_timer = 0.0
        self.collected = 0
        self.samples = []
        self.samples_needed = SAMPLES_IR if getattr(engine, 'use_ir', False) else SAMPLES_RGB
        if engine.use_ir:
            print("[i] IR 模式，减少采样数")

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
                p.fillRect(bx, by_, int(bar_w * self.collected / self.samples_needed), bar_h, QColor(0, 200, 255))

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
            if self.collected >= self.samples_needed:
                self._filter_samples()
                self.current_idx += 1
                self.phase = "prep"
                self.phase_timer = 0
                self.collected = 0
                if self.current_idx >= len(CALIB_POINTS):
                    self._finish()
        self.repaint()

    def _filter_samples(self):
        """过滤当前校准点的离群样本（z-score > 2.5 的特征剔除）"""
        if len(self.samples) < 20:
            return
        # 只过滤最近采集的这一批
        start = max(0, len(self.samples) - self.samples_needed)
        recent = [(f, t) for f, t in self.samples[start:]]
        feats = np.array([f for f, _ in recent])
        mean = feats.mean(axis=0)
        std = feats.std(axis=0) + 1e-6
        z = np.abs((feats - mean) / std).max(axis=1)
        keep = z < 2.5
        n_removed = (~keep).sum()
        if n_removed > 0:
            # 替换为保留样本
            self.samples[start:] = [recent[i] for i in range(len(recent)) if keep[i]]
            print(f"  [i] 点{self.current_idx+1}: 剔除 {n_removed} 个离群帧")

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

            # 多项式特征 + Ridge 回归 → 连续平滑曲面
            poly = PolynomialFeatures(degree=2, include_bias=False)
            X_poly = poly.fit_transform(X_norm)
            n_feat = X_poly.shape[1]
            model = RidgeCV(alphas=[0.01, 0.1, 0.5, 1.0, 5.0, 10.0, 50.0])
            model.fit(X_poly, y)
            print(f"[i] Poly(deg=2): 5→{n_feat} 维, best α={model.alpha_}")

            screen = QApplication.primaryScreen().geometry()
            # IR 模式使用独立的校准文件
            fname = "calibration_ir.npz" if self.engine.use_ir else "calibration.npz"
            save_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), fname)
            np.savez(save_path,
                     x_mean=self.x_mean.astype(np.float64),
                     x_std=self.x_std.astype(np.float64),
                     screen_w=screen.width(),
                     screen_h=screen.height(),
                     model=model,
                     poly_degree=2,
                     poly_features_in=5)
            print(f"[OK] 校准参数已保存: {save_path}")
            print(f"     样本数: {len(self.samples)}, 屏幕: {screen.width()}x{screen.height()}")

            # 校准质量评估
            X_all = np.array([s[0] for s in self.samples], dtype=np.float64)
            y_all = np.array([s[1] for s in self.samples], dtype=np.float64)
            Xn = (X_all - self.x_mean) / self.x_std
            Xn_poly = poly.transform(Xn)
            y_pred = model.predict(Xn_poly)
            errors = np.sqrt(((y_pred - y_all) ** 2).sum(axis=1))
            print(f"[i] 校准误差: 平均={errors.mean():.1f}px 最大={errors.max():.1f}px")

            # 每点误差
            per_point = SAMPLES_IR if self.engine.use_ir else SAMPLES_RGB
            for i, (px, py) in enumerate(CALIB_POINTS):
                start = i * per_point
                end = start + per_point
                if start < len(errors):
                    e = errors[start:end].mean()
                    print(f"     点{i+1} ({px:.0%},{py:.0%}): 平均误差 {e:.0f}px")

        self.close()
        self.calibration_done.emit()


def run_calibration(engine):
    """运行全屏 5 点校准，阻塞直到完成。"""
    window = CalibrationWindow(engine)
    loop = QEventLoop()
    window.calibration_done.connect(loop.quit)
    loop.exec()


class CenterCalibWindow(QWidget):
    """单点中心校准：注视屏幕中央 2 秒，修正漂移。"""
    calibration_done = Signal()

    def __init__(self, engine):
        super().__init__()
        self.engine = engine
        screen = QApplication.primaryScreen().geometry()
        self.setCursor(Qt.BlankCursor)
        self.setStyleSheet("background: #1a1a1a;")
        self.setWindowFlags(Qt.FramelessWindowHint | Qt.WindowStaysOnTopHint)
        self.setGeometry(screen)
        self.samples = []
        self.timer_count = int(2.5 * 30)  # 2.5 秒 @ 30fps
        self.frame = 0

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
        progress = self.frame / max(self.timer_count, 1)

        # 中心圆点
        p.setBrush(QColor(180, 160, 255, 160))
        p.setPen(Qt.NoPen)
        r = 30 + 6 * np.sin(self.frame * 0.1)
        p.drawEllipse(QPoint(cx, cy), r, r)

        # 进度环
        pen = QPen(QColor(200, 180, 255, 180), 3)
        p.setPen(pen)
        p.setBrush(Qt.NoBrush)
        p.drawEllipse(QPoint(cx, cy), 50, 50)

        p.setPen(QColor(255, 255, 255, 120))
        p.setFont(QFont("Arial", 18))
        p.drawText(self.rect(), Qt.AlignCenter, f"注视中心点\n{max(0, self.timer_count - self.frame)//30 + 1} 秒")

        p.end()

    def keyPressEvent(self, event):
        if event.key() == Qt.Key_Escape:
            self.timer.stop()
            self.close()
            self.calibration_done.emit()

    def _tick(self):
        self.frame += 1
        _, results = self.engine.read_camera()
        if results and results.multi_face_landmarks:
            feats = extract_features(results.multi_face_landmarks[0])
            self.samples.append(feats)
        self.repaint()
        if self.frame >= self.timer_count:
            self._finish()

    def _finish(self):
        self.timer.stop()
        self.frame = self.timer_count + 1  # 防止 _tick 再次触发 _finish
        if len(self.samples) < 10:
            self.close()
            self.calibration_done.emit()
            return

        X = np.array(self.samples, dtype=np.float32)
        x_mean = X.mean(axis=0)
        x_std = X.std(axis=0) + 1e-6
        X_norm = ((X - x_mean) / x_std).mean(axis=0, keepdims=True)

        calib_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "calibration.npz")
        calib = np.load(calib_path, allow_pickle=True)
        model = calib["model"].item()
        screen_w = float(calib["screen_w"])
        screen_h = float(calib["screen_h"])
        scale_x = self.engine.screen_w / screen_w
        scale_y = self.engine.screen_h / screen_h

        # 多项式变换（与主校准一致）
        if "poly_degree" in calib:
            from sklearn.preprocessing import PolynomialFeatures
            poly = PolynomialFeatures(degree=int(calib["poly_degree"]), include_bias=False)
            poly.fit(np.zeros((1, int(calib["poly_features_in"]))))
            X_norm = poly.transform(X_norm)

        pred = model.predict(X_norm)[0]
        pred[0] *= scale_x
        pred[1] *= scale_y

        # 计算中心偏移
        cx = self.engine.screen_w / 2
        cy = self.engine.screen_h / 2
        offset_x = cx - pred[0]
        offset_y = cy - pred[1]

        print(f"[i] 中心偏移: ({offset_x:.1f}, {offset_y:.1f}) px")

        # 修正整个模型：对所有训练样本的 target 加偏移，重新训练
        # 简化方案：在引擎中存储 bias 修正量
        self.engine.bias_x = offset_x
        self.engine.bias_y = offset_y

        self.close()
        self.calibration_done.emit()


def run_center_calibration(engine):
    """运行中心单点校准，修正漂移。"""
    window = CenterCalibWindow(engine)
    loop = QEventLoop()
    window.calibration_done.connect(loop.quit)
    loop.exec()
