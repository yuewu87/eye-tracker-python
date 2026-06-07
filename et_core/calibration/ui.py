"""校准 UI — 可选子模块，依赖 PySide6。"""

import os
import numpy as np
from PySide6.QtWidgets import QApplication, QWidget
from PySide6.QtCore import Qt, QTimer, QPoint, QEventLoop, Signal
from PySide6.QtGui import QPainter, QColor, QFont, QPen

from et_core.calibration.collector import CalibrationCollector
from et_core.calibration.trainer import train, save, evaluate

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

        self.timer = QTimer()
        self.timer.timeout.connect(self._tick)
        self.timer.setInterval(33)

        print("[i] 显示校准窗口...")
        self.showFullScreen()
        self.timer.start()

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
        import et_core
        pkg_dir = os.path.dirname(os.path.abspath(et_core.__file__))
        save_path = os.path.join(pkg_dir, fname)
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
        p.drawText(self.rect(), Qt.AlignCenter,
                   f"注视中心十字\n{max(0, self.timer_count - self.frame)//30 + 1} 秒")
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
    """多显示器校准：依次在每块屏幕上显示准星。"""

    calibration_done = Signal()

    def __init__(self, camera_processor, monitors: list):
        super().__init__()
        self.camera = camera_processor
        self.monitors = monitors
        print(f"[i] 显示器校准 — {len(monitors)} 块屏幕")

        self.setCursor(Qt.BlankCursor)
        self.setStyleSheet("background: #1a1a1a;")
        self.setWindowFlags(Qt.FramelessWindowHint | Qt.WindowStaysOnTopHint)

        self.current_idx = -1
        self.phase = "prep"
        self.phase_timer = 0.0
        self.collector = CalibrationCollector(samples_needed=60, settle_seconds=0.5)
        self._offsets = []

        self.timer = QTimer()
        self.timer.timeout.connect(self._tick)
        self.timer.setInterval(33)
        self._advance()

    def _advance(self):
        if self.current_idx >= 0 and self.current_idx < len(self.monitors):
            self.collector.get_samples()
        self.current_idx += 1
        if self.current_idx >= len(self.monitors):
            self._finish()
            return
        mx, my, mw, mh = self.monitors[self.current_idx]
        self.setGeometry(mx, my, mw, mh)
        self.phase = "prep"
        self.phase_timer = 0.0
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

        cx, cy = w // 2, h // 2

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
                self.timer.stop()
                self.hide()
                self._advance()
                return
        self.repaint()

    def _finish(self):
        self.timer.stop()
        self.close()
        self.calibration_done.emit()

    def get_offsets(self) -> list[float]:
        return self._offsets
