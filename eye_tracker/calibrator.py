"""Gaze calibration: 5-point calibration with MediaPipe iris detection + Ridge regression."""

import os
import sys
import numpy as np
import cv2
import mediapipe as mp
from sklearn.linear_model import Ridge
from PySide6.QtWidgets import QApplication, QWidget
from PySide6.QtCore import Qt, QTimer, QPoint
from PySide6.QtGui import QPainter, QColor, QFont

# ── MediaPipe setup ──────────────────────────────────────────────
FaceMesh = mp.solutions.face_mesh.FaceMesh

RIGHT_IRIS  = [468, 469, 470, 471, 472]
LEFT_IRIS   = [473, 474, 475, 476, 477]
R_EYE_OUTER, R_EYE_INNER = 33, 133
L_EYE_INNER, L_EYE_OUTER = 362, 263

CALIB_POINTS = [
    (0.1, 0.1), (0.9, 0.1), (0.5, 0.5), (0.1, 0.9), (0.9, 0.9),
]

SAMPLES_PER_POINT = 50
SETTLE_SECONDS = 1.0
PREP_SECONDS = 2.0


class CalibrationWindow(QWidget):
    def __init__(self):
        super().__init__()
        print("[i] 正在初始化校准...")

        screen = QApplication.primaryScreen().geometry()
        print(f"[i] 屏幕: {screen.width()}x{screen.height()}")

        self.setCursor(Qt.BlankCursor)
        self.setStyleSheet("background: #111;")
        self.setWindowFlags(
            Qt.FramelessWindowHint |
            Qt.WindowStaysOnTopHint
        )
        self.setGeometry(screen)

        self.current_idx = 0
        self.phase = "prep"
        self.phase_timer = 0.0
        self.collected = 0
        self.samples = []

        print("[i] 打开摄像头...")
        self.cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
        if not self.cap.isOpened():
            print("[!] 无法打开摄像头")
            sys.exit(1)
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        print("[i] 摄像头已打开")

        print("[i] 加载 MediaPipe FaceMesh...")
        self.face_mesh = FaceMesh(
            static_image_mode=False,
            max_num_faces=1,
            refine_landmarks=True,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5,
        )
        print("[i] MediaPipe 已就绪")

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

            p.fillRect(self.rect(), QColor(0, 0, 0))

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
        except Exception as e:
            print(f"[!] paint 错误: {e}", file=sys.stderr)
            import traceback
            traceback.print_exc()

    def keyPressEvent(self, event):
        if event.key() == Qt.Key_Escape:
            self.timer.stop()
            if self.cap.isOpened():
                self.cap.release()
            QApplication.quit()

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
        ret, frame = self.cap.read()
        if not ret:
            return
        frame = cv2.flip(frame, 1)
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self.face_mesh.process(rgb)
        if not results.multi_face_landmarks:
            return
        feats = self._extract_features(results.multi_face_landmarks[0])
        if feats is None:
            return
        px, py = CALIB_POINTS[self.current_idx]
        target = np.array([px * self.width(), py * self.height()])
        self.samples.append((feats, target))
        self.collected += 1

    def _extract_features(self, face_landmarks):
        lm = face_landmarks.landmark
        ri = np.mean([[lm[i].x, lm[i].y] for i in RIGHT_IRIS], axis=0)
        re = np.array([(lm[R_EYE_OUTER].x + lm[R_EYE_INNER].x) / 2,
                       (lm[R_EYE_OUTER].y + lm[R_EYE_INNER].y) / 2])
        li = np.mean([[lm[i].x, lm[i].y] for i in LEFT_IRIS], axis=0)
        le = np.array([(lm[L_EYE_INNER].x + lm[L_EYE_OUTER].x) / 2,
                       (lm[L_EYE_INNER].y + lm[L_EYE_OUTER].y) / 2])
        return np.concatenate([ri - re, li - le]).astype(np.float32)

    def _finish(self):
        self.timer.stop()
        self.cap.release()
        self.face_mesh.close()

        if len(self.samples) < 30:
            print("[!] 样本不足，请重新校准", file=sys.stderr)
            QApplication.quit()
            return

        X = np.array([s[0] for s in self.samples], dtype=np.float32)
        y = np.array([s[1] for s in self.samples], dtype=np.float32)
        self.x_mean = X.mean(axis=0)
        self.x_std  = X.std(axis=0) + 1e-6
        X_norm = (X - self.x_mean) / self.x_std

        model = Ridge(alpha=0.5)
        model.fit(X_norm, y)

        screen = QApplication.primaryScreen().geometry()
        save_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "calibration.npz")
        np.savez(save_path,
                 coef=model.coef_.astype(np.float32),
                 intercept=model.intercept_.astype(np.float32),
                 x_mean=self.x_mean.astype(np.float32),
                 x_std=self.x_std.astype(np.float32),
                 screen_w=screen.width(),
                 screen_h=screen.height())
        print(f"[OK] 校准参数已保存: {save_path}")
        print(f"     样本数: {len(self.samples)}, 屏幕: {screen.width()}x{screen.height()}")


def run_calibration():
    app = QApplication(sys.argv)
    window = CalibrationWindow()
    sys.exit(app.exec())


if __name__ == "__main__":
    run_calibration()
