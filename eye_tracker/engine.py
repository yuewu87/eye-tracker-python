"""GazeEngine — QObject-based gaze tracking engine.

Owns the camera, MediaPipe FaceMesh, and calibration model.
Emits gaze data at ~30 Hz for use by the main overlay and calibrator.
"""

import os

import cv2
import mediapipe as mp
import numpy as np
from PySide6.QtCore import QObject, Signal, QTimer


# ═══════════════════════════════════════════════════════════════════
# Shared iris landmark indices
# ═══════════════════════════════════════════════════════════════════

RIGHT_IRIS = [468, 469, 470, 471, 472]
LEFT_IRIS = [473, 474, 475, 476, 477]
R_EYE_OUTER, R_EYE_INNER = 33, 133
L_EYE_INNER, L_EYE_OUTER = 362, 263

_FaceMesh = mp.solutions.face_mesh.FaceMesh


def extract_features(face_landmarks):
    """Extract 7-D gaze features including head-pose context.

    Returns float32[7]:
      [dx_r, dy_r, dx_l, dy_l,  nose_dx, nose_dy, eye_dist]
    """
    lm = face_landmarks.landmark

    # Iris centers
    ri = np.mean([[lm[i].x, lm[i].y] for i in RIGHT_IRIS], axis=0)
    li = np.mean([[lm[i].x, lm[i].y] for i in LEFT_IRIS], axis=0)

    # Eye centers (midpoint of eye corners)
    re = np.array([(lm[R_EYE_OUTER].x + lm[R_EYE_INNER].x) / 2,
                   (lm[R_EYE_OUTER].y + lm[R_EYE_INNER].y) / 2])
    le = np.array([(lm[L_EYE_INNER].x + lm[L_EYE_OUTER].x) / 2,
                   (lm[L_EYE_INNER].y + lm[L_EYE_OUTER].y) / 2])

    # Face reference: midpoint between the two eyes
    face_cx = (re[0] + le[0]) / 2
    face_cy = (re[1] + le[1]) / 2
    eye_dist = float(np.linalg.norm(re - le))

    # Nose tip (landmark 1) as head-pose proxy
    nose = np.array([lm[1].x, lm[1].y])
    nose_dx = nose[0] - face_cx
    nose_dy = nose[1] - face_cy

    # Iris offsets from eye centers
    dr = ri - re
    dl = li - le

    return np.array([dr[0], dr[1], dl[0], dl[1],
                     nose_dx, nose_dy, eye_dist], dtype=np.float32)


# ═══════════════════════════════════════════════════════════════════
# GazeEngine
# ═══════════════════════════════════════════════════════════════════


class GazeEngine(QObject):
    """QObject that encapsulates camera capture, face-mesh inference,
    calibration prediction, and EMA-smoothed gaze output."""

    gaze_updated = Signal(float, float, float, float, bool)
    """(gaze_x, gaze_y, velocity_x, velocity_y, tracking)"""

    # ── Lifecycle ──────────────────────────────────────────────────

    def __init__(self, screen_w: int, screen_h: int):
        super().__init__()
        self.screen_w = screen_w
        self.screen_h = screen_h

        # Smoothed gaze position (init at centre)
        self.gaze_x = screen_w / 2.0
        self.gaze_y = screen_h / 2.0
        self.prev_gaze_x = screen_w / 2.0
        self.prev_gaze_y = screen_h / 2.0

        # Smoothing factor (EMA)
        self.alpha = 0.18

        # Whether a face was detected on the most recent tick
        self.tracking = False

        # Camera / MediaPipe (created by start_camera)
        self.cap = None
        self.face_mesh = None

        # Calibration (loaded by load_calibration)
        self.coef = None
        self.intercept = None
        self.x_mean = None
        self.x_std = None
        self.scale_x = 1.0
        self.scale_y = 1.0
        self._has_calib = False

        # Internal tick timer
        self.timer = QTimer()
        self.timer.timeout.connect(self._tick)

    # ── Camera control ─────────────────────────────────────────────

    def start_camera(self):
        """Open the default camera and initialise MediaPipe FaceMesh."""
        self.cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
        if not self.cap.isOpened():
            print("Warning: camera could not be opened")
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
        """Stop the tick timer but keep camera and face_mesh alive (for calibration)."""
        self.timer.stop()

    def resume(self):
        """Restart the tick timer after pause()."""
        self.timer.start(33)

    def stop_camera(self):
        """Stop the tick timer and release camera / MediaPipe resources."""
        self.timer.stop()
        if self.cap is not None:
            if self.cap.isOpened():
                self.cap.release()
            self.cap = None
        if self.face_mesh is not None:
            self.face_mesh.close()
            self.face_mesh = None

    def is_camera_ok(self) -> bool:
        """Return whether the camera is currently open and readable."""
        return self.cap is not None and self.cap.isOpened()

    # ── Calibration loading ────────────────────────────────────────

    @staticmethod
    def has_calibration(path: str) -> bool:
        """Return True if a calibration file exists at *path*."""
        return os.path.exists(path)

    def load_calibration(self, path: str):
        """Load a calibration.npz file and compute screen-ratio scales.

        The file must contain keys: coef, intercept, x_mean, x_std,
        screen_w, screen_h.
        """
        calib = np.load(path)
        self.coef = calib["coef"]
        self.intercept = calib["intercept"]
        self.x_mean = calib["x_mean"]
        self.x_std = calib["x_std"]
        calib_w = float(calib["screen_w"])
        calib_h = float(calib["screen_h"])
        self.scale_x = self.screen_w / calib_w
        self.scale_y = self.screen_h / calib_h
        self._has_calib = True

    def predict(self, features: np.ndarray):
        """Apply the calibration model to a feature vector.

        Returns (x, y) gaze point in screen coordinates.
        """
        x_norm = (features - self.x_mean) / self.x_std
        pred = self.coef @ x_norm + self.intercept
        pred[0] *= self.scale_x
        pred[1] *= self.scale_y
        return float(np.clip(pred[0], 0, self.screen_w)), float(np.clip(pred[1], 0, self.screen_h))

    # ── Single-frame camera read (also used by calibrator) ─────────

    def read_camera(self):
        """Read and process one camera frame.

        Returns (bgr_frame, results_or_None).
        The BGR frame is flipped horizontally (mirror view).
        ``results`` is the MediaPipe face-mesh output, or None if no
        face was detected or the read failed.
        """
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

    # ── Smoothing helpers ──────────────────────────────────────────

    def set_smoothing(self, alpha: float):
        """Set the EMA smoothing factor (clamped to [0.01, 0.5]).

        Lower values = smoother / more laggy.
        Higher values = more responsive / more jittery.
        """
        self.alpha = max(0.01, min(0.5, alpha))

    def reset_position(self):
        """Reset the gaze position to the centre of the screen."""
        self.gaze_x = self.screen_w / 2.0
        self.gaze_y = self.screen_h / 2.0
        self.prev_gaze_x = self.gaze_x
        self.prev_gaze_y = self.gaze_y

    # ── Internal tick ──────────────────────────────────────────────

    def _tick(self):
        """Periodic callback: read camera, predict, smooth, emit."""
        _, results = self.read_camera()
        if results is None:
            # The read may have failed; don't change state.
            # Emit with zero velocity and tracking=False every tick.
            self.tracking = False
            self.gaze_updated.emit(self.gaze_x, self.gaze_y, 0.0, 0.0, False)
            return

        if results.multi_face_landmarks:
            feats = extract_features(results.multi_face_landmarks[0])

            if self._has_calib:
                px, py = self.predict(feats)
            else:
                # Fallback: use raw iris offsets as a normalised gaze
                # (clamp to [0, 1] then scale by screen dimensions)
                px = float(np.clip(feats[0] + 0.5, 0, 1)) * self.screen_w
                py = float(np.clip(feats[2] + 0.5, 0, 1)) * self.screen_h

            # Dead zone: ignore sub-pixel jitter (< 2 px)
            dx_raw = px - self.gaze_x
            dy_raw = py - self.gaze_y
            if abs(dx_raw) < 1.0 and abs(dy_raw) < 1.0:
                px = self.gaze_x
                py = self.gaze_y

            # EMA smoothing
            self.gaze_x = self.alpha * px + (1.0 - self.alpha) * self.gaze_x
            self.gaze_y = self.alpha * py + (1.0 - self.alpha) * self.gaze_y
            self.tracking = True
        else:
            self.tracking = False

        # Velocity (px / frame)
        vx = self.gaze_x - self.prev_gaze_x
        vy = self.gaze_y - self.prev_gaze_y
        self.prev_gaze_x = self.gaze_x
        self.prev_gaze_y = self.gaze_y

        self.gaze_updated.emit(self.gaze_x, self.gaze_y, vx, vy, self.tracking)
