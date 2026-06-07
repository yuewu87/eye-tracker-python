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
