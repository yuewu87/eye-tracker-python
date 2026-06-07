"""Kalman 滤波（2D 位置） + 一阶 IIR（1D 信号）。"""

import numpy as np


class KalmanFilter:
    """4 状态 Kalman 滤波器：位置 + 速度，恒速模型。"""

    def __init__(self, dt=1/25):
        self.dt = dt
        self.x = np.zeros(4)
        self.P = np.eye(4) * 500
        self.F = np.array([[1, 0, dt, 0],
                           [0, 1, 0, dt],
                           [0, 0, 1,  0],
                           [0, 0, 0,  1]])
        self.H = np.array([[1, 0, 0, 0],
                           [0, 1, 0, 0]])
        self.Q = np.diag([0.5, 0.5, 2.0, 2.0])
        self.R = np.eye(2) * 40
        self.initialized = False

    def update(self, z: np.ndarray):
        if not self.initialized:
            self.x[:2] = z
            self.initialized = True
            return z
        x_pred = self.F @ self.x
        P_pred = self.F @ self.P @ self.F.T + self.Q
        y_innov = z - self.H @ x_pred
        S = self.H @ P_pred @ self.H.T + self.R
        K = P_pred @ self.H.T @ np.linalg.inv(S)
        self.x = x_pred + K @ y_innov
        self.P = (np.eye(4) - K @ self.H) @ P_pred
        return self.x[:2].copy()

    def predict_only(self):
        """只预测不更新，用于测量被丢弃时。"""
        if not self.initialized:
            return self.x[:2].copy()
        self.x = self.F @ self.x
        self.P = self.F @ self.P @ self.F.T + self.Q
        return self.x[:2].copy()

    def set_smoothness(self, factor: float):
        noise = 5 + factor * 200
        self.R = np.eye(2) * noise

    def reset(self):
        self.initialized = False
        self.P = np.eye(4) * 500


class IIRFilter:
    """一阶 IIR：out = alpha * input + (1 - alpha) * prev。"""

    def __init__(self, alpha=0.7):
        self.alpha = alpha
        self._value = None

    def update(self, value: float) -> float:
        if self._value is None:
            self._value = value
        else:
            self._value = self.alpha * value + (1 - self.alpha) * self._value
        return self._value

    def reset(self):
        self._value = None
