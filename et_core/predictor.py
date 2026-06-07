"""多项式回归视线预测器。加载 calibration.npz，预测像素坐标。"""

import numpy as np
from sklearn.preprocessing import PolynomialFeatures


class GazePredictor:
    """Poly(deg=2) + RidgeCV 视线预测器。"""

    def __init__(self, screen_w: int, screen_h: int):
        self.screen_w = screen_w
        self.screen_h = screen_h
        self.model = None
        self.x_mean = None
        self.x_std = None
        self._poly = None
        self.scale_x = 1.0
        self.scale_y = 1.0
        self.bias_x = 0.0
        self.bias_y = 0.0
        self._loaded = False

    @property
    def is_loaded(self) -> bool:
        return self._loaded

    def load(self, path: str):
        calib = np.load(path, allow_pickle=True)
        self.x_mean = calib["x_mean"]
        self.x_std = calib["x_std"]
        n_feat = len(self.x_mean)
        if n_feat != 5:
            raise ValueError(f"校准特征维度 {n_feat} != 5，请重新校准")
        self.model = calib["model"].item()
        self.scale_x = self.screen_w / float(calib["screen_w"])
        self.scale_y = self.screen_h / float(calib["screen_h"])

        if "poly_degree" in calib:
            degree = int(calib["poly_degree"])
            n_in = int(calib["poly_features_in"])
            self._poly = PolynomialFeatures(degree=degree, include_bias=False)
            self._poly.fit(np.zeros((1, n_in)))
        else:
            self._poly = None

        self._loaded = True

    def predict(self, features: np.ndarray):
        x_norm = ((features - self.x_mean) / self.x_std).reshape(1, -1)
        if self._poly is not None:
            x_norm = self._poly.transform(x_norm)
        pred = self.model.predict(x_norm)[0]
        pred[0] = pred[0] * self.scale_x + self.bias_x
        pred[1] = pred[1] * self.scale_y + self.bias_y
        return (float(np.clip(pred[0], 0, self.screen_w)),
                float(np.clip(pred[1], 0, self.screen_h)))
