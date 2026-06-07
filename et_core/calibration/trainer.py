"""校准模型训练、保存、加载、评估。"""

import os
import numpy as np
from sklearn.linear_model import RidgeCV
from sklearn.preprocessing import PolynomialFeatures


def train(samples: list, degree=2):
    """训练 Poly+RidgeCV 模型。

    samples: [(features(5,), target(2,)), ...]
    Returns: (model, x_mean, x_std, poly)
    """
    X = np.array([s[0] for s in samples], dtype=np.float64)
    y = np.array([s[1] for s in samples], dtype=np.float64)
    x_mean = X.mean(axis=0)
    x_std = X.std(axis=0) + 1e-6
    X_norm = (X - x_mean) / x_std

    poly = PolynomialFeatures(degree=degree, include_bias=False)
    X_poly = poly.fit_transform(X_norm)
    n_feat = X_poly.shape[1]

    model = RidgeCV(alphas=[0.01, 0.1, 0.5, 1.0, 5.0, 10.0, 50.0])
    model.fit(X_poly, y)
    print(f"[i] Poly(deg={degree}): 5→{n_feat} 维, best α={model.alpha_}")
    return model, x_mean, x_std, poly


def save(path: str, model, x_mean, x_std, screen_w: int, screen_h: int,
         poly: PolynomialFeatures):
    np.savez(path,
             x_mean=x_mean.astype(np.float64),
             x_std=x_std.astype(np.float64),
             screen_w=screen_w,
             screen_h=screen_h,
             model=model,
             poly_degree=2,
             poly_features_in=5)
    print(f"[OK] 校准参数已保存: {path}")
    print(f"     样本数: {len(x_mean)}, 屏幕: {screen_w}x{screen_h}")


def evaluate(samples: list, model, x_mean, x_std, poly: PolynomialFeatures,
             screen_w: int, screen_h: int, samples_per_point: int,
             point_labels: list[str] = None):
    """评估校准误差。"""
    X_all = np.array([s[0] for s in samples], dtype=np.float64)
    y_all = np.array([s[1] for s in samples], dtype=np.float64)
    Xn = (X_all - x_mean) / x_std
    Xn_poly = poly.transform(Xn)
    y_pred = model.predict(Xn_poly)
    errors = np.sqrt(((y_pred - y_all) ** 2).sum(axis=1))
    print(f"[i] 校准误差: 平均={errors.mean():.1f}px 最大={errors.max():.1f}px")

    if point_labels and samples_per_point > 0:
        for i, label in enumerate(point_labels):
            start = i * samples_per_point
            end = start + samples_per_point
            if start < len(errors):
                e = errors[start:end].mean()
                print(f"     {label}: 平均误差 {e:.0f}px")

    return errors.mean(), errors.max()
