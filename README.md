# Eye Tracker

> **普通摄像头 + MediaPipe 虹膜检测，精度有限。** 仅供学习和娱乐，不适合精确交互。

---

桌面视线追踪 — MediaPipe 虹膜检测 + 多项式回归 + Kalman 滤波。两套系统：`eye_tracker/` 是带 GUI 的独立应用，`et_core/` 是可嵌入其他项目的核心库。

## 目录

```
EYE/
├── example.py             # et_core 使用示例（多屏检测）
├── eye_tracker/           # 独立桌面应用（PySide6 GUI）
│   ├── main.py            #   入口 + 控制器 + 系统托盘
│   ├── engine.py          #   摄像头、特征提取、预测、Kalman
│   ├── calibrator.py      #   7 点校准 + 中心校准
│   ├── widgets.py         #   Overlay/Capture 窗口 + 光圈渲染
│   └── ir_source.py       #   IR 摄像头 TCP 源
├── et_core/               # 可嵌入核心库（零 GUI、import 即用）
│   ├── __init__.py        #   EyeTracker 顶层 API
│   ├── types.py           #   GazeResult 数据类
│   ├── engine.py          #   CameraProcessor + extract_features
│   ├── predictor.py       #   GazePredictor（Poly+RidgeCV）
│   ├── filter.py          #   KalmanFilter + IIRFilter
│   ├── monitor_detect.py  #   虹膜偏移多屏分类器（迟滞+最近邻）
│   ├── calibration/       #   校准子模块
│   │   ├── collector.py   #     回调驱动数据采集（零 GUI）
│   │   ├── trainer.py     #     模型训练/保存/评估
│   │   └── ui.py          #     可选 PySide6 校准窗口
│   ├── SPEC.md            #   设计规格
│   └── PLAN.md            #   实现计划
└── ir_bridge/             # C# IR 摄像头桥接
```

---

## et_core — 核心库

### 安装

依赖已在 `eye_tracker/environment.yml` 中，无需额外安装。

### 快速开始

```python
from et_core import EyeTracker

tracker = EyeTracker()
tracker.start()

while True:
    r = tracker.update()
    print(r.x, r.y, r.monitor_index)  # 坐标 + 当前屏幕

tracker.stop()
```

### 实现方法

| 组件 | 方法 |
|------|------|
| **人脸检测** | MediaPipe FaceMesh，`refine_landmarks=True` 输出虹膜关键点 468-477 |
| **人脸裁剪归一化** | 首帧全图扫脸取眼角 → 后续帧用上帧眼角裁剪当前帧 → 缩放到眼距≈200px。Landmarks 保持在裁剪空间，头部移动不变 |
| **特征提取** | `extract_features()` 返回 5 维：`[右虹膜 dx/眼距, dy/眼距, 左 dx/眼距, dy/眼距, log(眼距)]` |
| **视线预测** | `PolynomialFeatures(degree=2) + RidgeCV`（交叉验证选 α），校准采 7 点 × 120 帧，z-score > 2.5 剔除离群。**注意：deg=3 经实测过拟合，误差反而增大** |
| **坐标平滑** | 4 状态 Kalman 滤波（位置+速度恒速模型），Q=diag[0.5,0.5,2,2], R=eye(2)*40。**注意：马氏距离门控无效，R 与实测噪声失配** |
| **多屏检测** | 虹膜水平偏移 `(右dx + 左dx) / 2` 经 IIR(α=0.7) 平滑 → 最近邻匹配每屏校准值 → 4 帧迟滞防抖。转头是信号不是噪声 |
| **校准** | 两套独立：`calibration.npz`（7 点像素坐标）+ `monitor_calib.npz`（每屏虹膜偏移）。采集器回调驱动，UI 可选 PySide6 |

### 多屏检测校准

`example.py` 会自动检测并运行。每块屏幕依次显示全屏准星（倒计时 → 采集），程序窗口中查看结果。**必须确保显示器校准文件 `monitor_calib.npz`**。

### 摄像头兜底

若指定摄像头失败，自动尝试索引 0→1→2→3。

---

## eye_tracker — 桌面应用

### 环境

```bash
conda env create -f eye_tracker/environment.yml
conda activate eye-tracker
```

依赖：Python 3.10, PySide6 (conda-forge), OpenCV, MediaPipe, scikit-learn, numpy

### 使用

```bash
python eye_tracker/main.py
```

首次运行自动进入 7 点校准。

| 按钮 | 功能 |
|------|------|
| 开始/停止追踪 | 启动/关闭视线追踪 |
| 隐藏 | 切换 Overlay 光圈显隐 |
| 校准 | 重新 7 点校准 |
| 中心校准 | 注视中心 2.5 秒修正漂移 |
| 隐藏面板 | 最小化到系统托盘 |

关闭窗口 = 隐藏到托盘。

### OBS 设置

1. 添加「窗口捕获」→ 选择 `Eye Tracker - Capture`
2. 变换 → 拉伸至全屏
3. 滤镜 → 色度键 → 选黑色

### 常见问题

- PySide6 必须走 conda-forge，不可 pip（qwindows.dll 缺少系统 DLL）
- opencv-python-headless 走 pip（无 Qt 依赖，避免和 PySide6 冲突）
- QWidget 顶层窗口存为 Python 变量，否则 GC 回收导致闪现消失
- `QApplication.quit()` 退出整个程序；校准窗口用 `QEventLoop.quit()` 只退出嵌套循环
- `calibration.npz` 特征维度不匹配时自动拒绝
