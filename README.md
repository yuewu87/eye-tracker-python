# Eye Tracker

> **普通摄像头 + MediaPipe 虹膜检测，精度有限。** 仅供学习和娱乐，不适合精确交互。

---

桌面视线追踪 — MediaPipe 虹膜检测 + 多项式回归 + Kalman 滤波。

项目包含两个独立程序，**互不依赖**：

- **`eye_tracker/`** — 核心桌面应用。PySide6 GUI，OBS 捕获就绪，开箱即用
- **`et_core/`** — 可嵌入模块。零 GUI，`import et_core` 即可集成到其他 Python 项目

```
EYE/
├── example.py             # et_core 使用示例
├── eye_tracker/           # [核心] 独立桌面应用
│   ├── main.py            #   入口 + 控制器 + 系统托盘
│   ├── engine.py          #   摄像头、特征提取、预测、Kalman
│   ├── calibrator.py      #   7 点校准 + 中心校准
│   ├── widgets.py         #   Overlay/Capture 窗口 + 光圈渲染
│   └── ir_source.py       #   IR 摄像头 TCP 源
├── et_core/               # [可嵌入] 核心库，独立于 eye_tracker
│   ├── __init__.py        #   EyeTracker 顶层 API
│   ├── types.py           #   GazeResult 数据类
│   ├── engine.py          #   CameraProcessor + extract_features
│   ├── predictor.py       #   GazePredictor（Poly+RidgeCV）
│   ├── filter.py          #   KalmanFilter + IIRFilter
│   ├── monitor_detect.py  #   虹膜偏移多屏分类器
│   └── calibration/       #   校准子模块（采集器 + 训练器 + 可选 UI）
└── ir_bridge/             # C# IR 摄像头桥接
```

---

## eye_tracker — 核心桌面应用

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

### 实现方法

| 组件 | 方法 |
|------|------|
| **人脸检测** | MediaPipe FaceMesh，`refine_landmarks=True` 输出 478 点，虹膜点 468-477 |
| **人脸裁剪归一化** | 首帧全图扫脸取眼角 → 后续帧用上帧眼角裁剪当前帧 → 缩放到眼距≈200px。Landmarks 保持在裁剪空间，头距变化不影响特征 |
| **特征提取** | `extract_features()` 返回 5 维 float32：`[右虹膜 dx/眼距, dy/眼距, 左虹膜 dx/眼距, dy/眼距, log(眼距)]` |
| **视线预测** | `PolynomialFeatures(deg=2) + RidgeCV`，7 点 × 120 帧校准，z-score > 2.5 剔除离群帧。**deg=3 实测过拟合** |
| **坐标平滑** | 4 状态 Kalman 滤波（位置+速度恒速模型），Q=diag[0.5,0.5,2,2], R=eye(2)*40。**注意：马氏距离门控无效** |
| **双窗口设计** | OverlayWindow：透明全屏置顶，用户可见的光圈。CaptureWindow：全屏黑底，OBS 窗口捕获 + 色度键抠黑，独立于 Overlay |
| **校准流程** | 两阶段：prep（倒计时 1s, 十字准星呼吸动画）→ collect（采 120 帧, 前 0.5s 静默过渡）。QEventLoop 阻塞执行 |
| **中心校准** | 注视屏幕中央十字 2.5 秒，计算偏移量修正漂移 |
| **IR 模式** | C# 桥接 `ir_bridge/` 通过 Windows.Media.Capture API 访问 Windows Hello IR 传感器。RGB/IR 可切换，各自独立校准文件 |

### OBS 设置

1. 添加「窗口捕获」→ 选择 `Eye Tracker - Capture`
2. 变换 → 拉伸至全屏
3. 滤镜 → 色度键 → 选黑色

### 常见问题

- PySide6 必须走 conda-forge，不可 pip（qwindows.dll 缺少系统 DLL）
- opencv-python-headless 走 pip（无 Qt 依赖，避免和 PySide6 冲突）
- QWidget 顶层窗口必须存为 Python 变量，否则 GC 回收导致窗口闪现后消失
- `QApplication.quit()` 退出整个程序；校准窗口用 `QEventLoop.quit()` 只退出嵌套循环
- 环境变量 `GLOG_minloglevel` / `TF_CPP_MIN_LOG_LEVEL` 必须在 import mediapipe 前设置

---

## et_core — 可嵌入模块

与 `eye_tracker/` 完全独立，互不依赖。零 GUI 核心，`import et_core` 即可集成到 LLM 桌宠、屏幕捕获切换等场景。

### 快速开始

```python
from et_core import EyeTracker

tracker = EyeTracker()
tracker.start()

while True:
    r = tracker.update()
    print(r.x, r.y, r.monitor_index)  # 视线坐标 + 当前屏幕

tracker.stop()
```

示例脚本：`python example.py`（带自动多屏校准）。

### 与 eye_tracker 的区别

| | eye_tracker | et_core |
|--|------------|---------|
| 定位 | 桌面应用 | 可嵌入库 |
| GUI | PySide6（必需） | 零 GUI，UI 可选 |
| 调用方式 | 独立进程 | `import` |
| 摄像头 | QTimer 驱动 | 外部调用 `update()` |
| 多屏检测 | 不支持 | 内置 |
| 校准文件 | `eye_tracker/calibration.npz` | `et_core/calibration.npz`（互不干扰） |

### 实现方法

| 组件 | 方法 |
|------|------|
| **人脸检测** | 同上，复用 MediaPipe FaceMesh 管线 |
| **人脸裁剪归一化** | 同上，从 `GazeEngine` 移植为 `CameraProcessor`，去 QTimer |
| **特征提取** | 同上 `extract_features()` |
| **视线预测** | `GazePredictor`：加载 `calibration.npz` → Poly+RidgeCV 预测屏幕坐标 |
| **坐标平滑** | `KalmanFilter`：同上。`IIRFilter`：一阶 IIR 用于显示器分类的虹膜偏移平滑 |
| **多屏检测** | 虹膜水平偏移 `(右dx + 左dx) / 2` → IIR(α=0.7) → 最近邻匹配每屏校准值 → 4 帧迟滞防抖。转头是信号不是噪声。 |
| **摄像头兜底** | 指定摄像头失败则自动尝试 0→1→2→3 |
| **校准** | `calibration.npz`（7 点像素坐标）+ `monitor_calib.npz`（每屏虹膜偏移）。采集器回调驱动，UI 可选 PySide6，两套文件独立不干扰 |

### 接口

```python
# 创建（自动检测屏幕、多显示器几何）
tracker = EyeTracker(
    screen_w=None,           # None=自动
    screen_h=None,
    monitors=None,           # None=自动枚举
    camera_id=0,
    calib_path=None,         # None=默认 et_core/calibration.npz
    monitor_calib_path=None, # None=默认 et_core/monitor_calib.npz
)

# 生命周期
tracker.start()
result = tracker.update()   # GazeResult(x, y, vx, vy, tracking, monitor_index)
tracker.stop()

# 属性
tracker.tracking            # bool
tracker.gaze_x, gaze_y      # 主屏坐标
tracker.monitor_index       # 当前屏幕索引
tracker.monitors            # 多屏几何列表

# 校准（可选 PySide6）
tracker.run_calibration()           # 7 点
tracker.run_center_calibration()    # 中心单点
tracker.run_monitor_calibration()   # 多屏
```
