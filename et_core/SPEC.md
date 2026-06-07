# et_core — 视线追踪核心库 设计规格

## 1. 定位

纯 Python 库，零 GUI 依赖，`import et_core` 即可用。校准 UI 作为可选子模块，不 import 不触发 PySide6 依赖。

```
et_core/
├── __init__.py          # EyeTracker 顶层 API
├── types.py             # GazeResult, MonitorInfo NamedTuple/dataclass
├── engine.py            # 摄像头、MediaPipe、人脸裁剪、特征提取
├── predictor.py         # Poly+RidgeCV 预测、校准参数加载
├── filter.py            # Kalman 滤波器（位置）+ IIR（显示器分类）
├── monitor_detect.py    # 多显示器校准 + 实时判断
├── calibration/
│   ├── __init__.py
│   ├── collector.py     # 校准数据采集（回调驱动，零 GUI）
│   ├── trainer.py       # 模型训练 + calibration.npz 读写
│   └── ui.py            # 可选：PySide6 全屏校准窗口
└── SPEC.md              # 本文档
```

## 2. 顶层 API

```python
from et_core import EyeTracker
from et_core.types import GazeResult

tracker = EyeTracker(
    screen_w=None,      # 主屏宽度，None=自动检测
    screen_h=None,      # 主屏高度
    monitors=None,      # 多屏几何列表 [(x,y,w,h), ...]，None=自动枚举
    camera_id=0,        # 摄像头索引
    calib_path=None,    # 校准文件路径，None=默认 ./calibration.npz
)

# 生命周期
tracker.start()         # 开摄像头 + 加载校准 + 启动 25fps 循环
result = tracker.update()  # 阻塞一帧，返回最新 GazeResult
tracker.stop()          # 停止循环 + 释放摄像头

# 属性
tracker.tracking        # bool，当前是否检测到人脸
tracker.gaze_x, tracker.gaze_y  # 主屏像素坐标
tracker.monitor_index   # 当前注视的显示器索引（0-based, None=未分类）
```

### GazeResult

```python
@dataclass
class GazeResult:
    x: float              # 主屏像素 X
    y: float              # 主屏像素 Y
    vx: float             # 帧间速度 X（px/frame）
    vy: float             # 帧间速度 Y
    tracking: bool        # 是否检测到人脸
    monitor_index: int | None   # 当前屏幕索引
    timestamp: float      # time.perf_counter()
```

## 3. 模块职责

### 3.1 engine.py

直接复用现有 `engine.py` 的核心逻辑，去掉 PySide6 依赖：

- `GazeEngine(QObject)` → `CameraProcessor`（纯类，无 QTimer）
- `read_camera()` — 摄像头读取 + 人脸裁剪归一化，返回 `(frame, mp_results)`
- `extract_features()` — 5 维特征，逻辑不变
- 单帧方法 `process_frame()` — 调用 read_camera + extract_features，返回特征向量或 None
- QTimer 替换为外部驱动的 `update()` 调用，帧率由调用方控制

### 3.2 predictor.py

新模块，从现有 `engine.py` 的 predict/load_calibration 抽离：

```python
class GazePredictor:
    def load(self, path)           # 加载 calibration.npz
    def predict(self, features)    # → (x, y)
    @property
    def is_loaded(self) -> bool
```

不支持 deg≠2（现有教训：deg=3 过拟合）。

### 3.3 filter.py

从现有 KalmanFilter 抽离，新增 IIR 平滑：

```python
class KalmanFilter:
    # 同现有实现，去除马氏门控（已验证无效）
    def update(self, z) -> (x, y)
    def predict_only(self) -> (x, y)
    def reset()

class IIRFilter:
    """一阶 IIR：out = alpha*input + (1-alpha)*prev"""
    def __init__(self, alpha=0.7)
    def update(self, value) -> float
```

IIRFilter 用于显示器分类的虹膜偏移平滑——比 Kalman 轻量，1 维信号不需要 4 状态。

### 3.4 monitor_detect.py

```python
class MonitorDetector:
    def calibrate(self, offsets_per_monitor: list[float])  # 存储每屏虹膜水平偏移
    def classify(self, iris_h_offset: float) -> int | None  # 返回最近屏索引
    def is_calibrated(self) -> bool
    def save(self, path) / load(self, path)
```

**信号**：`iris_h_offset = (right_dx + left_dx) / 2`，即 `(feats[0] + feats[2]) / 2`。

**校准流程**：
1. 用户依次注视每块屏幕中心
2. 每屏采 60 帧，取 `iris_h_offset` 均值作为该屏参考值
3. 存为 `monitor_calib.npz`（独立于主校准文件）

**运行时**：
1. 每帧 `iris_h_offset` 经 IIRFilter 平滑（alpha=0.7）
2. 最近邻匹配到各屏幕参考值
3. 加入迟滞（hysteresis）：切换屏幕需连续 8 帧确认（~320ms），防止边界抖动

### 3.5 calibration/

#### collector.py（零 GUI）
```python
class CalibrationCollector:
    def start_point(self, screen_x, screen_y)   # 开始采集一个注视点
    def feed_frame(self, features)               # 喂入特征，自动计时+采样
    @property
    def is_done(self) -> bool                    # 该点采集完成
    def get_samples(self) -> list                # 返回该点所有样本
```

回调驱动，调用方自己决定何时开始、如何渲染 UI。

#### trainer.py
```python
def train(samples, degree=2) -> (model, x_mean, x_std, poly)
def save(path, model, x_mean, x_std, screen_w, screen_h, poly_degree, poly_features_in)
def load(path) -> (model, x_mean, x_std, ...)
def evaluate(samples, model, ...) -> (mean_error, max_error, per_point_errors)
```

#### ui.py（可选，依赖 PySide6）
```python
class CalibrationUI:
    """封装 collector + trainer 的全屏 7 点校准窗口"""
    def run_blocking(engine)   # 阻塞执行校准，返回成功/失败
    # 内部复用现有 calibrator.py 的 UI 逻辑
class CenterCalibrationUI:
    """单点中心校准"""
class MonitorCalibrationUI:
    """多显示器校准：每屏中心 1 点"""
```

## 4. 信号流

```
CameraProcessor.process_frame()  →  features(5,) 或 None
  ├─→ GazePredictor.predict(features)  →  (px, py)
  │     └─→ KalmanFilter.update(px, py)  →  (gaze_x, gaze_y)
  │
  └─→ iris_h_offset = (feats[0] + feats[2]) / 2
        └─→ IIRFilter.update(iris_h_offset)
              └─→ MonitorDetector.classify(offset)  →  monitor_index
```

## 5. 设计决策

| 决策 | 理由 |
|------|------|
| deg=2 固定 | deg=3 在实测中过拟合，误差 69.8→81.5px |
| 无马氏门控 | R=40 与实测噪声 σ≈70px 失配，门控全拦或全放 |
| IIR 替代 Kalman 做显示器分类 | 1D 信号不需要 4 状态 Kalman，IIR 延迟更低 |
| 显示器分类用迟滞 | 防止屏幕边界来回跳变 |
| 两套独立校准文件 | `calibration.npz`（坐标）+ `monitor_calib.npz`（屏分类）互不干扰 |
| 摄像头帧率由外部控制 | 调用方决定 `update()` 频率，库不做定时器 |
| Qt 依赖可选 | 只有 `calibration.ui` import PySide6 |

## 6. 依赖

```
核心（必需）：
  numpy, opencv-python-headless, mediapipe, scikit-learn

校准 UI（可选）：
  PySide6 (conda-forge)
```
