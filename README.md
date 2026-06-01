# Eye Tracker

> **建议：这个项目识别并不准确，顶多是个玩具。** 普通摄像头 + MediaPipe 虹膜检测的精度远不如专业眼动仪，只能标注大概在看哪个区域，不适合精确交互。仅供学习和娱乐。

---

桌面视线追踪光圈 — 用普通摄像头实现眼动仪效果。MediaPipe 虹膜检测 + GBR 机器学习模型 + Kalman 滤波，实时显示视线位置。

**适用场景：** 游戏直播、OBS 录制、演示标注

## 效果

- 空心白环 + 紫色外发光跟随视线移动
- 快速扫视时显示彗星拖尾
- 控制面板可隐藏 Overlay（玩家看不到但 OBS 能录制）
- 5 点校准 + 中心单点快速修正

## 环境

```bash
# 创建 conda 环境
conda env create -f eye_tracker/environment.yml
conda activate eye-tracker
```

依赖：Python 3.10, PySide6, OpenCV, MediaPipe, scikit-learn, numpy

## 使用

双击 `run.bat` 启动，或：

```bash
conda activate eye-tracker
python eye_tracker/main.py
```

首次运行自动进入 5 点校准（注视屏幕上依次出现的圆点）。

### 控制面板

| 按钮 | 功能 |
|------|------|
| 开始追踪 / 停止追踪 | 启动/关闭视线追踪 |
| 隐藏 | 切换 Overlay 光圈显示（自己看不看） |
| 校准 | 重新 5 点校准 |
| 中心校准 | 注视中心 2.5 秒快速修正漂移 |
| 隐藏面板 | 最小化到系统托盘 |

关闭窗口 → 隐藏到托盘，托盘右键 → 退出。

### OBS 设置

1. 添加「窗口捕获」→ 选择 `Eye Tracker - Capture`
2. 变换 → 拉伸至全屏
3. 滤镜 → 色度键 → 选黑色

## 技术

| 组件 | 说明 |
|------|------|
| 虹膜检测 | MediaPipe Face Mesh + 虹膜关键点 (468-477) |
| 特征提取 | 虹膜偏移 + 3D 头部姿态 (solvePnP)，眼距归一化 |
| 预测模型 | GradientBoostingRegressor（非线性回归） |
| 平滑 | 4 状态 Kalman 滤波（位置+速度） |
| 光圈渲染 | PySide6 透明全屏 Overlay + QMainWindow 捕获窗口 |

## 文件

```
eye_tracker/
├── main.py           # 入口 + 控制器
├── engine.py         # 追踪引擎（摄像头、特征、模型、Kalman）
├── calibrator.py     # 5 点校准 + 中心校准
├── widgets.py        # 界面窗口 + 光圈渲染
└── environment.yml   # conda 环境
```
