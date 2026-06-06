# Eye Tracker

> **建议：这个项目识别并不准确，顶多是个玩具。** 普通摄像头 + MediaPipe 虹膜检测的精度远不如专业眼动仪，只能标注大概在看哪个区域，不适合精确交互。仅供学习和娱乐。

---

桌面视线追踪光圈 — 用普通摄像头实现眼动仪效果。MediaPipe 虹膜检测 + 多项式回归 + Kalman 滤波。

**适用场景：** 游戏直播、OBS 录制、演示标注

## 效果

- 白色空心圆环 + 紫色外发光跟随视线
- 快速移动时彗星拖尾（速度 > 15px/frame）
- 控制面板可切换 Overlay 显隐（自己看不看，OBS 独立录制）
- 7 点校准 + 中心单点漂移修正
- RGB / IR 双模式切换（IR 需 Windows Hello 摄像头，实验性）
- 系统托盘后台运行

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
| 特征提取 | 5 维：虹膜偏移 + 眼距（最简映射） |
| 人脸归一化 | 上帧眼角裁剪当前帧 → 缩放到固定眼距，头部移动不变 |
| 预测模型 | PolynomialFeatures(deg=2) + RidgeCV（平滑连续曲面） |
| 平滑 | 4 状态 Kalman 滤波（位置+速度） |
| 光圈渲染 | PySide6 透明全屏 Overlay + QMainWindow 黑底捕获窗口 |

## 文件

```
eye_tracker/
├── main.py           # 入口 + 控制器
├── engine.py         # 追踪引擎（摄像头、特征、模型、Kalman）
├── calibrator.py     # 7 点校准 + 中心校准
├── widgets.py        # 界面窗口 + 光圈渲染
├── ir_source.py      # IR 摄像头 TCP 源
└── environment.yml   # conda 环境
ir_bridge/
├── Program.cs        # C# IR 桥接（Windows.Media.Capture API）
└── ir_bridge.csproj
```
