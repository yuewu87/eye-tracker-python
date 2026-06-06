# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## 运行

```bash
conda activate eye-tracker
python eye_tracker/main.py
```

或双击 `run.bat`。项目无测试、无 lint。

## 架构

四文件单进程桌面应用，PySide6 + MediaPipe + OpenCV。

### 信号流

```
摄像头 → engine.read_camera()
  → 人脸裁剪 + 缩放到眼距≈200px → MediaPipe FaceMesh 提取 478 点
  → extract_features() 计算 5 维特征 → Poly(deg=2)+RidgeCV 预测
  → Kalman 滤波 → gaze_updated 信号
  → Overlay.update_state() + Capture.update_state() + MainWindow.update_status()
```

### 模块职责

| 文件 | 职责 |
|------|------|
| `main.py` | App 控制器：创建引擎和窗口，管理追踪/校准状态切换，系统托盘 |
| `engine.py` | GazeEngine(QObject)：摄像头/IR 源、人脸裁剪归一化、特征提取、Poly+Ridge 预测、Kalman 滤波 |
| `calibrator.py` | 两个全屏校准窗口：`CalibrationWindow`(7点) 和 `CenterCalibWindow`(单点)。均用 QEventLoop 阻塞执行 |
| `widgets.py` | 四个窗口类 + `draw_glow()` 光圈渲染 |
| `ir_source.py` | IR 摄像头 TCP 源，连接 C# 桥接程序 |

### 特征提取 (`engine.py:extract_features`)

返回 5 维 float32：`[右虹膜 dx/ed, dy/ed, 左虹膜 dx/ed, dy/ed, 对数眼距]`

- 虹膜偏移 = 虹膜中心 - 眼睛中心，**除以眼距归一化**（头部前后移动不变）
- 环境变量 `GLOG_minloglevel`/`TF_CPP_MIN_LOG_LEVEL` 必须在 import mediapipe 前设置

### 人脸裁剪归一化 (`engine.py:read_camera`)

首帧全图扫脸取眼角坐标，后续帧用上帧眼角裁剪当前帧、缩放到眼距≈200px。Landmarks 保持在裁剪空间中，特征空间恒定不管头距。

### 双窗口设计

- **OverlayWindow**：透明全屏置顶点击穿透。`_toggle_overlay()` 用 `setVisible()` 控制
- **CaptureWindow**：全屏黑底 QMainWindow，OBS 窗口捕获 + 色度键抠黑。**独立于 Overlay**
- 启动追踪时 `overlay.hide(); overlay.show()` 强制重渲染

### 校准

- `calibration.npz` 不存在时自动弹出 7 点校准（RGB）/ `calibration_ir.npz`（IR）
- `run_calibration(engine)` 用 QEventLoop 阻塞主循环直到 `calibration_done` 信号
- 两阶段：prep（倒计时1s, 十字准星呼吸动画）→ collect（采120帧, 前0.5s静默过渡）
- 校准后自动剔除 z-score > 2.5 的离群帧
- 中心校准：注视屏幕中央十字 2.5 秒，计算偏移量存入 `engine.bias_x/y`
- 旧校准文件特征维度不匹配时自动拒绝

### 依赖注意事项

- PySide6 走 conda-forge（不可 pip），否则 qwindows.dll 缺少系统 DLL
- opencv-python-headless 走 pip（无 Qt 依赖，避免和 PySide6 的 Qt DLL 冲突）

### IR 摄像头 (Windows Hello)

- C# 桥接程序 `ir_bridge/` 通过 Windows.Media.Capture API 访问 IR 传感器
- 编译：`cd ir_bridge && dotnet build -c Release`
- `--ir` 参数启动 IR 模式，RGB/IR 可切换，各自独立校准文件

## 常见陷阱

- QWidget 顶层窗口必须存为 Python 变量，否则被 GC 回收导致窗口闪现后消失
- `QApplication.quit()` 会退出整个程序；校准窗口用 `QEventLoop.quit()` 只退出嵌套循环
- MediaPipe 的 `refine_landmarks=True` 才输出虹膜关键点 468-477
