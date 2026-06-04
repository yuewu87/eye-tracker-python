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
  → MediaPipe FaceMesh 提取 478 点 → extract_features() 计算 10 维特征
  → GBR 模型预测屏幕坐标 → Kalman 滤波 → gaze_updated 信号
  → Overlay.update_state() + Capture.update_state() + MainWindow.update_status()
```

### 模块职责

| 文件 | 职责 |
|------|------|
| `main.py` | App 控制器：创建引擎和窗口，管理追踪/校准状态切换，系统托盘 |
| `engine.py` | GazeEngine(QObject)：摄像头、MediaPipe、特征提取、GBR 预测、Kalman 滤波。通过 `gaze_updated(x,y,vx,vy,tracking)` 信号输出 |
| `calibrator.py` | 两个全屏校准窗口：`CalibrationWindow`(5点) 和 `CenterCalibWindow`(单点)。均用 QEventLoop 阻塞执行 |
| `widgets.py` | 四个窗口类 + `draw_glow()` 光圈渲染：OverlayWindow(透明全屏)、CaptureWindow(黑底OBS用)、MainWindow(控制面板) |

### 特征提取 (`engine.py:extract_features`)

返回 10 维 float32：`[右虹膜 dx, dy, 左虹膜 dx, dy, 鼻尖 dx, 鼻尖 dy, 对数眼距, yaw, pitch, roll]`

- 虹膜偏移 = 虹膜中心 - 眼睛中心，**除以眼距归一化**（头部前后移动不变性）
- `_compute_head_pose()` 用 solvePnP 从 6 个 3D 面部关键点计算头部旋转角
- 环境变量 `GLOG_minloglevel`/`TF_CPP_MIN_LOG_LEVEL` 必须在 import mediapipe 前设置

### 双窗口设计

- **OverlayWindow**：透明全屏置顶点击穿透，用户直接看到。`_toggle_overlay()` 用 `setVisible()` 控制
- **CaptureWindow**：全屏黑底 QMainWindow，OBS 窗口捕获 + 色度键抠黑。**独立于 Overlay，关闭 Overlay 不影响 OBS 录制**
- 启动追踪时 `overlay.hide(); overlay.show()` 强制重渲染，否则首次不显示

### 校准

- `calibration.npz` 不存在时自动弹出 5 点校准
- `run_calibration(engine)` 用 QEventLoop 阻塞主循环直到 `calibration_done` 信号
- 校准期间主窗口 hide，引擎 pause（停 tick、保留摄像头），校准完 resume
- 中心校准只注视中央 2.5 秒，计算偏移量存入 `engine.bias_x/y`
- 旧校准文件与新特征维度不匹配时自动拒绝并提示重做

### 依赖注意事项

- PySide6 走 conda-forge（不可 pip），否则 qwindows.dll 缺少系统 DLL
- opencv-python-headless 走 pip（无 Qt 依赖，避免和 PySide6 的 Qt DLL 冲突）

## 常见陷阱

- QWidget 顶层窗口必须存为 Python 变量，否则被 GC 回收导致窗口闪现后消失
- `QApplication.quit()` 会退出整个程序；校准窗口用 `QEventLoop.quit()` 只退出嵌套循环
- MediaPipe 的 `refine_landmarks=True` 才输出虹膜关键点 468-477
