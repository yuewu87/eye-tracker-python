# Eye Tracker V2 — Design Spec

## Goal
Merge calibrator + tracker into a single PySide6 desktop app with a dark-themed
main panel, improved fluid aperture rendering, and integrated calibration flow.

## Architecture

Single entry point `main.py`, three supporting modules:

```
eye_tracker/
├── main.py           # Entry point + MainWindow
├── engine.py         # Gaze engine (camera, MediaPipe, prediction, smoothing)
├── calibrator.py     # Fullscreen calibration (refactored to return Engine)
├── widgets.py        # All QWidget subclasses + draw_glow renderer
├── environment.yml
└── calibration.npz   # Generated per-user
```

### Module Responsibilities

**engine.py — GazeEngine (QObject)**
- Owns cv2.VideoCapture + MediaPipe FaceMesh
- Loads calibration.npz, predicts screen coordinates
- Applies dead-zone + EMA smoothing
- Emits `gaze_updated(x, y, vx, vy, tracking)` signal 30x/sec
- One instance shared by calibration and tracking

**calibrator.py — run_calibration(engine)**
- Fullscreen 5-point calibration using the passed engine
- Returns updated calibration parameters
- Engine is stopped/resumed around calibration

**widgets.py — All rendering**
- `MainWindow` — dark panel, status LED, coordinate display, buttons
- `OverlayWindow` — transparent fullscreen click-through aperture
- `CaptureWindow` — black fullscreen OBS capture window
- `draw_glow(painter, x, y, vx, vy, pulse)` — fluid-deforming ring aperture

**main.py — App entry**
- Creates QApplication, Engine, MainWindow
- On "Start Tracking": shows Overlay + Capture, starts engine
- On "Calibrate": pauses tracking, runs calibration, resumes
- On "Hide": toggles overlay visibility

## Main Window Layout

```
┌──────────────────────────┐
│  ● Eye Tracker    [_][X] │  ● = green (tracking) / gray (idle)
│                          │
│  视线: (847, 362)         │
│  平滑: ───●───  0.12     │
│                          │
│  ┌────────────────────┐  │
│  │     开始追踪        │  │  霓虹绿 #0f0
│  └────────────────────┘  │
│  ┌──────────┐┌────────┐ │
│  │   隐藏    ││   校准  │ │  辅助按钮
│  └──────────┘└────────┘ │
└──────────────────────────┘
```

## Aperture Fluid Deformation

Base shape: hollow ring, r=42, drawn with QPen (NoBrush fill).
Fluid effect: the ring deforms into an ellipse along the velocity vector.

```
stretch = clamp(1.0 + |velocity| * 0.08, 1.0, 1.8)
rx = r * stretch          # along movement direction
ry = r / stretch          # perpendicular
rotation = atan2(vy, vx)  # align to movement
```

Additional layers:
- Outer glow: radial gradient, static, r+16
- Inner highlight: thin ring at r-3
- 3 bright droplets orbiting the ring (keep from current version)

Color: cyan-blue conical gradient that rotates with pulse.

## Smoothing Control

Slider in main window: range 0.02–0.50, default 0.12.
Lower = smoother but laggier. Stored as `alpha` in the EMA formula.

## Calibration Integration

1. User clicks "校准" → main.py pauses engine timer
2. CalibrationWindow shows fullscreen (uses engine's camera pass-through)
3. 5-point Ridge regression (same as current calibrator.py)
4. On complete: saves calibration.npz, reloads engine, resumes tracking if active

## State Machine

```
[Start] → MainWindow shown
  ├─ calibration.npz exists? → ready
  └─ no calibration.npz → auto-starts calibration → ready

[Ready]
  ├─ "开始追踪" → overlay + capture shown, engine starts → Tracking
  └─ "校准" → calibration flow → back to Ready

[Tracking]
  ├─ "隐藏" → overlay hidden, engine keeps running
  ├─ "校准" → pause engine, calibration flow, resume → Tracking
  └─ close window → shutdown
```

## Error Handling
- Camera unavailable: show error in status label, disable start button
- Face not detected: "未检测到人脸" status, keep last position
- Calibration fails (< 30 samples): show warning, don't save, return to main

## Non-Goals
- Hotkey registration (removed; control panel buttons suffice)
- Multi-monitor support
- Profile/settings persistence beyond calibration.npz
