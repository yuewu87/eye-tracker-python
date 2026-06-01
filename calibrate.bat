@echo off
call conda activate eye-tracker
python -c "from engine import GazeEngine; from calibrator import run_calibration; from PySide6.QtWidgets import QApplication; import sys; app=QApplication(sys.argv); s=app.primaryScreen().geometry(); e=GazeEngine(s.width(),s.height()); e.start_camera(); run_calibration(e); e.stop_camera()"
pause
