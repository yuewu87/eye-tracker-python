@echo off
call conda activate eye-tracker
python eye_tracker/main.py --ir
pause
