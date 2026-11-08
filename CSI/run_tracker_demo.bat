@echo off
setlocal EnableExtensions
title WiFi CSI 2D Tracker - DEMO MODE
cd /d "%~dp0"

echo ============================================
echo   WiFi CSI 2D Tracker - DEMO MODE
echo   (No hardware needed - simulated person)
echo ============================================
echo.

if not exist ".venv\Scripts\python.exe" (
    echo [ERROR] Virtual environment not found in .venv.
    pause
    exit /b 1
)

call ".venv\Scripts\activate.bat"
python src\dual_link_tracker.py --demo
pause
