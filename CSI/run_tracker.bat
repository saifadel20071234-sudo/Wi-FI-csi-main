@echo off
setlocal EnableExtensions
title WiFi CSI 2D Heatmap Tracker
cd /d "%~dp0"

echo ============================================================
echo   WiFi CSI 2D Heatmap Tracker (Dual-Link / Single-Link)
echo ============================================================
echo.

if not exist ".venv\Scripts\python.exe" (
    echo [ERROR] Virtual environment not found in .venv.
    pause
    exit /b 1
)

call ".venv\Scripts\activate.bat"

echo Launching 2D Heatmap Tracker...
echo - To calibrate: press 'c' while keeping room empty for 3s.
echo - To clear heatmap: press 'z'.
echo - To adjust threshold: press '+' or '-'.
echo - To quit: press 'q' or Esc.
echo.

python src\dual_link_tracker.py %*

if errorlevel 1 (
    echo.
    echo ------------------------------------------------------------
    echo Notice: Serial connection failed or no ESP32 detected.
    echo Would you like to run in 2D DEMO simulation mode right now?
    echo ------------------------------------------------------------
    set /p choice="Run 2D Demo? (Y/N): "
    if /i "%choice%"=="Y" (
        python src\dual_link_tracker.py --demo
    )
)

pause
