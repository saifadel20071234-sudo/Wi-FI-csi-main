@echo off
setlocal EnableExtensions
cd /d "%~dp0"

echo ============================================================
echo    WiFi CSI Sensing - DEMO MODE (no hardware needed)
echo ============================================================
echo.

where python >nul 2>nul
if errorlevel 1 (
    echo [ERROR] Python not found. Install Python 3.10-3.12 and add to PATH.
    pause
    exit /b 1
)

if not exist ".venv\Scripts\python.exe" (
    python -m venv .venv
)
call ".venv\Scripts\activate.bat"
python -m pip install --quiet --upgrade pip
python -m pip install --quiet -r requirements.txt

echo.
echo Launching simulated sensing (a virtual person walks left/right) ...
echo Press 'q' in the window to quit.
echo.
python src\main.py --demo

pause
