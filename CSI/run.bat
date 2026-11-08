@echo off
setlocal EnableExtensions
cd /d "%~dp0"

echo ============================================================
echo    WiFi CSI Sensing - Human Detection Launcher
echo ============================================================
echo.

rem --- 1) Check Python -------------------------------------------------
where python >nul 2>nul
if errorlevel 1 (
    echo [ERROR] Python is not installed or not in PATH.
    echo         Install Python 3.10-3.12 from https://www.python.org/downloads/
    echo         and tick "Add python.exe to PATH" during install.
    echo.
    pause
    exit /b 1
)

rem --- 2) Create venv on first run -------------------------------------
if not exist ".venv\Scripts\python.exe" (
    echo [1/3] Creating virtual environment (.venv) ...
    python -m venv .venv
    if errorlevel 1 (
        echo [ERROR] Failed to create virtual environment.
        pause
        exit /b 1
    )
)

call ".venv\Scripts\activate.bat"

rem --- 3) Install dependencies -----------------------------------------
echo [2/3] Installing dependencies (first run downloads packages) ...
python -m pip install --quiet --upgrade pip
python -m pip install --quiet -r requirements.txt
if errorlevel 1 (
    echo.
    echo [WARNING] Some packages failed to install.
    echo           If the error mentions "opencv-python" and you are on a very
    echo           new Python version, install Python 3.10-3.12 and re-run.
    echo.
)

rem --- 4) Launch -------------------------------------------------------
echo [3/3] Starting application ...
echo.
echo   Tips:
echo     - Press 'q' in the window to quit.
echo     - To try WITHOUT hardware first, close and run: run_demo.bat
echo.
python src\main.py %*

echo.
echo Application closed.
pause
