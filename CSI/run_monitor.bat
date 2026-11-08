@echo off
setlocal EnableExtensions
cd /d "%~dp0"

echo ============================================================
echo   WiFi CSI Real-Time Scope & Presence Monitor (Single ESP32)
echo ============================================================
echo.

if not exist ".venv\Scripts\python.exe" (
    echo [ERROR] Virtual environment not found.
    echo Please install Python 3.11 and create .venv.
    pause
    exit /b 1
)

call ".venv\Scripts\activate.bat"

echo Launching Real-Time Presence Monitor...
echo - To calibrate baseline: press 'c' while keeping room still.
echo - To adjust threshold: press '+' or '-'.
echo - To quit: press 'q' or Esc.
echo.

python src\scope_monitor.py %*

if errorlevel 1 (
    echo.
    echo ------------------------------------------------------------
    echo Notice: No live ESP32 serial port was detected.
    echo Would you like to run in DEMO simulation mode right now?
    echo ------------------------------------------------------------
    set /p choice="Run Demo? (Y/N): "
    if /i "%choice%"=="Y" (
        python src\scope_monitor.py --demo
    )
)

pause
