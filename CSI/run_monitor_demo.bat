@echo off
setlocal EnableExtensions
cd /d "%~dp0"

echo ============================================================
echo   WiFi CSI People Counter - DEMO MODE (Simulating Passages)
echo ============================================================
echo.
echo Simulating: 1 Person crossing (+1), then 2 People crossing (+2)...
echo Press 'q' in the window to quit, 'z' to reset counter.
echo.

call ".venv\Scripts\activate.bat"
python src\scope_monitor.py --demo

pause
