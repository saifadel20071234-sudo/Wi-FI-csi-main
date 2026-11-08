@echo off
setlocal EnableExtensions
cd /d "%~dp0CSI"
call run_monitor.bat %*
