@echo off
cd /d "%~dp0"
py -3 download_sources.py
if errorlevel 1 python download_sources.py
pause
