@echo off
setlocal
cd /d "%~dp0"
python benchmark.py %*
exit /b %errorlevel%
