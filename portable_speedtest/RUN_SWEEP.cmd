@echo off
setlocal
cd /d "%~dp0"
python sweep.py %*
exit /b %errorlevel%
