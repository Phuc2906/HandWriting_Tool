@echo off
echo === Handwriting Recognition App ===
echo.

REM Check if Python is installed
python --version >nul 2>&1
if errorlevel 1 (
    echo Error: Python is not installed or not in PATH
    pause
    exit /b 1
)

REM Run the app
echo Starting Handwriting Recognition App...
python run_app.py

pause