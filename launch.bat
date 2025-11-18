@echo off
REM Burpee Tracker Launcher for Windows
REM One-click startup script

echo ========================================
echo    Burpee Tracker by Simon Wong
echo ========================================
echo.

REM Check if Python is installed
python --version >nul 2>&1
if errorlevel 1 (
    echo ERROR: Python is not installed or not in PATH
    echo Please install Python 3.8+ from python.org
    pause
    exit /b 1
)

echo [1/3] Checking dependencies...
pip show flask >nul 2>&1
if errorlevel 1 (
    echo Installing Flask...
    pip install flask
)

pip show opencv-python >nul 2>&1
if errorlevel 1 (
    echo Installing OpenCV...
    pip install opencv-python
)

pip show mediapipe >nul 2>&1
if errorlevel 1 (
    echo Installing MediaPipe...
    pip install mediapipe
)

echo [2/3] Starting server...
echo Server will start at http://localhost:5000
echo.

REM Start Python server in background
start /B python burpee_tracker.py

REM Wait for server to initialize
echo Waiting for server to start...
timeout /t 3 /nobreak >nul

echo [3/3] Opening browser...
start http://localhost:5000

echo.
echo ========================================
echo Server is running!
echo Press Ctrl+C to stop the server
echo ========================================
echo.

REM Keep window open and wait for Ctrl+C
pause >nul