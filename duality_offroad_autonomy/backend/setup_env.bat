@echo off
echo ==========================================
echo Duality AI - Integrated Autonomy Challenge
echo Environment Setup Script (Windows/VENV)
echo ==========================================

cd /d "%~dp0"

echo Checking for Python...
python --version >nul 2>nul
if %errorlevel% neq 0 (
    echo Error: Python is not installed or not in your PATH.
    echo Please install Python 3.9+ from python.org
    exit /b 1
)

if not exist "venv" (
    echo Creating virtual environment 'venv'...
    python -m venv venv
) else (
    echo Virtual environment 'venv' already exists.
)

echo Activating virtual environment...
call venv\Scripts\activate
if %errorlevel% neq 0 (
    echo Failed to activate virtual environment.
    exit /b 1
)

echo Upgrading pip...
python -m pip install --upgrade pip

echo ------------------------------------------
echo CLEANING UP DEPENDENCY CONFLICTS...
echo Uninstalling potentially conflicting NumPy/OpenCV versions...
pip uninstall -y numpy opencv-python opencv-python-headless
echo ------------------------------------------

echo Installing dependencies (Pinned)...
pip install -r requirements.txt

echo ------------------------------------------
echo RUNNING HEALTH CHECK...
python health_check.py
if %errorlevel% neq 0 (
    echo HEALTH CHECK FAILED. Please review errors.
    exit /b 1
)

echo ==========================================
echo SETUP COMPLETE & VERIFIED!
echo To run the backend:
echo cd backend
echo .\venv\Scripts\activate
echo uvicorn app:app --reload
echo ==========================================
pause
