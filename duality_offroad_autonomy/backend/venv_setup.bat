@echo off
echo ==========================================
echo Duality AI - Integrated Autonomy Challenge
echo Environment Setup Script
echo ==========================================

cd /d "%~dp0"

if not exist "venv" (
    echo Creating virtual environment...
    python -m venv venv
) else (
    echo Virtual environment already exists.
)

echo Activating virtual environment...
call venv\Scripts\activate

echo Installing dependencies...
pip install -r requirements.txt

echo ==========================================
echo Setup Complete!
echo To run the backend:
echo cd backend
echo call venv\Scripts\activate
echo uvicorn app:app --reload
echo ==========================================
pause
