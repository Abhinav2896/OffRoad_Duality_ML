#!/bin/bash

echo "=========================================="
echo "Duality AI - Integrated Autonomy Challenge"
echo "Environment Setup Script (Linux/Mac)"
echo "=========================================="

cd "$(dirname "$0")"

if [ ! -d "venv" ]; then
    echo "Creating virtual environment..."
    python3 -m venv venv
else
    echo "Virtual environment already exists."
fi

echo "Activating virtual environment..."
source venv/bin/activate

echo "Installing dependencies..."
pip install -r requirements.txt

echo "=========================================="
echo "Setup Complete!"
echo "To run the backend:"
echo "cd backend"
echo "source venv/bin/activate"
echo "uvicorn app:app --reload"
echo "=========================================="
