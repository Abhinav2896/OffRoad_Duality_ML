@echo off
echo ======================================================
echo Duality AI - Environment Health Check
echo ======================================================

python -c "import torch; print(f'Torch: {torch.__version__} (CUDA: {torch.cuda.is_available()})')"
if %errorlevel% neq 0 goto :error

python -c "import cv2; print(f'OpenCV: {cv2.__version__}')"
if %errorlevel% neq 0 goto :error

python -c "import segmentation_models_pytorch; print('SMP: OK')"
if %errorlevel% neq 0 goto :error

python -c "import fastapi; print(f'FastAPI: {fastapi.__version__}')"
if %errorlevel% neq 0 goto :error

python -c "import seaborn; print(f'Seaborn: {seaborn.__version__}')"
if %errorlevel% neq 0 goto :error

echo ======================================================
echo HEALTH CHECK PASSED: All dependencies installed.
echo ======================================================
exit /b 0

:error
echo ======================================================
echo HEALTH CHECK FAILED: Missing dependencies.
echo Please check the error messages above.
echo ======================================================
exit /b 1
