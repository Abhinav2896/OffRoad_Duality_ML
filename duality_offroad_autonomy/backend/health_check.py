import sys
import pkg_resources

def check_package(package_name, min_version=None, max_version=None, exact_version=None):
    try:
        version = pkg_resources.get_distribution(package_name).version
        print(f"[{package_name}] Found version: {version}")
        
        if exact_version and version != exact_version:
             print(f"ERROR: {package_name} must be exactly {exact_version}")
             return False
             
        if min_version and pkg_resources.parse_version(version) < pkg_resources.parse_version(min_version):
            print(f"ERROR: {package_name} version {version} is too old. Min required: {min_version}")
            return False
            
        if max_version and pkg_resources.parse_version(version) >= pkg_resources.parse_version(max_version):
            print(f"ERROR: {package_name} version {version} is too new. Max allowed: <{max_version}")
            return False
            
        return True
    except pkg_resources.DistributionNotFound:
        print(f"ERROR: {package_name} not installed")
        return False

print("==========================================")
print("     ENVIRONMENT HEALTH CHECK")
print("==========================================")

failed = False

# 1. Critical: NumPy MUST be < 2.0.0 and >= 1.26.4 (we pinned 1.26.4)
if not check_package("numpy", min_version="1.26.4", max_version="2.0.0"):
    failed = True

# 2. Critical: OpenCV
if not check_package("opencv-python-headless", exact_version="4.9.0.80"):
    # Fallback check if user installed standard opencv-python instead
    if not check_package("opencv-python", exact_version="4.9.0.80"):
         failed = True

# 3. Check others
packages = ["torch", "fastapi", "uvicorn", "seaborn"]
for pkg in packages:
    if not check_package(pkg):
        failed = True

print("==========================================")
if failed:
    print("HEALTH CHECK FAILED! See errors above.")
    sys.exit(1)
else:
    # Try importing critical modules to catch runtime errors
    try:
        import numpy
        import cv2
        import torch
        print(f"Runtime Import Check: NumPy {numpy.__version__}, OpenCV {cv2.__version__}, Torch {torch.__version__}")
        print("HEALTH CHECK PASSED")
        sys.exit(0)
    except Exception as e:
        print(f"Runtime Import Failed: {e}")
        sys.exit(1)
