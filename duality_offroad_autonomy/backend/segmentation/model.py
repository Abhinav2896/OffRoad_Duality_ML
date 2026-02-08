import torch
import os
import segmentation_models_pytorch as smp

# Class Definition strictly according to specification
# 0: Sky (10000)
# 1: Landscape (7100)
# 2: Dry Grass (300)
# 3: Ground Clutter (550)
# 4: Rocks (800)
# 5: Logs (700)
# 6: Bushes (200)
# 7: Trees (100)
NUM_CLASSES = 8
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

def get_model_path():
    """
    Returns the single canonical absolute path to the model weights.
    Location: backend/model/best_model.pth
    """
    # This file is in backend/segmentation/model.py
    # We want backend/model/best_model.pth
    # So we go up one level to backend/, then into model/
    current_dir = os.path.dirname(os.path.abspath(__file__))
    backend_dir = os.path.dirname(current_dir)
    model_dir = os.path.join(backend_dir, "model")
    
    # Ensure directory exists
    if not os.path.exists(model_dir):
        os.makedirs(model_dir, exist_ok=True)
        
    return os.path.join(model_dir, "best_model.pth")

def get_model(device='cpu', weights=None):
    """
    Returns the DeepLabV3+ model with ResNet34 encoder.
    """
    model = smp.DeepLabV3Plus(
        encoder_name="resnet34",        # Encoder: ResNet34
        encoder_weights="imagenet",     # ImageNet pretrained
        in_channels=3,                  # RGB Input
        classes=NUM_CLASSES,            # 8 Output Classes
    )
    
    if weights:
        try:
            model.load_state_dict(torch.load(weights, map_location=device))
            print(f"Loaded weights from {weights}")
        except Exception as e:
            print(f"Could not load weights: {e}")
            
    model.to(device)
    return model
