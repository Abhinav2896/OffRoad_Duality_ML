import os
import torch
import numpy as np
import cv2
import segmentation_models_pytorch as smp
import matplotlib.pyplot as plt
from model import get_model, NUM_CLASSES
from dataset import DualityDataset, visualize_mask, get_preprocessing

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
MODEL_PATH = '../model/best_model.pth'
OUTPUT_DIR = '../../outputs/overlays'

def run_inference(image_path, model, device):
    # Load and preprocess image
    image = cv2.imread(image_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    
    preprocessing_fn = smp.encoders.get_preprocessing_fn('resnet34', 'imagenet')
    preprocess = get_preprocessing(preprocessing_fn)
    
    sample = preprocess(image=image, mask=None) # No mask needed for inference
    x = torch.from_numpy(sample['image']).unsqueeze(0).to(device)
    
    with torch.no_grad():
        logits = model(x)
        pred_mask = torch.argmax(logits, dim=1).squeeze().cpu().numpy()
        
    return image, pred_mask

def save_overlay(image, mask, save_path):
    vis_mask = visualize_mask(mask)
    alpha = 0.5
    overlay = cv2.addWeighted(image, 1 - alpha, vis_mask, alpha, 0)
    
    # Save as BGR for cv2
    overlay_bgr = cv2.cvtColor(overlay, cv2.COLOR_RGB2BGR)
    cv2.imwrite(save_path, overlay_bgr)

if __name__ == '__main__':
    if not os.path.exists(OUTPUT_DIR):
        os.makedirs(OUTPUT_DIR)
        
    model = get_model(device=DEVICE, weights=MODEL_PATH)
    model.eval()
    
    # Example usage - would iterate over test set
    print(f"Model loaded from {MODEL_PATH}")
    print("Run this script with specific image paths to generate overlays.")
