import os
import csv
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.cuda.amp import autocast, GradScaler
import segmentation_models_pytorch as smp
import albumentations as albu
from sklearn.metrics import confusion_matrix, jaccard_score
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

from model import get_model, NUM_CLASSES, get_model_path
from dataset import DualityDataset, CLASS_NAMES, visualize_mask, get_preprocessing

# Configuration
# Robust path detection
DEFAULT_DATA_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../Data"))
BATCH_SIZE = 8
EPOCHS = 100  # Increased to 100 for Early Stopping
LR = 3e-4
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
OUTPUT_DIR = os.path.join(os.path.dirname(__file__), "../../outputs")
CSV_LOG_PATH = os.path.join(OUTPUT_DIR, "training_metrics.csv")

def get_training_augmentation():
    """
    Strong Data Augmentation for Better Generalization
    """
    train_transform = [
        # 1. Geometric Augmentations
        albu.HorizontalFlip(p=0.5),
        albu.ShiftScaleRotate(scale_limit=0.2, rotate_limit=10, shift_limit=0.1, p=0.7, border_mode=0),
        albu.RandomCrop(height=320, width=320, p=1.0),
        
        # 2. Color/Noise Augmentations
        albu.GaussNoise(p=0.2),
        albu.Perspective(p=0.5),
        albu.OneOf([
            albu.CLAHE(p=1),
            albu.RandomBrightnessContrast(p=1),
            albu.RandomGamma(p=1),
        ], p=0.5),
        
        albu.OneOf([
            albu.Sharpen(p=1),
            albu.Blur(blur_limit=3, p=1),
            albu.MotionBlur(blur_limit=3, p=1),
        ], p=0.2),

        # 3. Ensure Size
        albu.PadIfNeeded(min_height=320, min_width=320, border_mode=0, p=1),
    ]
    return albu.Compose(train_transform)

def get_validation_augmentation():
    """Add paddings to make image shape divisible by 32"""
    test_transform = [
        # Pad to nearest multiple of 32 (540 -> 544, 960 -> 960)
        albu.PadIfNeeded(min_height=None, min_width=None, pad_height_divisor=32, pad_width_divisor=32, border_mode=0)
    ]
    return albu.Compose(test_transform)

class EarlyStopping:
    """Early stops the training if validation mIoU doesn't improve after a given patience."""
    def __init__(self, patience=10, min_delta=0.001):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.best_score = None
        self.early_stop = False

    def __call__(self, val_miou):
        score = val_miou

        if self.best_score is None:
            self.best_score = score
        elif score < self.best_score + self.min_delta:
            self.counter += 1
            print(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.counter = 0

def train_epoch(model, loader, optimizer, loss_fn, device, scaler):
    model.train()
    running_loss = 0.0
    
    for images, masks in loader:
        images = images.to(device)
        masks = masks.to(device).long()
        
        optimizer.zero_grad()
        
        # Mixed Precision Training
        if device == 'cuda':
            with autocast():
                logits = model(images)
                loss = loss_fn(logits, masks)
            
            scaler.scale(loss).backward()
            scaler.scale(optimizer).step()
            scaler.update()
        else:
            logits = model(images)
            loss = loss_fn(logits, masks)
            loss.backward()
            optimizer.step()
        
        running_loss += loss.item()
        
    return running_loss / len(loader)

def validate(model, loader, loss_fn, device):
    model.eval()
    running_loss = 0.0
    all_preds = []
    all_targets = []
    
    with torch.no_grad():
        for images, masks in loader:
            images = images.to(device)
            masks = masks.to(device).long()
            
            # Mixed Precision Inference (Optional, but good for speed)
            if device == 'cuda':
                with autocast():
                    logits = model(images)
                    loss = loss_fn(logits, masks)
            else:
                logits = model(images)
                loss = loss_fn(logits, masks)
                
            running_loss += loss.item()
            
            preds = torch.argmax(logits, dim=1).cpu().numpy()
            all_preds.append(preds.flatten())
            all_targets.append(masks.cpu().numpy().flatten())
            
    all_preds = np.concatenate(all_preds)
    all_targets = np.concatenate(all_targets)
    
    # Calculate IoU
    iou = jaccard_score(all_targets, all_preds, average='weighted', labels=range(NUM_CLASSES))
    
    return running_loss / len(loader), iou, all_preds, all_targets

def plot_confusion_matrix(y_true, y_pred, save_path):
    cm = confusion_matrix(y_true, y_pred, labels=range(NUM_CLASSES))
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=False, fmt='d', cmap='Blues', 
                xticklabels=[CLASS_NAMES[i] for i in range(NUM_CLASSES)],
                yticklabels=[CLASS_NAMES[i] for i in range(NUM_CLASSES)])
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.title('Confusion Matrix')
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()

def plot_loss_curve(train_losses, val_losses, save_path):
    plt.figure(figsize=(10, 5))
    plt.plot(train_losses, label='Train Loss')
    plt.plot(val_losses, label='Val Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training and Validation Loss')
    plt.legend()
    plt.savefig(save_path)
    plt.close()

def save_per_class_iou(y_true, y_pred, save_path):
    iou_per_class = jaccard_score(y_true, y_pred, average=None, labels=range(NUM_CLASSES))
    
    data = []
    for i, iou in enumerate(iou_per_class):
        data.append({
            "Class ID": i,
            "Class Name": CLASS_NAMES[i],
            "IoU": f"{iou:.4f}"
        })
        
    df = pd.DataFrame(data)
    df.to_csv(save_path, index=False)
    print("\nPer-Class IoU:")
    print(df)

def main():
    # 1. Dataset Paths (Strict Adherence to Duality Structure)
    # Expected: Data/train/color_images, Data/train/segmentation
    
    # Try absolute path first, then relative fallback
    data_dir = os.environ.get("DUALITY_DATA_DIR", DEFAULT_DATA_DIR)
    
    if not os.path.exists(data_dir):
        # Fallback to finding Data folder relative to current script
        data_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../Data"))
    
    print(f"Using Dataset Directory: {data_dir}")
    
    if not os.path.exists(data_dir):
        print(f"CRITICAL ERROR: Dataset not found at {DEFAULT_DATA_DIR} or {os.path.abspath(data_dir)}")
        print("Please ensure 'Data' folder exists in project root.")
        return

    # Use actual Duality structure discovered
    # Data/Offroad_Segmentation_Training_Dataset/train/Color_Images
    train_root = os.path.join(data_dir, 'Offroad_Segmentation_Training_Dataset')
    
    x_train_dir = os.path.join(train_root, 'train', 'Color_Images')
    y_train_dir = os.path.join(train_root, 'train', 'Segmentation')
    x_valid_dir = os.path.join(train_root, 'val', 'Color_Images')
    y_valid_dir = os.path.join(train_root, 'val', 'Segmentation')

    # 2. Strict Validation of Path Existence
    for p in [x_train_dir, y_train_dir, x_valid_dir, y_valid_dir]:
        if not os.path.exists(p):
            print(f"CRITICAL: Required dataset subfolder missing: {p}")
            print("Ensure structure is: Data/{split}/color_images and Data/{split}/segmentation")
            return

    # 3. Log Image Counts
    try:
        num_train = len(os.listdir(x_train_dir))
        num_val = len(os.listdir(x_valid_dir))
        print(f"Dataset Verified:\n - Train Images: {num_train}\n - Val Images: {num_val}")
        
        if num_train == 0 or num_val == 0:
            print("Dataset folders are empty!")
            return
    except Exception as e:
        print(f"Error checking dataset content: {e}")
        return

    # Ensure Output Dirs
    os.makedirs(os.path.join(OUTPUT_DIR, "failure_gallery"), exist_ok=True)
    
    # Model & Preprocessing
    # Ensure ResNet34 ImageNet weights (handled by get_model default)
    model = get_model(device=DEVICE)
    preprocessing_fn = smp.encoders.get_preprocessing_fn('resnet34', 'imagenet')
    
    # 4. Create Datasets
    try:
        train_dataset = DualityDataset(
            x_train_dir, 
            y_train_dir, 
            augmentation=get_training_augmentation(),
            preprocessing=get_preprocessing(preprocessing_fn),
        )
        
        val_dataset = DualityDataset(
            x_valid_dir, 
            y_valid_dir, 
            augmentation=get_validation_augmentation(),
            preprocessing=get_preprocessing(preprocessing_fn),
        )
    except Exception as e:
        print(f"Dataset Structure Error: {e}")
        return
    
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=0, drop_last=True)
    val_loader = DataLoader(val_dataset, batch_size=1, shuffle=False, num_workers=0)
    
    # 5. Advanced Loss Function (Focal + Dice)
    dice_loss = smp.losses.DiceLoss(mode='multiclass')
    focal_loss = smp.losses.FocalLoss(mode='multiclass')
    # ce_loss = nn.CrossEntropyLoss() # Deprecated in favor of Focal
    
    def loss_fn(pred, target):
        return focal_loss(pred, target) + dice_loss(pred, target)
    
    # Optimizer & Scheduler
    optimizer = optim.AdamW(model.parameters(), lr=LR)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='max', factor=0.5, patience=5, min_lr=1e-6
    )
    
    # Mixed Precision Scaler
    scaler = GradScaler(enabled=(DEVICE == 'cuda'))
    
    # Early Stopping
    early_stopping = EarlyStopping(patience=10, min_delta=0.001)
    
    # Training Loop
    best_iou = 0.0
    train_losses = []
    val_losses = []
    
    # Initialize CSV Log
    with open(CSV_LOG_PATH, mode='w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['Epoch', 'Train Loss', 'Val Loss', 'Val mIoU', 'LR'])

    print(f"Starting Training for Max {EPOCHS} Epochs with Early Stopping...")
    
    for epoch in range(EPOCHS):
        current_lr = optimizer.param_groups[0]['lr']
        train_loss = train_epoch(model, train_loader, optimizer, loss_fn, DEVICE, scaler)
        
        # Validation
        val_loss, val_iou, _, _ = validate(model, val_loader, loss_fn, DEVICE)
        
        train_losses.append(train_loss)
        val_losses.append(val_loss)
        
        print(f"Epoch {epoch+1}/{EPOCHS} | LR: {current_lr:.2e} | Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f} | Val mIoU: {val_iou:.4f}")
        
        # Log to CSV
        with open(CSV_LOG_PATH, mode='a', newline='') as f:
            writer = csv.writer(f)
            writer.writerow([epoch+1, train_loss, val_loss, val_iou, current_lr])
        
        # Scheduler Step
        scheduler.step(val_iou)
        
        # Save Best Model
        if val_iou > best_iou:
            best_iou = val_iou
            save_path = os.path.join(OUTPUT_DIR, "best_model.pth")
            torch.save(model.state_dict(), save_path)
            print(f"Saved best model to: {save_path} (mIoU: {best_iou:.4f})")
        
        # Early Stopping Check
        early_stopping(val_iou)
        if early_stopping.early_stop:
            print("Early stopping triggered. Training finished.")
            break

    # Final Evaluation & Metrics Generation
    print("Generating Final Metrics...")
    # Load best model
    best_model_path = os.path.join(OUTPUT_DIR, "best_model.pth")
    model.load_state_dict(torch.load(best_model_path))
    _, final_iou, all_preds, all_targets = validate(model, val_loader, loss_fn, DEVICE)
    
    print(f"Final Best mIoU: {final_iou:.4f}")
    
    # 1. Loss Curves
    plot_loss_curve(train_losses, val_losses, os.path.join(OUTPUT_DIR, "loss_curve.png"))
    
    # 2. Confusion Matrix
    plot_confusion_matrix(all_targets, all_preds, os.path.join(OUTPUT_DIR, "confusion_matrix.png"))
    
    # 3. Per-Class IoU Table
    save_per_class_iou(all_targets, all_preds, os.path.join(OUTPUT_DIR, "iou_metrics.csv"))
    
    print("Training Complete. Metrics saved to outputs/")

if __name__ == '__main__':
    main()
