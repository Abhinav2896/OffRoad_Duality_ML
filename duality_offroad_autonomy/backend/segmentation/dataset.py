import os
import numpy as np
import cv2
import torch
from torch.utils.data import Dataset as BaseDataset

# STRICT Class IDs from specification
CLASS_ID_MAP = {
    10000: 0,   # Sky
    7100: 1,    # Landscape (Ground)
    300: 2,     # Dry Grass
    550: 3,     # Ground Clutter
    800: 4,     # Rocks
    700: 5,     # Logs
    200: 6,     # Bushes
    100: 7      # Trees
}

# Reverse map for inference decoding
ID_TO_CLASS_MAP = {v: k for k, v in CLASS_ID_MAP.items()}

CLASS_NAMES = {
    0: "Sky",
    1: "Landscape",
    2: "Dry Grass",
    3: "Ground Clutter",
    4: "Rocks",
    5: "Logs",
    6: "Bushes",
    7: "Trees"
}

# Colors for visualization (RGB)
CLASS_COLORS = {
    0: [135, 206, 235], # Sky (SkyBlue)
    1: [128, 128, 128], # Landscape (Gray)
    2: [255, 255, 0],   # Dry Grass (Yellow)
    3: [165, 42, 42],   # Ground Clutter (Brown)
    4: [255, 0, 0],     # Rocks (Red)
    5: [139, 69, 19],   # Logs (SaddleBrown)
    6: [0, 255, 0],     # Bushes (Lime)
    7: [0, 128, 0]      # Trees (Green)
}

class DualityDataset(BaseDataset):
    """
    Dataset class for Duality AI Challenge.
    """
    def __init__(self, images_dir, masks_dir, augmentation=None, preprocessing=None):
        if not os.path.exists(images_dir):
             raise FileNotFoundError(f"Images directory not found: {images_dir}")
        if not os.path.exists(masks_dir):
             raise FileNotFoundError(f"Masks directory not found: {masks_dir}")
             
        self.ids = os.listdir(images_dir)
        self.images_fps = [os.path.join(images_dir, image_id) for image_id in self.ids]
        self.masks_fps = [os.path.join(masks_dir, image_id) for image_id in self.ids]
        
        # Validation: Check if corresponding masks exist
        valid_indices = []
        for i, mask_path in enumerate(self.masks_fps):
            if os.path.exists(mask_path):
                valid_indices.append(i)
            else:
                print(f"Warning: Mask not found for {self.images_fps[i]}, skipping.")
        
        self.ids = [self.ids[i] for i in valid_indices]
        self.images_fps = [self.images_fps[i] for i in valid_indices]
        self.masks_fps = [self.masks_fps[i] for i in valid_indices]

        self.augmentation = augmentation
        self.preprocessing = preprocessing
    
    def __getitem__(self, i):
        # Read image
        image = cv2.imread(self.images_fps[i])
        if image is None:
             # Handle corrupted image gracefully
             print(f"Error reading image: {self.images_fps[i]}")
             # Return a dummy zero tensor or handle in DataLoader (hard to do inside getitem)
             # Better to fail loudly during training for now
             raise ValueError(f"Could not read image: {self.images_fps[i]}")

        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # Read mask (Load as 16-bit to preserve IDs like 7100, 10000)
        mask = cv2.imread(self.masks_fps[i], cv2.IMREAD_UNCHANGED)
        if mask is None:
             raise ValueError(f"Could not read mask: {self.masks_fps[i]}")
        
        # Remap IDs to 0-7
        # Create a mask filled with a 'ignore' index or default class if needed.
        # Here we assume all pixels belong to one of the classes. 
        # If there are unknown classes, they should probably be mapped to Clutter or similar.
        # We initialize with 3 (Ground Clutter) as a safe default if needed, or 0 (Sky).
        mask_mapped = np.zeros_like(mask, dtype=np.uint8)
        
        for original_id, train_id in CLASS_ID_MAP.items():
            mask_mapped[mask == original_id] = train_id
            
        # apply augmentations
        if self.augmentation:
            sample = self.augmentation(image=image, mask=mask_mapped)
            image, mask_mapped = sample['image'], sample['mask']
            
        # apply preprocessing
        if self.preprocessing:
            sample = self.preprocessing(image=image, mask=mask_mapped)
            image, mask_mapped = sample['image'], sample['mask']
            
        return image, mask_mapped.long()
        
    def __len__(self):
        return len(self.ids)

def visualize_mask(mask_idx):
    """
    Convert a (H, W) mask of indices 0-7 to an RGB image.
    """
    h, w = mask_idx.shape
    vis_img = np.zeros((h, w, 3), dtype=np.uint8)
    for class_id, color in CLASS_COLORS.items():
        vis_img[mask_idx == class_id] = color
    return vis_img

def get_preprocessing(preprocessing_fn):
    """Construct preprocessing transform"""
    import albumentations as albu
    
    def to_tensor(x, **kwargs):
        if x.ndim == 3:
            return torch.from_numpy(x.transpose(2, 0, 1).astype('float32'))
        elif x.ndim == 2:
            return torch.from_numpy(x.astype('int64'))
        else:
            raise ValueError(f"Unsupported dimension count: {x.ndim}")
        
    _transform = [
        albu.Lambda(image=preprocessing_fn),
        albu.Lambda(image=to_tensor, mask=to_tensor),
    ]
    return albu.Compose(_transform)
