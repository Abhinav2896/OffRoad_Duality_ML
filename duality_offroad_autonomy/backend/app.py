import os
import io
import cv2
import base64
import numpy as np
import torch
import uvicorn
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.staticfiles import StaticFiles
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
import segmentation_models_pytorch as smp
from PIL import Image

from segmentation.model import get_model, NUM_CLASSES, DEVICE, get_model_path
from segmentation.dataset import visualize_mask, get_preprocessing
from path_planner import PathPlanner
from nlp_reasoner import NLPReasoner

# Configuration
OUTPUT_DIR = "../outputs/overlays"

app = FastAPI(title="Duality AI Autonomy API")

# CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global instances
model = None
path_planner = PathPlanner()
nlp_reasoner = NLPReasoner()
preprocessing_fn = smp.encoders.get_preprocessing_fn('resnet34', 'imagenet')
preprocess = get_preprocessing(preprocessing_fn)

def encode_image_to_base64(image_rgb):
    """Encodes a numpy RGB image to base64 string"""
    image_bgr = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR)
    _, buffer = cv2.imencode('.png', image_bgr)
    return base64.b64encode(buffer).decode('utf-8')

def run_inference(image_np):
    """
    Runs inference with automatic padding to ensure dimensions divisible by 32.
    Returns pred_mask (H, W) as numpy array.
    """
    h, w = image_np.shape[:2]
    pad_h = (32 - h % 32) % 32
    pad_w = (32 - w % 32) % 32
    
    if pad_h > 0 or pad_w > 0:
        image_padded = cv2.copyMakeBorder(image_np, 0, pad_h, 0, pad_w, cv2.BORDER_CONSTANT, value=[0, 0, 0])
    else:
        image_padded = image_np
        
    sample = preprocess(image=image_padded, mask=None)
    x = sample['image'].unsqueeze(0).to(DEVICE)
    
    with torch.no_grad():
        logits = model(x)
        pred_mask = torch.argmax(logits, dim=1).squeeze().cpu().numpy()
        
    if pad_h > 0 or pad_w > 0:
        pred_mask = pred_mask[:h, :w]
        
    return pred_mask

@app.on_event("startup")
async def startup_event():
    global model
    model_path = get_model_path()
    print(f"Loading model from {model_path}...")
    
    # Robust Model Loading
    try:
        if os.path.exists(model_path):
            model = get_model(device=DEVICE, weights=model_path)
            model.eval()
            print(f"Model loaded successfully from {model_path}")
        else:
            print(f"WARNING: Model file not found at {model_path}")
            print("CRITICAL: Inference will use random weights. Predictions will be garbage.")
            print("Please run backend/segmentation/train.py to generate the model.")
            model = get_model(device=DEVICE, weights=None)
            model.eval()
    except Exception as e:
        print(f"CRITICAL ERROR loading model: {e}")
        # Initialize empty model to keep server alive, but requests will likely fail or be garbage
        model = get_model(device=DEVICE, weights=None)
        model.eval()

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    """
    Returns the segmentation overlay image and metrics.
    Strict JSON contract:
    {
        "overlay_image": "<base64>",
        "metrics": { "miou": float, "per_class_iou": {...} }
    }
    """
    if model is None:
        raise HTTPException(status_code=500, detail="Model not initialized")
    
    try:
        # Read Image
        contents = await file.read()
        image = Image.open(io.BytesIO(contents)).convert("RGB")
        image_np = np.array(image)
        
        # Inference
        pred_mask = run_inference(image_np)
            
        # SAFETY ASSERTION: Ensure pred_mask is NumPy array for downstream OpenCV/NLP tasks
        assert isinstance(pred_mask, np.ndarray), "Prediction mask must be a NumPy array"
        assert isinstance(image_np, np.ndarray), "Input image must be a NumPy array"
            
        # Create Overlay
        vis_mask = visualize_mask(pred_mask)
        overlay = cv2.addWeighted(image_np, 0.6, vis_mask, 0.4, 0)
        
        # Calculate Metrics (Distribution)
        # Note: True IoU requires ground truth, which we don't have in inference.
        # We return class distribution as a proxy for "metrics" to satisfy the UI.
        total_pixels = pred_mask.size
        unique, counts = np.unique(pred_mask, return_counts=True)
        class_dist = {int(k): float(v)/total_pixels for k, v in zip(unique, counts)}
        
        # Format metrics to match expected schema
        # We use class distribution as "per_class_iou" placeholder since we can't compute real IoU
        metrics = {
            "miou": 0.0, # Placeholder as no GT available
            "per_class_distribution": class_dist
        }
        
        return {
            "overlay_image": encode_image_to_base64(overlay),
            "metrics": metrics
        }
    except Exception as e:
        print(f"Error in /predict: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/path")
async def get_path(file: UploadFile = File(...)):
    """
    Returns path image as base64.
    Strict JSON contract:
    {
        "path_image": "<base64>"
    }
    """
    if model is None:
        raise HTTPException(status_code=500, detail="Model not initialized")
        
    try:
        contents = await file.read()
        image = Image.open(io.BytesIO(contents)).convert("RGB")
        image_np = np.array(image)
        
        # Inference
        pred_mask = run_inference(image_np)
            
        assert isinstance(pred_mask, np.ndarray), "Prediction mask must be a NumPy array"
            
        # Plan Path using cost-aware planner
        path, _, _ = path_planner.plan_path_costaware(pred_mask, image_np)
        
        # Visualize Path
        vis_img = path_planner.visualize_path(image_np, path)
        
        return {
            "path_image": encode_image_to_base64(vis_img)
        }
    except Exception as e:
        print(f"Error in /path: {e}")
        try:
            if 'image_np' in locals() and isinstance(image_np, np.ndarray):
                return {
                    "path_image": encode_image_to_base64(image_np)
                }
        except Exception as _:
            pass
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/report")
async def get_report(file: UploadFile = File(...)):
    """
    Returns the NLP situational report.
    Strict JSON contract:
    {
        "status": "READY | CAUTION | BLOCKED",
        "summary": "string"
    }
    """
    if model is None:
        raise HTTPException(status_code=500, detail="Model not initialized")
        
    try:
        contents = await file.read()
        image = Image.open(io.BytesIO(contents)).convert("RGB")
        image_np = np.array(image)
        
        # Inference
        pred_mask = run_inference(image_np)
            
        assert isinstance(pred_mask, np.ndarray), "Prediction mask must be a NumPy array"
            
        # NLP Reasoner on cleaned mask (shadow-immune and noise-suppressed)
        cleaned_mask = path_planner.clean_mask_for_reasoner(pred_mask, image_np)
        result = nlp_reasoner.analyze_scene(cleaned_mask)
        
        return {
            "status": result["status"],
            "summary": result["report"]
        }
    except Exception as e:
        print(f"Error in /report: {e}")
        raise HTTPException(status_code=500, detail=str(e))

# Serve Frontend and Outputs
# Check if frontend exists
if os.path.exists("../frontend"):
    app.mount("/outputs", StaticFiles(directory="../outputs"), name="outputs")
    app.mount("/", StaticFiles(directory="../frontend", html=True), name="frontend")
else:
    print("WARNING: Frontend directory not found. API only mode.")

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
