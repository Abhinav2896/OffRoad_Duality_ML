# Off-Road Autonomous Vision: TechnoMania 2.0 (Duality AI)

This repository contains the Phase 1 solution for the **Integrated Autonomy Challenge**.  
We developed an end-to-end deep learning pipeline to enable Unmanned Ground Vehicles (UGVs) to navigate complex off-road desert terrain using synthetic data from the Duality AI Falcon platform.

---

## 🚀 Project Overview

The core of this project is a **Semantic Segmentation** system that classifies every pixel in a camera feed into terrain or object categories. This visual understanding enables the UGV to:
*   ✅ **Identify Drivable Surfaces**: Landscape, dry grass, and flat ground.
*   ❌ **Avoid Obstacles**: Rocks, logs, trees, and bushes.
*   🧠 **Reason About Hazards**: Analyze scene complexity and issue safety reports.
*   🛣️ **Plan Paths**: Generate safe trajectories through obstacle fields.

### **Key Deliverables**
*   **Segmentation Model**: DeepLabV3+ with ResNet34 backbone trained on ~3000 synthetic images.
*   **Path Planner**: Potential-field-based algorithm for safe corridor detection.
*   **NLP Reasoner**: Automated hazard analysis and status reporting (READY/CAUTION/BLOCKED).
*   **Web Dashboard**: Real-time visualization of segmentation, path planning, and reports.

---

## 🛠️ Technical Implementation

### 1. Segmentation Architecture
*   **Model**: DeepLabV3+ (State-of-the-art for semantic segmentation).
*   **Backbone**: ResNet34 (Pretrained on ImageNet).
*   **Input Resolution**: 544x960 (Padding handled automatically).
*   **Classes**:
    *   0: Sky
    *   1: Landscape (Drivable)
    *   2: Dry Grass (Drivable)
    *   3: Ground Clutter
    *   4: Rocks (Hard Obstacle)
    *   5: Logs (Hard Obstacle)
    *   6: Bushes (Soft Obstacle)
    *   7: Trees (Hard Obstacle)

### 2. Training Pipeline Upgrades
To maximize performance on the Duality dataset, we implemented several advanced training strategies in `backend/segmentation/train.py`:
*   **Loss Function**: Combined **Focal Loss + Dice Loss** to handle class imbalance (e.g., small rocks vs. large sky).
*   **Optimization**: `AdamW` optimizer with `ReduceLROnPlateau` scheduler.
*   **Early Stopping**: Automatically stops training when validation mIoU plateaus (Patience=10).
*   **Augmentation**: Aggressive geometric and color transformations (Flip, ShiftScaleRotate, RandomGamma, Blur) to improve generalization.
*   **Mixed Precision**: Uses `torch.cuda.amp` for faster, memory-efficient training.

### 3. Path Planning & Reasoning
*   **Path Planner** (`backend/path_planner.py`): Converts segmentation masks into a cost map. It uses a potential field approach where obstacles exert a repulsive force, guiding the path through the safest "valleys" (corridors).
*   **NLP Reasoner** (`backend/nlp_reasoner.py`): Calculates hazard density statistics and generates a natural language safety report (e.g., "CAUTION: Moderate hazard density (22.5%). Watch for Rocks.").

---

## 📂 Repository Structure

```
duality_offroad_autonomy/
├── backend/
│   ├── app.py                 # FastAPI backend server
│   ├── path_planner.py        # Path finding logic
│   ├── nlp_reasoner.py        # Hazard analysis logic
│   ├── model/                 # Model weights directory
│   │   └── best_model.pth     # Trained model weights
│   ├── segmentation/
│   │   ├── train.py           # Advanced training script
│   │   ├── model.py           # DeepLabV3+ definition
│   │   └── dataset.py         # Duality dataset loader
│   └── venv/                  # Python virtual environment
├── frontend/
│   ├── index.html             # Dashboard UI
│   ├── script.js              # Frontend logic
│   └── style.css              # Styling
├── outputs/                   # Generated artifacts
│   ├── training_metrics.csv   # Training logs
│   ├── loss_curve.png         # Performance graphs
│   └── confusion_matrix.png   # Class accuracy analysis
└── report/
    └── REPORT.md              # Detailed project report
```

---

## 💻 Installation & Usage

### Prerequisites
*   Python 3.8+
*   CUDA-enabled GPU (Recommended for training)

### 1. Setup Environment
```bash
# Clone the repository
git clone https://github.com/Abhinav2896/OffRoad_Duality_ML.git
cd OffRoad_Duality_ML

# Create and activate virtual environment (Windows)
python -m venv duality_offroad_autonomy/backend/venv
.\duality_offroad_autonomy\backend\venv\Scripts\activate

# Install dependencies
pip install -r duality_offroad_autonomy/backend/requirements.txt
```

### 2. Run Training
To retrain the model with the upgraded pipeline:
```bash
python duality_offroad_autonomy/backend/segmentation/train.py
```
*Outputs (logs, best model, graphs) will be saved to `duality_offroad_autonomy/outputs/`.*

### 3. Run the Dashboard
Start the backend server:
```bash
python duality_offroad_autonomy/backend/app.py
```
Then open `duality_offroad_autonomy/frontend/index.html` in your web browser to interact with the system.

---

## 📊 Performance & Artifacts

### Key Metrics
*   **Best Validation mIoU**: ~0.61 (Targeting 0.75+ with current upgrades)
*   **Inference Speed**: ~85ms per frame on RTX 3060

### Trained Model Weights
Due to GitHub file size limits, the `best_model.pth` is included in this repo via Git LFS or available externally if needed.
*   **Location**: `duality_offroad_autonomy/backend/model/best_model.pth`



*This project was developed for the Duality AI Integrated Autonomy Challenge.*
