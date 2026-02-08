# Duality AI - Integrated Autonomy System (TechnoMania 2.0)

## 1. Project Overview
This project is an end-to-end Integrated Autonomy System designed for the Duality AI Challenge. It combines Computer Vision (Track B) and NLP Situational Reasoning (Track A) into a unified Full-Stack Web Application.

**Key Features:**
- **Semantic Segmentation:** DeepLabV3+ with ResNet34 encoder trained on synthetic desert data.
- **Autonomous Path Planning:** Horizon Guard, Ego-Connectivity checks, and Blackman Window smoothing for safe trajectory generation.
- **Situational Reasoning (NLP):** Automatic generation of "READY", "CAUTION", or "BLOCKED" reports based on scene hazard analysis.
- **Web Interface:** Modern JavaScript-based UI for real-time visualization and interaction.

## 2. Project Structure
```
duality_offroad_autonomy/
│
├── Data/                     # Dataset (train, val, testImages)
├── backend/
│   ├── app.py                # FastAPI Main Application
│   ├── requirements.txt      # Python Dependencies
│   ├── venv_setup.bat        # Windows Setup Script
│   ├── setup_env.sh          # Linux/Mac Setup Script
│   ├── model/                # Model weights storage
│   ├── segmentation/         # Model definition & training logic
│   │   ├── dataset.py
│   │   ├── model.py
│   │   ├── train.py
│   │   └── test.py
│   ├── path_planner.py       # Track B Logic
│   └── nlp_reasoner.py       # Track A Logic
│
├── frontend/                 # Web UI
│   ├── index.html
│   ├── style.css
│   └── script.js
│
├── outputs/                  # Generated results
│   ├── overlays/
│   ├── failure_gallery/
│   ├── loss_curve.png
│   ├── confusion_matrix.png
│   └── iou_metrics.csv
│
└── report/
    └── REPORT.md             # Technical Report Source
```

### Model Storage Convention
The model weights are strictly saved and loaded from a single canonical location:
**`backend/model/best_model.pth`**

This ensures consistency between the training script (`train.py`) and the inference API (`app.py`). Both scripts use absolute path resolution to locate this file, preventing errors regardless of the working directory.

## 3. Installation & Setup

### Prerequisites
- Python 3.9+ (Recommended)
- CUDA-enabled GPU (recommended)
- Duality Challenge Dataset in `Data/` folder.

> **Note on Dependencies:** This project requires **NumPy < 2.0** (specifically pinned to `1.26.4`) due to binary incompatibilities between NumPy 2.x and current OpenCV/Torch versions. The setup scripts handle this automatically.

### Automatic Setup
**Windows:**
```powershell
.\setup_env.bat
```

**Linux/Mac:**
```bash
cd backend
chmod +x setup_env.sh
./setup_env.sh
```
These scripts will create a virtual environment and install all dependencies.

## 4. Usage

### 1. Training the Model
The system expects the dataset at `../Data` relative to the backend.
```bash
cd backend/segmentation
python train.py
```
This will:
- Train for 8 epochs.
- Save `best_model.pth` to `backend/model/`.
- Generate `loss_curve.png`, `confusion_matrix.png`, and `iou_metrics.csv` in `outputs/`.

### 2. Running the Application
1. Start the Backend API:
   ```bash
   cd backend
   # Ensure venv is activated
   # Windows: call venv\Scripts\activate
   # Linux: source venv/bin/activate
   uvicorn app:app --reload
   ```
2. Open your browser and navigate to:
   `http://localhost:8000`

### 3. Using the Interface
1. Click **Choose File** and select a test image (e.g., from `Data/testImages/images`).
2. Click **Process Image**.
3. View the generated:
   - **Segmentation Overlay:** Visualizes terrain classes.
   - **Planned Path:** Shows the calculated safe corridor.
   - **NLP Report:** Textual status and hazard metrics.

## 5. Judge Verification Steps
1. **Dataset Detection:** The code automatically looks for `Data/` in the project root. If moved, ensure the path is correct or `dataset.py` logic will attempt to find it.
2. **Reproducibility:** Run `python train.py` to regenerate metrics. The random seeds are not fixed by default to allow exploration, but model architecture and hyperparameters are strict.
3. **Path Safety:** Inspect `outputs/overlays`. The `path_planner.py` explicitly checks that no path point lies on Class 4 (Rocks) or Class 0 (Sky).
4. **NLP Logic:** Verify that the "Status" (READY/CAUTION/BLOCKED) matches the visual density of hazards in the overlay.

## 6. Technical Details
- **Model:** DeepLabV3+ (ResNet34)
- **Classes:** Landscape (7100), Rocks (800), Logs (700), Bushes (200), Trees (100), Sky (10000), Dry Grass (300), Ground Clutter (550).
- **Path Planning:** Uses morphological operations and graph-based connectivity to ensure the rover only traverses safe, contiguous ground connected to its current position.
- **NLP:** Heuristic-based reasoning derived from pixel-wise class distribution statistics.

## 7. Credits
Developed for the Duality AI TechnoMania 2.0 Challenge.
