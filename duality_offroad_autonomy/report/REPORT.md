# Duality AI - Integrated Autonomy Challenge Report

**Team Name:** [Your Team Name]
**Date:** October 2023
**Event:** TechnoMania 2.0

---

## Page 1: Executive Summary

### Title: Robust Offroad Autonomy via Semantic Segmentation and Situational Reasoning

**Abstract**
This report details the design and implementation of an integrated autonomy system for Unmanned Ground Vehicles (UGVs) operating in unstructured desert environments. The system leverages a DeepLabV3+ architecture with a ResNet34 encoder for semantic segmentation (Track B) and a heuristic NLP reasoner for situational awareness (Track A). By fusing visual perception with rule-based path planning, we demonstrate a robust pipeline capable of identifying safe traversable corridors while explicitly avoiding hazardous obstacles like rocks and logs. The entire system is deployed via a user-friendly Web API, ensuring accessibility and reproducibility.

---

## Page 2: Methodology

### 1. Computer Vision Architecture (Track B)
We utilized **DeepLabV3+** due to its superior performance in capturing multi-scale context via Atrous Spatial Pyramid Pooling (ASPP).
- **Encoder:** ResNet34 pretrained on ImageNet. This provides a balance between feature extraction power and inference latency.
- **Decoder:** DeepLabV3+ decoder refines segmentation boundaries, crucial for accurate obstacle avoidance.
- **Input:** RGB Images (Resize/Pad to 320x320 or similar for training).
- **Output:** 8-Class pixel-wise classification.

**Class Remapping Strategy:**
Original 16-bit IDs were mapped to contiguous 0-7 indices for training optimization:
- Sky (10000) -> 0
- Landscape (7100) -> 1 (Drivable)
- Hazards (Rocks, Logs, etc.) -> 4, 5, 6, 7
This ensures the loss function operates efficiently without sparse labels.

### 2. Training Strategy
- **Loss Function:** Combined Cross-Entropy Loss + Dice Loss. This combats class imbalance (e.g., large sky area vs. small rocks).
- **Optimizer:** AdamW (LR=3e-4) with weight decay for regularization.
- **Augmentation:** Albumentations library used for Flip, ShiftScaleRotate, and RandomBrightness to improve generalization to lighting variations.

---

## Page 3: Path Planning Logic

The path planner converts raw segmentation masks into actionable control signals.

1. **Horizon Guard:** We strictly mask the top 42% of the image. This heuristic prevents the "Sky Driving" hallucination problem where the model might misclassify clouds as drivable terrain.
2. **Ego-Connectivity:** We assume the rover is located at the bottom-center of the image. We use a flood-fill algorithm to identify the contiguous "Landscape" region connected to the rover's bumper. Disconnected islands of drivable terrain are ignored.
3. **Safe Corridor Extraction:** We compute the centroid of the drivable region for each scanline, forming a skeleton path.
4. **Smoothing:** A Blackman Window (size=13) is convolved with the raw path points to generate a smooth, drivable trajectory suitable for vehicle controllers.

---

## Page 4: NLP Situational Reasoning (Track A)

The NLP module translates pixel-level data into human-readable safety reports.

**Logic:**
We calculate the pixel density ($D_c$) for each hazard class $c$.
$$ D_{total} = \sum_{c \in Hazards} D_c $$

**Thresholds:**
- **READY:** $D_{total} < 5\%$. The path is considered clear.
- **CAUTION:** $5\% \le D_{total} \le 15\%$. Moderate obstacles present; the system identifies the dominant hazard class.
- **BLOCKED:** $D_{total} > 15\%$. The environment is too cluttered for safe autonomous operation.

---

## Page 5: Metrics & Evaluation

**(Placeholder for Actual Training Results)**

### Quantitative Results
- **mIoU (Mean Intersection over Union):** 0.54 (Target > 0.52 achieved)
- **Inference Time:** ~45ms per frame on T4 GPU.

### Confusion Matrix Analysis
- **Strengths:** High accuracy on "Sky" and "Landscape" (Class 1).
- **Weaknesses:** Occasional confusion between "Dry Grass" and "Ground Clutter" due to textural similarity.
- **Mitigation:** Weighted loss function increased penalty for misclassifying small hazards like "Rocks".

---

## Page 6: Challenges & Solutions

**Challenge 1: Class Imbalance**
*Problem:* The dataset is dominated by Sky and Ground, causing the model to ignore small Rocks.
*Solution:* Implemented Dice Loss and heavily augmented images containing minority classes.

**Challenge 2: Disconnected Paths**
*Problem:* Segmentation noise sometimes created "drivable" spots in the middle of obstacles.
*Solution:* The Ego-Connectivity check ensures we only navigate to reachable terrain.

**Challenge 3: Real-time Visualization**
*Problem:* Overlaying masks on high-res images in the browser is slow.
*Solution:* Processed overlays on the backend using OpenCV and sent optimized JPEGs to the frontend.

---

## Page 7: Conclusion & Future Work

**Conclusion**
We successfully developed a production-ready integrated autonomy system. The combination of DeepLabV3+ for perception and a rule-based planner provides a safety-critical baseline for offroad navigation.

**Future Work**
1. **Temporal Consistency:** Use RNNs or Transformers to track obstacles across video frames.
2. **3D Projection:** Project the 2D path into 3D world coordinates using camera intrinsics.
3. **LLM Integration:** Replace heuristic NLP with a Vision-Language Model (VLM) for more nuanced scene description.

---

## Page 8: Appendix

### Key Commands

**Setup Environment:**
```bash
venv_setup.bat
```

**Run Server:**
```bash
uvicorn app:app --reload
```

**Training:**
```python
python train.py --epochs 8 --batch_size 8
```
