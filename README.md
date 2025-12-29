# Liveness Detection System (Section 1)

## 1. Project Overview
This project develops an AI model to distinguish between real faces (Live) and spoofed faces (Fake/Spoof) using Deep Learning.
- **Input:** RGB Images.
- **Output:** Liveness Score [0, 1] (0 = Real, 1 = Spoof).

## 2. Methodology & Rationale (Giải thích lựa chọn)

### A. Data Preprocessing (Crucial Step)
We observed that using raw images directly leads to "Data Leakage" where the model learns background noise (e.g., dark rooms, screen borders) instead of facial features.
- **Solution:** We implemented a robust Face Detection & Cropping pipeline using **OpenCV Haar Cascade**.
- **Scale Strategy:** Instead of tight cropping, we used a **Scale Factor of 1.6**.
    - *Rationale:* Expanding the crop allows the model to see context clues like phone bezels, paper edges, or lack of depth at the ears/neck, significantly improving Recall on Spoof attacks (from ~87% to ~97%).
- **Split:** Train/Dev/Test splits are consistent to ensure fair evaluation.

### B. Model Architecture
We selected **EfficientNet-B2** with Transfer Learning (ImageNet weights).
- *Rationale:* EfficientNet provides a better accuracy-to-latency ratio than older models like ResNet50 or VGG. B2 is chosen as a "sweet spot" for feature extraction depth without being too heavy for potential edge deployment.
- **Loss Function:** CrossEntropyLoss (Standard and effective for balanced data).

### C. Training Strategy
- **Optimizer:** Adam (LR=1e-3) with CosineAnnealing Scheduler for smooth convergence.
- **Regularization:** - `Dropout (p=0.3)` in the classifier head.
    - `EarlyStopping` to prevent overfitting.
- **Metric Focus:** We prioritize **Recall** (catching spoof attacks) over Precision, as missing a spoofer is a security risk.

## 3. Results (Performance)
Evaluated on the held-out Test Set:

| Metric | Value |
| :--- | :--- |
| **Accuracy** | 91.45% |
| **Recall (Spoof)** | **97.15%** |
| **ROC AUC** | **0.9704** |
| **F1-Score** | 0.9172 |

*> Note: The high Recall demonstrates the model's effectiveness in security contexts.*

## 4. Project Structure
```text
src/
├── classifier/
│   ├── data_module/    # Dataset & Dataloader logic
│   ├── model.py        # EfficientNet-B2 definition
│   ├── preprocess.py   # Face detection & cropping script
│   ├── train.py        # Main training loop
│   ├── predict.py      # Single image inference
│   └── ...
```

## 5. How to Run

### Installation
```bash
uv sync
```

### Preprocessing (Required first)
```bash
uv run src/classifier/preprocess.py
```

### Training
```bash
uv run -m classifier.train
```

### Inference (Demo)
```bash
uv run src/classifier/predict.py --image "data/raw/test/spoof/example.jpg"
```