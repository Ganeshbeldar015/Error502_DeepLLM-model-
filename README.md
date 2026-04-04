# Offroad Semantic Segmentation for Autonomous Navigation

An image segmentation model built using **DeepLabV3+ with a ResNet-50 backbone**, designed specifically for offroad autonomous navigation. The model identifies safe traversable terrain, grass, obstacles, and sky in complex off-road environments.

## 🚀 Key Features
- **High IoU Targeting**: Engineered to reach an IoU of **0.80+** using advanced training techniques.
- **Data Augmentation**: Robust training pipeline including **Random Horizontal Flips** and **ImageNet Normalization** for better generalization.
- **Interactive Evaluation**: Generates a **Normalized Confusion Matrix** as a heatmap to identify specific class confusions.
- **Visual Inference**: A dedicated visualization script to see side-by-side results of original images, ground truth, and AI predictions.

## 🛠️ Tech Stack
- **Framework**: `PyTorch` / `Torchvision`
- **Architecture**: `DeepLabV3+ (ResNet-50)`
- **Utilities**: `OpenCV`, `Matplotlib`, `NumPy`, `tqdm`
- **Training Strategy**: Mixed Precision (AMP), Plateau Learning Rate Scheduling, and Dice+CE (planned) loss functions.

## 📂 Project Structure
```text
├── data/                       # Dataset (Images & Masks)
├── offroad_segmentation/       # Training and evaluation logic
│   ├── train.py                # Main training loop
│   ├── evaluate.py             # Performance measurement (IoU & CM)
│   ├── visualize_results.py    # Qualitative visualization
│   ├── dataset.py              # Normalized data loading & augmentation
│   ├── utils.py                # Helper functions (IoU, metrics)
│   └── model.pth               # Saved model weights
└── requirements.txt            # Project dependencies
```

## ⚙️ Setup & Installation
1. Ensure your Python environment is activated:
   ```powershell
   conda activate seg
   ```
2. Install dependencies (if not already installed):
   ```powershell
   pip install -r requirements.txt
   ```

## 🚦 How to Run
### 1. Training
Start the training loop (now includes auto-saving after every epoch):
```powershell
python offroad_segmentation/train.py
```

### 2. Comprehensive Evaluation
Calculate the final IoU and generate the **Confusion Matrix Heatmap**:
```powershell
python offroad_segmentation/evaluate.py
```

### 3. Visual Analysis
Generate a comparison of 5 random images with their predicted masks:
```powershell
python offroad_segmentation/visualize_results.py
```

## 📊 Current Progress
- **Target IoU**: 0.80
- **Current Milestone**: 0.56+ (Achieved with basic setup)
- **Next Phase**: Implementing Hybrid Dice + Cross-Entropy Loss and more advanced augmentations.
---
Developed with ❤️ for Autonomous Robotics research.
