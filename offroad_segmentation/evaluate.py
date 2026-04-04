import torch
from torch.utils.data import DataLoader
from torchvision import models
from dataset import SegmentationDataset
from utils import compute_iou
from tqdm import tqdm
import numpy as np
import matplotlib.pyplot as plt
from torchmetrics.classification import MulticlassAveragePrecision

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

def main():
    val_dataset = SegmentationDataset("../data/val/images", "../data/val/masks", split="val")
    val_loader = DataLoader(val_dataset, batch_size=4, num_workers=2)

    model = models.segmentation.deeplabv3_resnet50(weights=None)
    model.classifier[4] = torch.nn.Conv2d(256, 10, kernel_size=1)

    # ✅ Fix: Use strict=False to ignore training-only auxiliary classifier keys
    try:
        model.load_state_dict(torch.load("model.pth", map_location=DEVICE), strict=False)
        print("✓ Loaded model.pth")
    except FileNotFoundError:
        print("✗ model.pth not found.")
        return

    model.to(DEVICE)
    model.eval()

    ious = []
    conf_matrix = torch.zeros(10, 10, dtype=torch.int64)
    # ✅ Fix: Calculate mAP on CPU to save VRAM
    mAP_metric = MulticlassAveragePrecision(num_classes=10, average='macro').to('cpu')

    print(f"Evaluating on {DEVICE}...")
    with torch.no_grad():
        for imgs, masks in tqdm(val_loader, desc="Validating"):
            imgs = imgs.to(DEVICE)
            masks = masks.to(DEVICE).long()

            outputs = model(imgs)['out']
            preds = torch.argmax(outputs, dim=1)

            # ✅ Update mAP (Calculate on CPU to avoid Out of Memory)
            mAP_metric.update(outputs.cpu(), masks.cpu())

            # ✅ Fast Vectorized Confusion Matrix Update
            indices = 10 * masks.view(-1) + preds.view(-1)
            conf_matrix += torch.bincount(indices, minlength=100).reshape(10, 10).cpu()

            iou = compute_iou(outputs, masks).item()
            ious.append(iou)

    if ious:
        avg_iou = sum(ious) / len(ious)
        final_mAP = mAP_metric.compute().item()
        print(f"\nFinal Validation IoU: {avg_iou:.4f}")
        print(f"Final Validation mAP: {final_mAP:.4f}")
        
        # ✅ Save Heatmap
        save_confusion_matrix(conf_matrix)
    else:
        print("\nNo validation data found.")

def save_confusion_matrix(cm):
    cm = cm.cpu().numpy()
    # Normalize by row (true class proportions)
    cm_norm = cm.astype('float') / (cm.sum(axis=1)[:, np.newaxis] + 1e-6)

    plt.figure(figsize=(10, 8))
    plt.imshow(cm_norm, cmap='Blues')
    plt.title('Confusion Matrix (Normalized)')
    plt.colorbar()
    plt.xlabel('Predicted Class')
    plt.ylabel('True Class')

    # Add text annotations
    for i in range(10):
        for j in range(10):
            plt.text(j, i, f"{cm_norm[i, j]:.2f}", 
                     ha="center", va="center", color="black" if cm_norm[i,j] < 0.5 else "white")

    plt.savefig("confusion_matrix.png")
    print("✓ Confusion matrix saved as confusion_matrix.png")

if __name__ == "__main__":
    main()