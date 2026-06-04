import torch
from torch.utils.data import DataLoader
from torchvision import models
from dataset import SegmentationDataset
from utils import compute_iou
from tqdm import tqdm
import numpy as np
import matplotlib.pyplot as plt
from torchmetrics.classification import MulticlassAveragePrecision
import yaml
import os

def load_config():
    config_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "config.yaml")
    with open(config_path, "r") as f:
        return yaml.safe_load(f)

def main():
    config = load_config()
    
    # Determine device
    if config["training"]["device"] == "auto":
        DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    else:
        DEVICE = config["training"]["device"]

    val_dataset = SegmentationDataset(
        config["data"]["val"]["images"], 
        config["data"]["val"]["masks"], 
        split="val"
    )
    val_loader = DataLoader(
        val_dataset, 
        batch_size=config["evaluation"]["batch_size"], 
        num_workers=config["evaluation"]["num_workers"],
        pin_memory=True,
        persistent_workers=True
    )

    model = models.segmentation.deeplabv3_resnet50(weights=None)
    model.classifier[4] = torch.nn.Conv2d(256, config["training"]["num_classes"], kernel_size=1)

    try:
        model.load_state_dict(torch.load("model.pth", map_location=DEVICE), strict=False)
        print("✓ Loaded model.pth")
    except FileNotFoundError:
        print("✗ model.pth not found.")
        return

    model.to(DEVICE)
    model.eval()

    num_classes = config["training"]["num_classes"]
    conf_matrix = torch.zeros(num_classes, num_classes, dtype=torch.int64)
    mAP_metric = MulticlassAveragePrecision(num_classes=num_classes, average='macro').to('cpu')

    print(f"Evaluating on {DEVICE}...")
    with torch.no_grad():
        for imgs, masks in tqdm(val_loader, desc="Validating"):
            imgs = imgs.to(DEVICE)
            masks = masks.to(DEVICE).long()

            outputs = model(imgs)['out']
            preds = torch.argmax(outputs, dim=1)

            if config["evaluation"]["downsample_for_map"]:
                downsample_size = tuple(config["evaluation"]["downsample_size"])
                outputs_small = torch.nn.functional.interpolate(
                    outputs, 
                    size=downsample_size, 
                    mode='bilinear', 
                    align_corners=False
                )
                masks_small = torch.nn.functional.interpolate(
                    masks.unsqueeze(1).float(), 
                    size=downsample_size, 
                    mode='nearest'
                ).squeeze(1).long()
                mAP_metric.update(outputs_small.cpu(), masks_small.cpu())
            else:
                mAP_metric.update(outputs.cpu(), masks.cpu())

            indices = num_classes * masks.view(-1) + preds.view(-1)
            conf_matrix += torch.bincount(indices, minlength=num_classes*num_classes).reshape(num_classes, num_classes).cpu()

    if conf_matrix.sum() > 0:
        print("\nCalculating final metrics...")
        tp = torch.diag(conf_matrix)
        fp = torch.sum(conf_matrix, dim=0) - tp
        fn = torch.sum(conf_matrix, dim=1) - tp
        union = tp + fp + fn
        ious_per_class = tp / (union + 1e-6)
        avg_iou = ious_per_class.mean().item()
        
        final_mAP = mAP_metric.compute().item()
        
        pixel_accuracy = (torch.diag(conf_matrix).sum() / conf_matrix.sum()).item()
        
        print(f"Final Validation IoU: {avg_iou:.4f}")
        print(f"Final Validation mAP: {final_mAP:.4f}")
        print(f"Final Pixel Accuracy: {pixel_accuracy:.4f}")
        
        save_confusion_matrix(conf_matrix, num_classes)
    else:
        print("\nNo validation data found.")

def save_confusion_matrix(cm, num_classes):
    cm = cm.cpu().numpy()
    cm_norm = cm.astype('float') / (cm.sum(axis=1)[:, np.newaxis] + 1e-6)

    plt.figure(figsize=(10, 8))
    plt.imshow(cm_norm, cmap='Blues')
    plt.title('Confusion Matrix (Normalized)')
    plt.colorbar()
    plt.xlabel('Predicted Class')
    plt.ylabel('True Class')

    for i in range(num_classes):
        for j in range(num_classes):
            plt.text(j, i, f"{cm_norm[i, j]:.2f}", 
                     ha="center", va="center", color="black" if cm_norm[i,j] < 0.5 else "white")

    plt.savefig("confusion_matrix.png")
    print("✓ Confusion matrix saved as confusion_matrix.png")

if __name__ == "__main__":
    main()