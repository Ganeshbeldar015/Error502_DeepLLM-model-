import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import models
from torchvision.models.segmentation import DeepLabV3_ResNet50_Weights
from dataset import SegmentationDataset
from tqdm import tqdm
from utils import compute_iou

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

def main():
    # 🔥 FAST MODE (limit dataset for now)
    train_dataset = SegmentationDataset("../data/train/images", "../data/train/masks", split="train")
    val_dataset = SegmentationDataset("../data/val/images", "../data/val/masks", split="val")

    train_loader = DataLoader(
        train_dataset,
        batch_size=2,              # 📉 Reduced batch size (OOM prevention)
        shuffle=True,
        num_workers=4,            # 🔥 faster loading
        pin_memory=True,
        drop_last=True             # ✅ FIX: Avoids BatchNorm error on last batch
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=2,              # 📉 Reduced batch size
        num_workers=4,
        pin_memory=True
    )

    # 🔥 Pretrained model
    model = models.segmentation.deeplabv3_resnet50(
        weights=DeepLabV3_ResNet50_Weights.DEFAULT
    )
    model.classifier[4] = nn.Conv2d(256, 10, kernel_size=1)

    model.to(DEVICE)

    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4) # Start with 1e-4
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=2)
    loss_fn = nn.CrossEntropyLoss()
    scaler = torch.amp.GradScaler('cuda') # ✅ Added GradScaler for AMP (Mixed Precision)

    # Initialize tracking variables for IoU monitoring
    running_iou = 0.0
    successful_epochs = 0

    for epoch in range(30):   # Target: 0.45+ IoU
        model.train()
        loop = tqdm(train_loader)

        for imgs, masks in loop:
            imgs = imgs.to(DEVICE, non_blocking=True)
            masks = masks.to(DEVICE, non_blocking=True).long() # ensure long for cross entropy

            # ✅ Mixed Precision Training (saves memory)
            with torch.amp.autocast('cuda'):
                outputs = model(imgs)['out']
                loss = loss_fn(outputs, masks)

            optimizer.zero_grad()
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            loop.set_postfix(loss=loss.item())

        # ✅ VALIDATION
        model.eval()
        ious = []

        with torch.no_grad():
            for imgs, masks in val_loader:
                imgs = imgs.to(DEVICE, non_blocking=True)
                masks = masks.to(DEVICE, non_blocking=True)

                outputs = model(imgs)['out']
                iou = compute_iou(outputs, masks).item() # ✅ Keep it on CPU
                ious.append(iou)

        avg_iou = sum(ious) / len(ious)

        # Update running average IoU only for successful epochs (IoU > 0)
        if avg_iou > 0:
            running_iou = (running_iou * successful_epochs + avg_iou) / (successful_epochs + 1)
            successful_epochs += 1

        print(f"\nEpoch {epoch} done | IoU: {avg_iou:.4f} | Running Avg IoU: {running_iou:.4f} | LR: {optimizer.param_groups[0]['lr']:.6f}")

        # ✅ Adjust learning rate based on performance
        scheduler.step(avg_iou)

        # ✅ SAVE MODEL EVERY EPOCH
        torch.save(model.state_dict(), "model.pth")
        print(f"Model saved to model.pth (Epoch {epoch})")

    print(f"\nTraining complete! Final IoU: {avg_iou:.4f} | Running Average IoU (across {successful_epochs} epochs): {running_iou:.4f}")


if __name__ == "__main__":
    main()