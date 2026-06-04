import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import models
from torchvision.models.segmentation import DeepLabV3_ResNet50_Weights
from dataset import SegmentationDataset
from tqdm import tqdm
from utils import compute_iou
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

    # Load datasets
    limit = config["training"].get("limit_dataset", None)
    train_dataset = SegmentationDataset(
        config["data"]["train"]["images"], 
        config["data"]["train"]["masks"], 
        split="train",
        limit=limit
    )
    val_dataset = SegmentationDataset(
        config["data"]["val"]["images"], 
        config["data"]["val"]["masks"], 
        split="val",
        limit=limit
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=config["training"]["batch_size"],
        shuffle=True,
        num_workers=config["dataloader"]["num_workers"],
        pin_memory=config["dataloader"]["pin_memory"],
        persistent_workers=config["dataloader"]["persistent_workers"],
        drop_last=config["dataloader"]["drop_last"]
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=config["training"]["batch_size"],
        num_workers=config["dataloader"]["num_workers"],
        pin_memory=config["dataloader"]["pin_memory"],
        persistent_workers=config["dataloader"]["persistent_workers"]
    )

    # Load model
    model = models.segmentation.deeplabv3_resnet50(
        weights=DeepLabV3_ResNet50_Weights.DEFAULT if config["model"]["pretrained"] else None
    )
    model.classifier[4] = nn.Conv2d(256, config["training"]["num_classes"], kernel_size=1)
    model.to(DEVICE)

    # Optimizer and scheduler
    if config["optimizer"]["type"] == "Adam":
        optimizer = torch.optim.Adam(model.parameters(), lr=config["training"]["learning_rate"])
    
    if config["optimizer"]["lr_scheduler"] == "ReduceLROnPlateau":
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, 
            mode='max', 
            factor=config["optimizer"]["scheduler_factor"], 
            patience=config["optimizer"]["scheduler_patience"]
        )

    # Loss function
    if config["loss"]["type"] == "CrossEntropyLoss":
        loss_fn = nn.CrossEntropyLoss()
    
    # Only use GradScaler if CUDA is available
    scaler = None
    if DEVICE == "cuda":
        scaler = torch.amp.GradScaler('cuda')

    # Initialize tracking variables
    running_iou = 0.0
    successful_epochs = 0

    for epoch in range(config["training"]["epochs"]):
        model.train()
        loop = tqdm(train_loader)

        for imgs, masks in loop:
            imgs = imgs.to(DEVICE, non_blocking=True)
            masks = masks.to(DEVICE, non_blocking=True).long()

            if DEVICE == "cuda":
                with torch.amp.autocast('cuda'):
                    outputs = model(imgs)['out']
                    loss = loss_fn(outputs, masks)

                optimizer.zero_grad()
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
            else:
                outputs = model(imgs)['out']
                loss = loss_fn(outputs, masks)

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            loop.set_postfix(loss=loss.item())

        # Validation
        model.eval()
        ious = []

        with torch.no_grad():
            for imgs, masks in val_loader:
                imgs = imgs.to(DEVICE, non_blocking=True)
                masks = masks.to(DEVICE, non_blocking=True)

                outputs = model(imgs)['out']
                iou = compute_iou(outputs, masks, num_classes=config["training"]["num_classes"]).item()
                ious.append(iou)

        avg_iou = sum(ious) / len(ious)

        if avg_iou > 0:
            running_iou = (running_iou * successful_epochs + avg_iou) / (successful_epochs + 1)
            successful_epochs += 1

        print(f"\nEpoch {epoch} done | IoU: {avg_iou:.4f} | Running Avg IoU: {running_iou:.4f} | LR: {optimizer.param_groups[0]['lr']:.6f}")

        scheduler.step(avg_iou)

        torch.save(model.state_dict(), "model.pth")
        print(f"Model saved to model.pth (Epoch {epoch})")

    print(f"\nTraining complete! Final IoU: {avg_iou:.4f} | Running Average IoU (across {successful_epochs} epochs): {running_iou:.4f}")


if __name__ == "__main__":
    main()