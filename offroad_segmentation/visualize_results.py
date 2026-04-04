import torch
import cv2
import matplotlib.pyplot as plt
import numpy as np
from torchvision import models, transforms
from dataset import SegmentationDataset

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

def visualize_results(num_images=5):
    # Load dataset
    val_dataset = SegmentationDataset("../data/val/images", "../data/val/masks", split="val")
    
    # Load model architecture
    model = models.segmentation.deeplabv3_resnet50(weights=None)
    model.classifier[4] = torch.nn.Conv2d(256, 10, kernel_size=1)
    
    # Load trained weights
    # ✅ Fix: Use strict=False to ignore training-only auxiliary classifier keys
    model.load_state_dict(torch.load("model.pth", map_location=DEVICE), strict=False)
    model.to(DEVICE)
    model.eval()

    # Get random indices
    indices = np.random.choice(len(val_dataset), num_images, replace=False)

    plt.figure(figsize=(15, 5 * num_images))

    for i, idx in enumerate(indices):
        img, mask = val_dataset[idx]
        
        # Inference
        with torch.no_grad():
            input_tensor = img.unsqueeze(0).to(DEVICE)
            output = model(input_tensor)['out']
            preds = torch.argmax(output, dim=1).squeeze(0).cpu().numpy()

        # ✅ Un-normalize image for display (otherwise it will look very dark/strange)
        inv_normalize = transforms.Normalize(
            mean=[-0.485/0.229, -0.456/0.224, -0.406/0.225],
            std=[1/0.229, 1/0.224, 1/0.225]
        )
        img_display = inv_normalize(img).permute(1, 2, 0).cpu().numpy()
        img_display = np.clip(img_display, 0, 1) # Ensure values are in valid range [0, 1]
        mask_display = mask.cpu().numpy()

        # Plot Original
        plt.subplot(num_images, 3, i*3 + 1)
        plt.imshow(img_display)
        plt.title(f"Image {idx} - Original")
        plt.axis('off')

        # Plot Ground Truth
        plt.subplot(num_images, 3, i*3 + 2)
        plt.imshow(mask_display, cmap='tab10')
        plt.title(f"Ground Truth")
        plt.axis('off')

        # Plot Prediction
        plt.subplot(num_images, 3, i*3 + 3)
        plt.imshow(preds, cmap='tab10')
        plt.title(f"Prediction")
        plt.axis('off')

    plt.tight_layout()
    plt.savefig("visualize_results.png")
    print("\nVisualization results saved to visualize_results.png")
    plt.show()

if __name__ == "__main__":
    visualize_results()
