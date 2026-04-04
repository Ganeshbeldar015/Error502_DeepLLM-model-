import os
import torch
import cv2
import matplotlib.pyplot as plt
import numpy as np
from torchvision import models, transforms

# Config
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
TRIAL_PATH = "../data/test/images/Trial"
# PRO_TIP: If you want to use the previous val path, just swap it here:
# TRIAL_PATH = "../data/val/images"

def predict():
    # 1. Initialize model structure
    model = models.segmentation.deeplabv3_resnet50(weights=None)
    model.classifier[4] = torch.nn.Conv2d(256, 10, kernel_size=1)
    
    # 2. Load latest weights
    if not os.path.exists("model.pth"):
        print("✗ Error: model.pth not found. Please train the model first.")
        return
        
    model.load_state_dict(torch.load("model.pth", map_location=DEVICE), strict=False)
    model.to(DEVICE)
    model.eval()

    # 3. Preparation (Normalization)
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    
    # 4. Get images
    images = [f for f in os.listdir(TRIAL_PATH) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
    if not images:
        print(f"✗ No images found in {TRIAL_PATH}")
        return

    print(f"✓ Found {len(images)} trial images. Running inference on {DEVICE}...")

    plt.figure(figsize=(15, 5 * len(images)))

    for i, img_name in enumerate(images):
        img_path = os.path.join(TRIAL_PATH, img_name)
        
        # Load & Preprocess
        original_img = cv2.imread(img_path)
        original_img = cv2.cvtColor(original_img, cv2.COLOR_BGR2RGB)
        resized_img = cv2.resize(original_img, (512, 512))
        
        input_tensor = torch.tensor(resized_img).permute(2, 0, 1).float() / 255.0
        input_tensor = normalize(input_tensor).unsqueeze(0).to(DEVICE)

        # Predict
        with torch.no_grad():
            output = model(input_tensor)['out']
            preds = torch.argmax(output, dim=1).squeeze(0).cpu().numpy()

        # Plot Output
        plt.subplot(len(images), 2, i*2 + 1)
        plt.imshow(resized_img)
        plt.title(f"Trial: {img_name}")
        plt.axis('off')

        plt.subplot(len(images), 2, i*2 + 2)
        plt.imshow(preds, cmap='tab10')
        plt.title(f"Model Prediction")
        plt.axis('off')

    plt.tight_layout()
    plt.savefig("trial_results.png")
    print("✓ All done! Results saved as trial_results.png")
    plt.show()

if __name__ == "__main__":
    predict()
