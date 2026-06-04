import os
import torch
import cv2
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import numpy as np
from torchvision import models, transforms

# ── Config ──────────────────────────────────────────────────────────────────
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
TRIAL_PATH = "../data/test/images/Trial2"
# TRIAL_PATH = "../data/val/images"   ← swap here if needed

# ── Colour palette that matches the ground-truth style in your screenshot ───
#    10 classes → 10 distinct colours (tab10 order, explicit so it's stable)
PALETTE = np.array([
    [54,  162, 235],   # class 0 – sky / blue
    [255, 127,  14],   # class 1 – ground / orange
    [ 44, 160,  44],   # class 2 – vegetation / green
    [214,  39,  40],   # class 3 – red
    [148, 103, 189],   # class 4 – purple
    [140,  86,  75],   # class 5 – brown
    [227, 119, 194],   # class 6 – pink
    [127, 127, 127],   # class 7 – grey
    [188, 189,  34],   # class 8 – olive
    [ 23, 190, 207],   # class 9 – cyan
], dtype=np.uint8)


def mask_to_color(pred_mask: np.ndarray) -> np.ndarray:
    """Convert a (H, W) class-index mask → (H, W, 3) RGB colour image."""
    h, w = pred_mask.shape
    color_mask = np.zeros((h, w, 3), dtype=np.uint8)
    for cls_idx, color in enumerate(PALETTE):
        color_mask[pred_mask == cls_idx] = color
    return color_mask


def predict():
    # 1. Build model
    model = models.segmentation.deeplabv3_resnet50(weights=None)
    model.classifier[4] = torch.nn.Conv2d(256, 10, kernel_size=1)

    # 2. Load weights
    if not os.path.exists("model.pth"):
        print("✗  model.pth not found – please train the model first.")
        return
    model.load_state_dict(torch.load("model.pth", map_location=DEVICE), strict=False)
    model.to(DEVICE).eval()

    # 3. Normalisation transform
    normalize = transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std =[0.229, 0.224, 0.225],
    )

    # 4. Collect images (cap at 20 to keep the figure manageable)
    images = sorted(
        f for f in os.listdir(TRIAL_PATH)
        if f.lower().endswith(('.png', '.jpg', '.jpeg'))
    )[:20]

    if not images:
        print(f"✗  No images found in {TRIAL_PATH}")
        return

    print(f"✓  Found {len(images)} image(s). Running inference on {DEVICE} …")

    # ── Figure layout: 2 columns (Original | Prediction) ─────────────────
    fig, axes = plt.subplots(
        nrows=len(images), ncols=2,
        figsize=(10, 5 * len(images)),
        squeeze=False,
    )

    for i, img_name in enumerate(images):
        img_path = os.path.join(TRIAL_PATH, img_name)

        # Load & resize
        bgr = cv2.imread(img_path)
        if bgr is None:
            print(f"  ⚠  Could not read {img_path}, skipping.")
            continue
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        resized = cv2.resize(rgb, (512, 512))

        # Preprocess → tensor
        t = torch.from_numpy(resized).permute(2, 0, 1).float() / 255.0
        t = normalize(t).unsqueeze(0).to(DEVICE)

        # Inference
        with torch.no_grad():
            out   = model(t)['out']
            preds = torch.argmax(out, dim=1).squeeze(0).cpu().numpy()

        # Convert mask to colour image (same palette as ground-truth masks)
        color_pred = mask_to_color(preds)

        # ── Left column: original ─────────────────────────────────────────
        axes[i][0].imshow(resized)
        axes[i][0].set_title(f"Original: {img_name}", fontsize=11)
        axes[i][0].axis('off')

        # ── Right column: coloured prediction ────────────────────────────
        axes[i][1].imshow(color_pred)
        axes[i][1].set_title("Prediction", fontsize=11)
        axes[i][1].axis('off')

    plt.tight_layout()
    out_path = "trial_results.png"
    plt.savefig(out_path, dpi=150, bbox_inches='tight')
    print(f"✓  Results saved → {out_path}")
    plt.show()


if __name__ == "__main__":
    predict()