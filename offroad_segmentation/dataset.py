import os
import cv2
import torch
import random
from torch.utils.data import Dataset
from torchvision import transforms

class SegmentationDataset(Dataset):
    def __init__(self, image_dir, mask_dir, split="train", limit=None):
        self.image_dir = image_dir
        self.mask_dir = mask_dir
        self.split = split

        self.images = [f for f in os.listdir(image_dir)
                       if f.lower().endswith(('.png', '.jpg', '.jpeg'))]

        if limit is not None:
            self.images = self.images[:limit]

        # ✅ ImageNet standard normalization (MANDATORY for 0.8+ IoU)
        self.normalize = transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )

        # ✨ NEW: Robust Color Augmentation (helps with Real-World textures)
        self.color_aug = transforms.Compose([
            transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.2, hue=0.1),
            transforms.RandomGrayscale(p=0.1),
            # Optional: Add small blur to handle low-quality real-world images
            transforms.GaussianBlur(kernel_size=(3, 3), sigma=(0.1, 2.0))
        ])

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img_name = self.images[idx]
        img_path = os.path.join(self.image_dir, img_name)
        mask_path = os.path.join(self.mask_dir, img_name)

        # ✅ Load image
        image = cv2.imread(img_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = cv2.resize(image, (512, 512), interpolation=cv2.INTER_LINEAR)

        # ✅ Load mask
        mask = cv2.imread(mask_path, 0)
        mask = cv2.resize(mask, (512, 512), interpolation=cv2.INTER_NEAREST)

        # 🔥 AUGMENTATION (Avoids overfitting, drives IoU to 0.8)
        if self.split == "train":
            # Random Horizontal Flip
            if random.random() > 0.5:
                image = cv2.flip(image, 1)
                mask = cv2.flip(mask, 1)

        # Convert to tensor & scale to [0, 1]
        image = torch.tensor(image).permute(2, 0, 1).float() / 255.0
        
        # ✨ NEW: Apply color augmentation only during training
        if self.split == "train":
            image = self.color_aug(image)

        # Normalize image (IMPORTANT: must happen after /255.0 and before training)
        image = self.normalize(image)

        # Convert mask to torch tensor and clamp classes
        mask = torch.tensor(mask).long()
        mask = torch.clamp(mask, 0, 9)

        return image, mask