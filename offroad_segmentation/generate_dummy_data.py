import numpy as np
import cv2
import os

def create_dummy_data(folder, num_samples=5):
    img_dir = os.path.join(folder, "images")
    mask_dir = os.path.join(folder, "masks")
    
    os.makedirs(img_dir, exist_ok=True)
    os.makedirs(mask_dir, exist_ok=True)
    
    for i in range(num_samples):
        # Create a random image
        img = np.random.randint(0, 255, (512, 512, 3), dtype=np.uint8)
        cv2.imwrite(os.path.join(img_dir, f"sample_{i}.png"), img)
        
        # Create a dummy mask (just some random classes 0-9)
        mask = np.random.randint(0, 10, (512, 512), dtype=np.uint8)
        cv2.imwrite(os.path.join(mask_dir, f"sample_{i}.png"), mask)

if __name__ == "__main__":
    for split in ["train", "val"]:
        create_dummy_data(os.path.join("data", split))
    print("Done creating dummy data.")
