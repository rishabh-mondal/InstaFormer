import os
import torch
from torch.utils.data import Dataset
from torchvision import transforms
from PIL import Image
import random
import numpy as np

class INIT_Dataset(Dataset):
    """
    Dataset class for unpaired images with Oriented Bounding Boxes (OBB).
    Supports 8-point OBB format: [class_id, x1, y1, x2, y2, x3, y3, x4, y4]
    Normalized coordinates (0-1).
    """
    def __init__(self, data_cfg, train_mode=True):
        self.data_cfg = data_cfg
        self.train_mode = train_mode

        self.dir_A = data_cfg.dir_A  # Path for Dataset A
        self.dir_B = data_cfg.dir_B  # Path for Dataset B
        self.label_dir_A = data_cfg.label_dir_A  # Path for OBB labels (A)
        self.label_dir_B = data_cfg.label_dir_B  # Path for OBB labels (B)

        self.image_paths_A = sorted([os.path.join(self.dir_A, f) for f in os.listdir(self.dir_A) if f.endswith(".tif")])
        self.image_paths_B = sorted([os.path.join(self.dir_B, f) for f in os.listdir(self.dir_B) if f.endswith(".tif")])

        self.label_paths_A = sorted([os.path.join(self.label_dir_A, f) for f in os.listdir(self.label_dir_A) if f.endswith(".txt")])
        self.label_paths_B = sorted([os.path.join(self.label_dir_B, f) for f in os.listdir(self.label_dir_B) if f.endswith(".txt")])

        self.transform = transforms.Compose([
            transforms.Resize((data_cfg.height, data_cfg.width)),  # Resize images
            transforms.ToTensor()  # Convert to PyTorch tensors
        ])

        self.dataset_size = min(len(self.image_paths_A), len(self.image_paths_B), data_cfg.max_dataset_size)

    def __len__(self):
        return self.dataset_size

    def load_obb_labels(self, label_path):
        """
        Loads Oriented Bounding Box (OBB) labels from a .txt file.
        Format: class_id, x1, y1, x2, y2, x3, y3, x4, y4
        """
        if not os.path.exists(label_path):
            return torch.zeros((0, 9), dtype=torch.float32)  # Empty tensor if no boxes

        try:
            boxes = np.loadtxt(label_path).reshape(-1, 9)  # Ensure correct shape
            return torch.tensor(boxes, dtype=torch.float32)
        except Exception as e:
            print(f"⚠️ Error loading OBB labels from {label_path}: {e}")
            return torch.zeros((0, 9), dtype=torch.float32)

    def __getitem__(self, idx):
        # Select random images from A and B
        img_A_path = random.choice(self.image_paths_A)
        img_B_path = random.choice(self.image_paths_B)

        img_A = Image.open(img_A_path).convert("RGB")
        img_B = Image.open(img_B_path).convert("RGB")

        img_A = self.transform(img_A)
        img_B = self.transform(img_B)

        # Load OBB labels
        label_A_path = img_A_path.replace(".tif", ".txt").replace(self.dir_A, self.label_dir_A)
        label_B_path = img_B_path.replace(".tif", ".txt").replace(self.dir_B, self.label_dir_B)

        A_box = self.load_obb_labels(label_A_path)
        B_box = self.load_obb_labels(label_B_path)

        return {
            "A": img_A,
            "B": img_B,
            "A_box": A_box,  # OBB labels for A
            "B_box": B_box   # OBB labels for B
        }
