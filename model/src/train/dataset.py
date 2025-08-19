"""
dataset.py

This module defines the UAPDataset for loading and processing Mars rover images
and their associated annotation files. We assume the data is organized by split:

Each annotation file (in the "annotations" folder) is in YOLO format:
  "class x_center y_center width height" (all values normalized to [0,1])
"""

import os
import glob
import cv2
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import config

class UAPDataset(Dataset):
    def __init__(self, split='train', img_size=config.IMG_SIZE, transform=None):
        """
        Args:
            split (str): 'train', 'eval', or 'test' to specify the data split.
            img_size (int): Target size to resize images (img_size x img_size).
            transform (callable, optional): Optional transform to apply on a sample.
        """
        self.split = split
        self.img_size = img_size
        self.transform = transform
        # Root directory for the chosen split
        self.root_dir = os.path.join(config.DATA_DIR, split)

        # DEBUG: show which directory we're looking at
        print(f"[DEBUG] UAPDataset split={split}, root_dir={self.root_dir}")

        # Load image paths from the "img" subfolder
        img_dir = os.path.join(self.root_dir, "img")
        self.image_paths = sorted(glob.glob(os.path.join(img_dir, "*.jpg")))
        print(f"[DEBUG] Found {len(self.image_paths)} image files in {img_dir}")

        # Load corresponding annotation paths from the "annotations" subfolder
        ann_dir = os.path.join(self.root_dir, "annotations")
        self.label_paths = sorted(glob.glob(os.path.join(ann_dir, "*.txt")))
        print(f"[DEBUG] Found {len(self.label_paths)} label files in {ann_dir}")

        assert len(self.image_paths) == len(self.label_paths), \
            f"Mismatch between number of images ({len(self.image_paths)}) and label files ({len(self.label_paths)})."

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        # Load image
        img_path = self.image_paths[idx]
        image = cv2.imread(img_path)
        if image is None:
            raise ValueError(f"Image not found: {img_path}")
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = cv2.resize(image, (self.img_size, self.img_size))

        # Load annotations
        label_path = self.label_paths[idx]
        boxes = []
        classes = []
        with open(label_path, 'r') as f:
            for line in f.readlines():
                parts = line.strip().split()
                if len(parts) == 5:
                    cls, x_center, y_center, width, height = map(float, parts)
                    # Convert normalized coordinates to absolute pixel values
                    x1 = (x_center - width / 2) * self.img_size
                    y1 = (y_center - height / 2) * self.img_size
                    x2 = (x_center + width / 2) * self.img_size
                    y2 = (y_center + height / 2) * self.img_size
                    boxes.append([x1, y1, x2, y2])
                    classes.append(int(cls))

        boxes = torch.tensor(boxes, dtype=torch.float32)
        classes = torch.tensor(classes, dtype=torch.int64)

        sample = {'image': image, 'boxes': boxes, 'labels': classes}

        if self.transform:
            sample = self.transform(sample)
        else:
            # Default: convert image to tensor and scale to [0,1]
            sample['image'] = transforms.ToTensor()(sample['image'])

        return sample

def get_dataloader(split='train', img_size=config.IMG_SIZE, batch_size=config.BATCH_SIZE):
    """
    Returns a DataLoader for the specified data split.
    """
    dataset = UAPDataset(split=split, img_size=img_size)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=config.NUM_WORKERS)
    return loader

if __name__ == "__main__":
    # Quick test to ensure dataset loading works correctly.
    train_loader = get_dataloader(split='train')
    for batch in train_loader:
        images = batch['image']
        boxes = batch['boxes']
        labels = batch['labels']
        print("Image batch shape:", images.shape)
        print("Boxes shape:", boxes.shape)
        print("Labels shape:", labels.shape)
        break
