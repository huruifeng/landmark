"""
Dataset and data loading utilities for landmark image retrieval.
"""

import os
import pandas as pd
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms


class LandmarkDataset(Dataset):
    """Dataset for loading landmark images with labels."""

    def __init__(self, csv_path, image_dir, transform=None):
        self.df = pd.read_csv(csv_path)
        self.image_dir = image_dir
        self.transform = transform

        # Build label mapping: landmark_id -> contiguous index
        unique_labels = sorted(self.df["landmark_id"].unique())
        self.label_to_idx = {lbl: idx for idx, lbl in enumerate(unique_labels)}
        self.idx_to_label = {idx: lbl for lbl, idx in self.label_to_idx.items()}
        self.num_classes = len(unique_labels)

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        img_path = os.path.join(self.image_dir, row["filename"])
        image = Image.open(img_path).convert("RGB")
        label = self.label_to_idx[row["landmark_id"]]

        if self.transform:
            image = self.transform(image)

        return image, label, row["filename"]


def get_train_transform(image_size=224):
    return transforms.Compose([
        transforms.RandomResizedCrop(image_size, scale=(0.7, 1.0)),
        transforms.RandomHorizontalFlip(),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225]),
    ])


def get_val_transform(image_size=224):
    return transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(image_size),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225]),
    ])


def create_data_loaders(data_dir, batch_size=32, num_workers=4, image_size=224):
    """Create train and validation data loaders."""
    image_dir = os.path.join(data_dir, "images")
    train_csv = os.path.join(data_dir, "train.csv")
    val_csv = os.path.join(data_dir, "val.csv")

    train_dataset = LandmarkDataset(train_csv, image_dir, get_train_transform(image_size))
    val_dataset = LandmarkDataset(val_csv, image_dir, get_val_transform(image_size))

    # Ensure val uses the same label mapping as train
    val_dataset.label_to_idx = train_dataset.label_to_idx
    val_dataset.idx_to_label = train_dataset.idx_to_label
    val_dataset.num_classes = train_dataset.num_classes

    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True,
        num_workers=num_workers, pin_memory=True, drop_last=True,
    )
    val_loader = DataLoader(
        val_dataset, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=True,
    )

    return train_loader, val_loader, train_dataset.num_classes
