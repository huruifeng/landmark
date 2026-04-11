"""
Training script for the landmark retrieval model.

Usage:
    python src/train.py --data_dir data/gldv2_micro --epochs 10 --batch_size 32
"""

import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'  # Quick fix for potential OpenMP issues on some platforms

import argparse
import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR

from dataset import create_data_loaders
from model import LandmarkRetrievalModel


data_dir = "../data/gldv2_micro"
plot = True
save_dir = "checkpoints"

epochs = 10
batch_size = 32
lr = 1e-4

embedding_dim = 512
image_size = 224
num_workers = 4
patience = 5


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Data
train_loader, val_loader, num_classes = create_data_loaders(
    data_dir, batch_size, num_workers, image_size
)
print(f"Classes: {num_classes}, Train: {len(train_loader.dataset)}, Val: {len(val_loader.dataset)}")

# Model — device is passed in and managed internally
model = LandmarkRetrievalModel(num_classes, embedding_dim, device=device)

# Optimizer: lower lr for pretrained backbone, higher for new layers
backbone_params = list(model.backbone.parameters())
new_params = list(model.embedding.parameters()) + list(model.arcface.parameters())
optimizer = AdamW([
    {"params": backbone_params, "lr": lr * 0.1},
    {"params": new_params, "lr": lr},
], weight_decay=1e-4)

scheduler = CosineAnnealingLR(optimizer, T_max=epochs)
criterion = nn.CrossEntropyLoss()

# Train — returns history DataFrame
history = model.fit(
    train_loader=train_loader,
    optimizer=optimizer,
    scheduler=scheduler,
    criterion=criterion,
    epochs=epochs,
    save_dir=save_dir,
    val_loader=val_loader,
    patience=patience,
    plot=plot,
)

history_csv = os.path.join(save_dir, "train_history.csv")
history.to_csv(history_csv, index=False)
print(f"History saved to {history_csv}")


