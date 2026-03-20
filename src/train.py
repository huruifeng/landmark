"""
Training script for the landmark retrieval model.

Usage:
    python src/train.py --data_dir data/gldv2_micro --epochs 10 --batch_size 32
"""

import os
import argparse
import time
import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR

from dataset import create_data_loaders
from model import LandmarkRetrievalModel


def train_one_epoch(model, loader, criterion, optimizer, device):
    model.train()
    total_loss = 0.0
    correct = 0
    total = 0

    for images, labels, _ in loader:
        images, labels = images.to(device), labels.to(device)

        logits = model(images, labels)
        loss = criterion(logits, labels)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item() * images.size(0)
        preds = logits.argmax(dim=1)
        correct += (preds == labels).sum().item()
        total += images.size(0)

    return total_loss / total, correct / total


@torch.no_grad()
def validate(model, loader, criterion, device):
    model.eval()
    total_loss = 0.0
    correct = 0
    total = 0

    for images, labels, _ in loader:
        images, labels = images.to(device), labels.to(device)

        logits = model(images, labels)
        loss = criterion(logits, labels)

        total_loss += loss.item() * images.size(0)
        preds = logits.argmax(dim=1)
        correct += (preds == labels).sum().item()
        total += images.size(0)

    return total_loss / total, correct / total


def main():
    parser = argparse.ArgumentParser(description="Train landmark retrieval model")
    parser.add_argument("--data_dir", type=str, default="data/gldv2_micro")
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--embedding_dim", type=int, default=512)
    parser.add_argument("--image_size", type=int, default=224)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--save_dir", type=str, default="checkpoints")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Data
    train_loader, val_loader, num_classes = create_data_loaders(
        args.data_dir, args.batch_size, args.num_workers, args.image_size
    )
    print(f"Classes: {num_classes}, Train: {len(train_loader.dataset)}, Val: {len(val_loader.dataset)}")

    # Model
    model = LandmarkRetrievalModel(num_classes, args.embedding_dim).to(device)

    # Optimizer: lower lr for pretrained backbone, higher for new layers
    backbone_params = list(model.backbone.parameters())
    new_params = list(model.embedding.parameters()) + list(model.arcface.parameters())
    optimizer = AdamW([
        {"params": backbone_params, "lr": args.lr * 0.1},
        {"params": new_params, "lr": args.lr},
    ], weight_decay=1e-4)

    scheduler = CosineAnnealingLR(optimizer, T_max=args.epochs)
    criterion = nn.CrossEntropyLoss()

    os.makedirs(args.save_dir, exist_ok=True)
    best_val_acc = 0.0

    for epoch in range(1, args.epochs + 1):
        t0 = time.time()

        train_loss, train_acc = train_one_epoch(model, train_loader, criterion, optimizer, device)
        val_loss, val_acc = validate(model, val_loader, criterion, device)
        scheduler.step()

        elapsed = time.time() - t0
        print(
            f"Epoch {epoch}/{args.epochs} ({elapsed:.0f}s) | "
            f"Train Loss: {train_loss:.4f} Acc: {train_acc:.4f} | "
            f"Val Loss: {val_loss:.4f} Acc: {val_acc:.4f}"
        )

        # Save best model
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save({
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "num_classes": num_classes,
                "embedding_dim": args.embedding_dim,
                "val_acc": val_acc,
            }, os.path.join(args.save_dir, "best_model.pth"))
            print(f"  -> Saved best model (val_acc={val_acc:.4f})")

    # Save final model
    torch.save({
        "epoch": args.epochs,
        "model_state_dict": model.state_dict(),
        "num_classes": num_classes,
        "embedding_dim": args.embedding_dim,
        "val_acc": val_acc,
    }, os.path.join(args.save_dir, "final_model.pth"))
    print(f"Training complete. Best val accuracy: {best_val_acc:.4f}")


if __name__ == "__main__":
    main()
