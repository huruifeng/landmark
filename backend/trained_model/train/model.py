"""
Landmark retrieval model with ArcFace head for metric learning.
"""

import math
import os
import time
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models


class ArcFaceHead(nn.Module):
    """ArcFace classification head for metric learning.

    Produces angular-margin softmax logits that encourage the model
    to learn discriminative, well-separated embeddings.

    Args:
    - embedding_dim (int): Dimensionality of input embeddings.
    - num_classes (int): Number of classes for classification.
    - s (float): Scaling factor for logits (default: 30.0).
    - m (float): Angular margin in radians (default: 0.3).
    

    """

    def __init__(self, embedding_dim, num_classes, s=30.0, m=0.3):
        super().__init__()
        self.s = s
        self.m = m
        self.weight = nn.Parameter(torch.FloatTensor(num_classes, embedding_dim))
        nn.init.xavier_uniform_(self.weight)

    def forward(self, embeddings, labels):
        # Normalize weights and embeddings
        w = F.normalize(self.weight, dim=1)
        x = F.normalize(embeddings, dim=1)

        # Cosine similarity
        cosine = F.linear(x, w)
        cosine = cosine.clamp(-1.0 + 1e-7, 1.0 - 1e-7)

        # ArcFace margin
        theta = torch.acos(cosine)
        target_logits = torch.cos(theta + self.m)

        # One-hot encode labels and apply margin only to target class
        one_hot = torch.zeros_like(cosine)
        one_hot.scatter_(1, labels.unsqueeze(1), 1.0)
        logits = one_hot * target_logits + (1.0 - one_hot) * cosine

        return logits * self.s


class LandmarkRetrievalModel(nn.Module):
    """ResNet50 backbone with embedding layer and ArcFace head."""

    def __init__(self, num_classes, embedding_dim=512, device=None):
        super().__init__()
        self.embedding_dim = embedding_dim
        self.device = device if device is not None else torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        )

        # Backbone: ResNet50 pretrained
        backbone = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)
        backbone_out_dim = backbone.fc.in_features
        backbone.fc = nn.Identity()
        self.backbone = backbone

        # Embedding head
        self.embedding = nn.Sequential(
            nn.Linear(backbone_out_dim, embedding_dim),
            nn.BatchNorm1d(embedding_dim),
        )

        # ArcFace classification head (used only during training)
        self.arcface = ArcFaceHead(embedding_dim, num_classes)

        self.to(self.device)

    def extract_embedding(self, x):
        """Extract L2-normalized embedding for retrieval."""
        features = self.backbone(x)
        emb = self.embedding(features)
        return F.normalize(emb, dim=1)

    def forward(self, x, labels=None):
        """
        Forward pass.
        - During training (labels provided): returns ArcFace logits.
        - During inference (no labels): returns normalized embeddings.
        """
        emb = self.extract_embedding(x)
        if labels is not None:
            return self.arcface(emb, labels)
        return emb

    def train_one_epoch(self, loader, criterion, optimizer):
        """Run one training epoch. Returns (avg_loss, accuracy)."""
        self.train()
        total_loss = 0.0
        correct = 0
        total = 0

        for images, labels, _ in loader:
            images, labels = images.to(self.device), labels.to(self.device)

            logits = self(images, labels)
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
    def validate(self, loader, criterion):
        """Run validation. Returns (avg_loss, accuracy)."""
        self.eval()
        total_loss = 0.0
        correct = 0
        total = 0

        for images, labels, _ in loader:
            images, labels = images.to(self.device), labels.to(self.device)

            logits = self(images, labels)
            loss = criterion(logits, labels)

            total_loss += loss.item() * images.size(0)
            preds = logits.argmax(dim=1)
            correct += (preds == labels).sum().item()
            total += images.size(0)

        return total_loss / total, correct / total

    def fit(
        self,
        train_loader,
        optimizer,
        scheduler,
        criterion,
        epochs,
        save_dir,
        val_loader=None,
        patience=5,
        plot=False,
    ):
        """
        Full training loop with optional validation, early stopping, and history plotting.

        Args:
            train_loader: DataLoader for training data.
            optimizer: Optimizer instance.
            scheduler: LR scheduler instance.
            criterion: Loss function.
            epochs (int): Maximum number of epochs.
            save_dir (str): Directory to save checkpoints.
            val_loader: DataLoader for validation (enables early stopping). Optional.
            patience (int): Early-stopping patience in epochs (only used when val_loader given).
            plot (bool): If True, plot loss/accuracy curves after training.

        Returns:
            pd.DataFrame: History with columns epoch, train_loss, train_acc,
                          and val_loss, val_acc when val_loader is provided.
        """
        os.makedirs(save_dir, exist_ok=True)

        history = []
        best_val_acc = 0.0
        patience_counter = 0
        last_epoch = epochs
        num_classes = self.arcface.weight.size(0)

        for epoch in range(1, epochs + 1):
            t0 = time.time()

            train_loss, train_acc = self.train_one_epoch(train_loader, criterion, optimizer)
            row = {"epoch": epoch, "train_loss": train_loss, "train_acc": train_acc}

            val_loss, val_acc = None, None
            if val_loader is not None:
                val_loss, val_acc = self.validate(val_loader, criterion)
                row["val_loss"] = val_loss
                row["val_acc"] = val_acc

            scheduler.step()
            history.append(row)

            elapsed = time.time() - t0
            log = (
                f"Epoch {epoch}/{epochs} ({elapsed:.0f}s) | "
                f"Train Loss: {train_loss:.4f} Acc: {train_acc:.4f}"
            )
            if val_loader is not None:
                log += f" | Val Loss: {val_loss:.4f} Acc: {val_acc:.4f}"
            print(log)

            # Early stopping + best-model checkpoint
            if val_loader is not None:
                if val_acc > best_val_acc:
                    best_val_acc = val_acc
                    patience_counter = 0
                    torch.save(
                        {
                            "epoch": epoch,
                            "model_state_dict": self.state_dict(),
                            "num_classes": num_classes,
                            "embedding_dim": self.embedding_dim,
                            "val_acc": val_acc,
                        },
                        os.path.join(save_dir, "best_model.pth"),
                    )
                    print(f"  -> Saved best model (val_acc={val_acc:.4f})")
                else:
                    patience_counter += 1
                    if patience_counter >= patience:
                        print(
                            f"Early stopping after epoch {epoch} "
                            f"(no improvement for {patience} epochs)."
                        )
                        last_epoch = epoch
                        break

        # Save final model
        torch.save(
            {
                "epoch": last_epoch,
                "model_state_dict": self.state_dict(),
                "num_classes": num_classes,
                "embedding_dim": self.embedding_dim,
                "val_acc": best_val_acc if val_loader is not None else None,
            },
            os.path.join(save_dir, "final_model.pth"),
        )

        history_df = pd.DataFrame(history)

        summary = f"Training complete. Best val accuracy: {best_val_acc:.4f}" if val_loader is not None \
            else "Training complete."
        print(summary)

        if plot:
            self._plot_history(history_df)

        return history_df

    def _plot_history(self, history_df):
        """Plot training (and validation) loss and accuracy curves."""
        try:
            import matplotlib.pyplot as plt
        except ImportError:
            print("matplotlib not installed — skipping plot.")
            return

        has_val = "val_loss" in history_df.columns
        fig, axes = plt.subplots(1, 2, figsize=(12, 4))

        axes[0].plot(history_df["epoch"], history_df["train_loss"], label="Train Loss")
        if has_val:
            axes[0].plot(history_df["epoch"], history_df["val_loss"], label="Val Loss")
        axes[0].set_xlabel("Epoch")
        axes[0].set_ylabel("Loss")
        axes[0].set_title("Loss History")
        axes[0].legend()
        axes[0].grid(True)

        axes[1].plot(history_df["epoch"], history_df["train_acc"], label="Train Acc")
        if has_val:
            axes[1].plot(history_df["epoch"], history_df["val_acc"], label="Val Acc")
        axes[1].set_xlabel("Epoch")
        axes[1].set_ylabel("Accuracy")
        axes[1].set_title("Accuracy History")
        axes[1].legend()
        axes[1].grid(True)

        plt.tight_layout()
        plt.show()
