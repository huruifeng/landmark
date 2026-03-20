"""
Landmark retrieval model with ArcFace head for metric learning.
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models


class ArcFaceHead(nn.Module):
    """ArcFace classification head for metric learning.

    Produces angular-margin softmax logits that encourage the model
    to learn discriminative, well-separated embeddings.
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

    def __init__(self, num_classes, embedding_dim=512):
        super().__init__()
        self.embedding_dim = embedding_dim

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
            logits = self.arcface(emb, labels)
            return logits
        return emb
