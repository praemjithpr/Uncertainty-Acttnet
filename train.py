"""
Training & Uncertainty Optimization Pipeline for Hybrid Quantum Uncertainty Network
Author: Praemjith P R
"""

import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.cuda.amp import GradScaler, autocast
from model import HybridQuantumUncertaintyNet

class CombinedDiceCrossEntropyLoss(nn.Module):
    """
    Combined Soft Dice and Cross-Entropy loss for robust multi-class segmentation.
    """
    def __init__(self, smooth: float = 1.0):
        super(CombinedDiceCrossEntropyLoss, self).__init__()
        self.smooth = smooth
        self.ce = nn.CrossEntropyLoss()

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        ce_loss = self.ce(logits, targets)
        
        num_classes = logits.shape[1]
        probs = torch.softmax(logits, dim=1)
        targets_one_hot = torch.zeros_like(logits).scatter_(1, targets.unsqueeze(1), 1)

        dice_total = 0.0
        for c in range(num_classes):
            p = probs[:, c]
            t = targets_one_hot[:, c]
            intersection = (p * t).sum()
            dice = (2. * intersection + self.smooth) / (p.sum() + t.sum() + self.smooth)
            dice_total += (1.0 - dice)

        dice_loss = dice_total / num_classes
        return 0.5 * ce_loss + 0.5 * dice_loss

def train_epoch(model, dataloader, criterion, optimizer, scaler, device):
    model.train()
    running_loss = 0.0

    for batch_idx, (images, masks) in enumerate(dataloader):
        images, masks = images.to(device), masks.to(device)

        optimizer.zero_grad()
        with autocast():
            logits = model(images)
            loss = criterion(logits, masks)

        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        running_loss += loss.item()

    return running_loss / len(dataloader)

if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Initializing HybridQuantumUncertaintyNet on device: {device}")

    model = HybridQuantumUncertaintyNet(in_channels=3, num_classes=4).to(device)
    dummy_input = torch.randn(2, 3, 256, 256).to(device)
    
    print("Testing forward pass...")
    output = model(dummy_input)
    print(f"Output shape: {output.shape}")

    print("Testing Monte Carlo uncertainty estimation (10 samples)...")
    mean_pred, uncertainty_map = model.predict_with_uncertainty(dummy_input, num_samples=10)
    print(f"Mean prediction shape: {mean_pred.shape}")
    print(f"Uncertainty variance map shape: {uncertainty_map.shape}")
    print("Model verification successful!")
