"""
Hybrid Quantum Uncertainty Network for High-Reliability Visual Segmentation
Model Architecture: Multi-Scale Attention UNet with Parameterized Quantum Layer & Monte Carlo Dropout
Author: Praemjith P R
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F

class MonteCarloDropout(nn.Dropout2d):
    """
    Monte Carlo Dropout layer that remains active during inference
    to quantify epistemic uncertainty via stochastic forward passes.
    """
    def forward(self, input_tensor: torch.Tensor) -> torch.Tensor:
        return F.dropout2d(input_tensor, self.p, training=True, inplace=self.inplace)

class QuantumFeatureEnhancement(nn.Module):
    """
    Simulated Parameterized Quantum Circuit (PQC) Layer for high-order feature mapping
    using trigonometric phase encoding.
    """
    def __init__(self, in_features: int, out_features: int):
        super(QuantumFeatureEnhancement, self).__init__()
        self.weights = nn.Parameter(torch.randn(out_features, in_features) * 0.05)
        self.phase_shift = nn.Parameter(torch.zeros(out_features))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Phase encoding projection along channels
        batch, channels, height, width = x.shape
        flat_x = x.permute(0, 2, 3, 1) # (B, H, W, C)
        projected = F.linear(flat_x, self.weights, self.phase_shift)
        quantum_state = torch.cos(projected) + torch.sin(projected)
        return quantum_state.permute(0, 3, 1, 2)

class AttentionGate(nn.Module):
    """
    Entropy-Gated Attention Mechanism for selective feature calibration.
    """
    def __init__(self, F_g: int, F_l: int, F_int: int):
        super(AttentionGate, self).__init__()
        self.W_g = nn.Sequential(
            nn.Conv2d(F_g, F_int, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm2d(F_int)
        )
        self.W_l = nn.Sequential(
            nn.Conv2d(F_l, F_int, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm2d(F_int)
        )
        self.psi = nn.Sequential(
            nn.Conv2d(F_int, 1, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm2d(1),
            nn.Sigmoid()
        )
        self.relu = nn.ReLU(inplace=True)

    def forward(self, g: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
        g1 = self.W_g(g)
        x1 = self.W_l(x)
        psi = self.relu(g1 + x1)
        psi = self.psi(psi)
        return x * psi

class ConvBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, dropout_rate: float = 0.2):
        super(ConvBlock, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            MonteCarloDropout(p=dropout_rate),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv(x)

class HybridQuantumUncertaintyNet(nn.Module):
    """
    Hybrid Quantum Uncertainty Network for High-Reliability Visual Segmentation.
    """
    def __init__(self, in_channels: int = 3, num_classes: int = 4, dropout_rate: float = 0.2):
        super(HybridQuantumUncertaintyNet, self).__init__()
        
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

        # Encoder
        self.enc1 = ConvBlock(in_channels, 64, dropout_rate)
        self.enc2 = ConvBlock(64, 128, dropout_rate)
        self.enc3 = ConvBlock(128, 256, dropout_rate)
        self.enc4 = ConvBlock(256, 512, dropout_rate)

        # Bottleneck with Quantum Layer
        self.bottleneck_conv = ConvBlock(512, 1024, dropout_rate)
        self.quantum_layer = QuantumFeatureEnhancement(1024, 1024)

        # Decoder
        self.up4 = nn.ConvTranspose2d(1024, 512, kernel_size=2, stride=2)
        self.att4 = AttentionGate(F_g=512, F_l=512, F_int=256)
        self.dec4 = ConvBlock(1024, 512, dropout_rate)

        self.up3 = nn.ConvTranspose2d(512, 256, kernel_size=2, stride=2)
        self.att3 = AttentionGate(F_g=256, F_l=256, F_int=128)
        self.dec3 = ConvBlock(512, 256, dropout_rate)

        self.up2 = nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2)
        self.att2 = AttentionGate(F_g=128, F_l=128, F_int=64)
        self.dec2 = ConvBlock(256, 128, dropout_rate)

        self.up1 = nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2)
        self.att1 = AttentionGate(F_g=64, F_l=64, F_int=32)
        self.dec1 = ConvBlock(128, 64, dropout_rate)

        self.final_conv = nn.Conv2d(64, num_classes, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        e1 = self.enc1(x)
        e2 = self.enc2(self.pool(e1))
        e3 = self.enc3(self.pool(e2))
        e4 = self.enc4(self.pool(e3))

        b = self.bottleneck_conv(self.pool(e4))
        b = self.quantum_layer(b)

        d4 = self.up4(b)
        x4 = self.att4(g=d4, x=e4)
        d4 = self.dec4(torch.cat((x4, d4), dim=1))

        d3 = self.up3(d4)
        x3 = self.att3(g=d3, x=e3)
        d3 = self.dec3(torch.cat((x3, d3), dim=1))

        d2 = self.up2(d3)
        x2 = self.att2(g=d2, x=e2)
        d2 = self.dec2(torch.cat((x2, d2), dim=1))

        d1 = self.up1(d2)
        x1 = self.att1(g=d1, x=e1)
        d1 = self.dec1(torch.cat((x1, d1), dim=1))

        out = self.final_conv(d1)
        return out

    def predict_with_uncertainty(self, x: torch.Tensor, num_samples: int = 10):
        """
        Executes Monte Carlo forward passes to calculate mean prediction and predictive variance (epistemic uncertainty).
        """
        self.train() # Keep dropout active
        predictions = []
        with torch.no_grad():
            for _ in range(num_samples):
                logits = self.forward(x)
                probs = F.softmax(logits, dim=1)
                predictions.append(probs)

        # Stack over sample dimension: (N, B, C, H, W)
        pred_stack = torch.stack(predictions, dim=0)
        mean_prediction = torch.mean(pred_stack, dim=0)
        uncertainty_map = torch.var(pred_stack, dim=0).mean(dim=1, keepdim=True)

        return mean_prediction, uncertainty_map
