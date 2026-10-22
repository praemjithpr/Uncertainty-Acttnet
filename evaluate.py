"""
Evaluation & Calibration Metrics for Hybrid Quantum Uncertainty Network
Author: Praemjith P R
"""

import torch
import numpy as np

def compute_miou(preds: torch.Tensor, targets: torch.Tensor, num_classes: int = 4) -> float:
    """
    Computes Mean Intersection over Union (mIoU).
    """
    ious = []
    preds = torch.argmax(preds, dim=1)
    for cls in range(num_classes):
        pred_inds = (preds == cls)
        target_inds = (targets == cls)
        intersection = (pred_inds & target_inds).long().sum().item()
        union = (pred_inds | target_inds).long().sum().item()
        if union == 0:
            continue
        ious.append(intersection / union)
    return float(np.mean(ious)) if ious else 0.0

def compute_expected_calibration_error(probs: torch.Tensor, targets: torch.Tensor, n_bins: int = 15) -> float:
    """
    Computes Expected Calibration Error (ECE) to quantify uncertainty reliability.
    """
    confidences, predictions = torch.max(probs, dim=1)
    accuracies = predictions.eq(targets)

    bin_boundaries = torch.linspace(0, 1, n_bins + 1)
    ece = 0.0

    for i in range(n_bins):
        bin_lower = bin_boundaries[i]
        bin_upper = bin_boundaries[i + 1]

        in_bin = confidences.gt(bin_lower) * confidences.le(bin_upper)
        prop_in_bin = in_bin.float().mean().item()

        if prop_in_bin > 0:
            accuracy_in_bin = accuracies[in_bin].float().mean().item()
            avg_confidence_in_bin = confidences[in_bin].mean().item()
            ece += np.abs(avg_confidence_in_bin - accuracy_in_bin) * prop_in_bin

    return float(ece)

if __name__ == "__main__":
    print("Testing evaluation metrics...")
    dummy_probs = torch.softmax(torch.randn(4, 4, 128, 128), dim=1)
    dummy_targets = torch.randint(0, 4, (4, 128, 128))
    
    miou = compute_miou(dummy_probs, dummy_targets)
    ece = compute_expected_calibration_error(dummy_probs, dummy_targets)
    print(f"Sample mIoU: {miou:.4f}")
    print(f"Sample Expected Calibration Error (ECE): {ece:.4f}")
