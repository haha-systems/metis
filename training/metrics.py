"""Quality and calibration metrics for categorical decisions and score heads."""

from typing import Dict, Sequence

import torch
import torch.nn.functional as F


def classification_metrics(logits: torch.Tensor, labels: torch.Tensor, bins: int = 10) -> Dict[str, float]:
    probs = torch.softmax(logits, -1)
    pred = probs.argmax(-1)
    classes = probs.shape[-1]
    f1s = []
    for cls in range(classes):
        tp = ((pred == cls) & (labels == cls)).sum().float()
        fp = ((pred == cls) & (labels != cls)).sum().float()
        fn = ((pred != cls) & (labels == cls)).sum().float()
        f1s.append((2 * tp / (2 * tp + fp + fn).clamp_min(1)).item())
    confidence = probs.max(-1).values
    correct = (pred == labels).float()
    ece = torch.zeros((), device=probs.device)
    for i in range(bins):
        lower, upper = i / bins, (i + 1) / bins
        mask = (confidence > lower) & (confidence <= upper if i < bins - 1 else confidence <= upper)
        if mask.any():
            ece += mask.float().mean() * (confidence[mask].mean() - correct[mask].mean()).abs()
    one_hot = F.one_hot(labels, classes).to(probs.dtype)
    return {
        "accuracy": float((pred == labels).float().mean().item()),
        "macro_f1": sum(f1s) / classes,
        "cross_entropy": float(F.cross_entropy(logits, labels).item()),
        "brier_score": float(((probs - one_hot) ** 2).sum(-1).mean().item()),
        "expected_calibration_error": float(ece.item()),
    }


def score_metrics(score_logits: torch.Tensor, target: torch.Tensor) -> Dict[str, float]:
    probs = torch.softmax(score_logits, dim=-1)
    buckets = torch.arange(11, dtype=probs.dtype, device=probs.device) / 10.0
    mean = (probs * buckets).sum(-1)
    target_value = target.to(probs.dtype) / 10.0
    brier_target = F.one_hot(target, 11).to(probs.dtype)
    confidence, predicted = probs.max(-1)
    correct = (predicted == target).to(probs.dtype)
    ece = torch.zeros((), device=probs.device)
    for index in range(10):
        mask = (confidence > index / 10) & (confidence <= (index + 1) / 10)
        if mask.any():
            ece += mask.float().mean() * (confidence[mask].mean() - correct[mask].mean()).abs()
    return {
        "mae": float((mean - target_value).abs().mean().item()),
        "cross_entropy": float(F.cross_entropy(score_logits, target).item()),
        "brier_score": float(((probs - brier_target) ** 2).sum(-1).mean().item()),
        "expected_calibration_error": float(ece.item()),
    }


def fit_temperature(logits: torch.Tensor, labels: torch.Tensor) -> float:
    """Fit one positive scalar on held-out validation logits by NLL minimization."""
    # Clone outside inference mode so LBFGS can record gradients with respect
    # to temperature while treating the validation predictions as constants.
    logits = logits.detach().float().cpu().clone()
    labels = labels.detach().long().cpu().clone()
    log_temperature = torch.zeros((), requires_grad=True)
    optimizer = torch.optim.LBFGS([log_temperature], lr=0.1, max_iter=50)

    def closure():
        optimizer.zero_grad()
        loss = F.cross_entropy(logits / log_temperature.exp(), labels)
        loss.backward()
        return loss

    optimizer.step(closure)
    return float(log_temperature.detach().exp().clamp(0.05, 20.0).item())


def abstention_metrics(logits: torch.Tensor, labels: torch.Tensor) -> Dict[str, float]:
    from metis.schema import DecisionChoice
    probs = torch.softmax(logits, -1)
    predicted = probs.argmax(-1)
    abstain = int(DecisionChoice.ABSTAIN)
    truth = labels == abstain
    selected = predicted == abstain
    precision = ((selected & truth).sum() / selected.sum().clamp_min(1)).item()
    recall = ((selected & truth).sum() / truth.sum().clamp_min(1)).item()
    return {"abstention_precision": precision, "abstention_recall": recall}
