"""Utilities for discrete score distributions and uncertainty."""

import math
from typing import Dict

import torch

BUCKET_VALUES = tuple(i / 10.0 for i in range(11))


def probabilities(logits: torch.Tensor) -> torch.Tensor:
    return torch.softmax(logits, dim=-1)


def expected_value(logits: torch.Tensor) -> torch.Tensor:
    values = torch.as_tensor(BUCKET_VALUES, dtype=logits.dtype, device=logits.device)
    return (probabilities(logits) * values).sum(dim=-1)


def entropy(logits: torch.Tensor, normalize: bool = True) -> torch.Tensor:
    probs = probabilities(logits).clamp_min(torch.finfo(logits.dtype).tiny)
    result = -(probs * probs.log()).sum(dim=-1)
    return result / math.log(len(BUCKET_VALUES)) if normalize else result


def summarize_distribution(logits: torch.Tensor) -> Dict[str, torch.Tensor]:
    return {"mean": expected_value(logits), "entropy": entropy(logits), "probabilities": probabilities(logits)}

