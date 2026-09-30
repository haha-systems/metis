"""Seeded train/validation/test splits and machine-readable data metadata."""

from dataclasses import dataclass
from typing import List, Tuple

from .synthetic import Example, generate_dataset


@dataclass(frozen=True)
class DatasetSplits:
    train: List[Example]
    validation: List[Example]
    test: List[Example]
    seed: int
    version: str = "synthetic-v1"


def make_splits(size: int = 12000, seed: int = 0,
                fractions: Tuple[float, float, float] = (0.8, 0.1, 0.1)) -> DatasetSplits:
    if size < 10:
        raise ValueError("dataset size must be at least 10 to populate all splits")
    if abs(sum(fractions) - 1.0) > 1e-8 or min(fractions) <= 0:
        raise ValueError("split fractions must be positive and sum to 1")
    examples = generate_dataset(size, seed)
    n_train = int(size * fractions[0])
    n_val = int(size * fractions[1])
    return DatasetSplits(examples[:n_train], examples[n_train:n_train + n_val],
                         examples[n_train + n_val:], seed)
