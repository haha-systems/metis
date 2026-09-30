"""Evaluate a saved checkpoint on held-out synthetic data and report regret."""

import argparse
import json
from pathlib import Path

import torch

from metis.model import ModelConfig, build_model
from metis.schema import DecisionChoice
from .baselines import evaluate_baselines, compare_learned_to_baselines
from .dataset import make_splits
from .train import _evaluate, _predict_choices


def regret(examples, choices):
    values = []
    for item, choice in zip(examples, choices):
        values.append(max(item.utilities.values()) - item.utilities[DecisionChoice(choice).name])
    return sum(values) / max(1, len(values))


def evaluate_checkpoint(checkpoint: Path, size: int = 12000, seed: int = 17, device: str = "cpu"):
    saved = torch.load(checkpoint, map_location=device, weights_only=False)
    config = ModelConfig(**saved["model_config"])
    model = build_model(config).to(device)
    model.load_state_dict(saved["state_dict"])
    model.choice_temperature = saved.get("choice_temperature", 1.0)
    splits = make_splits(size, seed)
    metrics = _evaluate(model, splits.test, torch.device(device))
    predictions = _predict_choices(model, splits.test, torch.device(device))
    result = {
        "learned": metrics,
        "learned_regret": regret(splits.test, predictions),
        "baselines": evaluate_baselines(splits.test),
        "baseline_comparisons": compare_learned_to_baselines(splits.test, predictions),
        "test_count": len(splits.test), "dataset_seed": seed,
    }
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("checkpoint", type=Path)
    parser.add_argument("--size", type=int, default=12000)
    parser.add_argument("--seed", type=int, default=17)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    result = evaluate_checkpoint(args.checkpoint, args.size, args.seed, args.device)
    rendered = json.dumps(result, indent=2)
    if args.output:
        args.output.write_text(rendered + "\n")
    print(rendered)


if __name__ == "__main__":
    main()
