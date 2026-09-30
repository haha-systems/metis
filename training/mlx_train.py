"""Train the equivalent-schema MLX MLP baseline on Apple Silicon."""

import argparse
import json
from pathlib import Path
import platform
import random
import time
from importlib.metadata import PackageNotFoundError, version

import mlx.core as mx
import mlx.nn as nn
import mlx.optimizers as optim
import torch
from mlx.utils import tree_flatten

from metis.mlx_model import MLXDecisionMLP
from training.synthetic import score_bucket
from training.dataset import make_splits
from metis.heads import SCORE_NAMES
from benchmarks.mlx import benchmark as benchmark_mlx
from training.metrics import classification_metrics, score_metrics, abstention_metrics
from training.baselines import evaluate_baselines, compare_learned_to_baselines
from metis.provenance import source_fingerprint


def _batch(examples):
    from metis.mlx_model import encode_states
    return (encode_states([item.state for item in examples]),
            mx.array([[item.state.budget_pressure, item.state.resource_scarcity] for item in examples], dtype=mx.float32),
            mx.array([item.state.active_contradictions for item in examples], dtype=mx.float32),
            mx.array([item.choice for item in examples]),
            {name: mx.array([score_bucket(item.scores[name]) for item in examples]) for name in SCORE_NAMES})


def _evaluate(model, examples):
    _, _, _, choices, targets = _batch(examples)
    output = model([item.state for item in examples])
    choice_logits = torch.tensor(output["choice_logits"].tolist())
    choice_labels = torch.tensor(choices.tolist(), dtype=torch.long)
    decision_metrics = classification_metrics(choice_logits, choice_labels)
    decision_metrics.update(abstention_metrics(choice_logits, choice_labels))
    scores = {}
    for name, logits in output["score_logits"].items():
        scores[name] = score_metrics(torch.tensor(logits.tolist()),
                                     torch.tensor(targets[name].tolist(), dtype=torch.long))
    return {"decision": decision_metrics, "scores": scores}


def _predict_choices(model, examples):
    output = model([item.state for item in examples])
    return mx.softmax(output["choice_logits"], axis=-1).argmax(axis=-1).tolist()


def train(output: Path, size=12000, dataset_seed=17, training_seed=23, epochs=12,
          batch_size=128, learning_rate=.001, device="gpu"):
    random.seed(training_seed)
    mx.random.seed(training_seed)
    mx.set_default_device(mx.gpu if device == "gpu" else mx.cpu)
    splits = make_splits(size, dataset_seed)
    examples = list(splits.train)
    model = MLXDecisionMLP()
    parameter_count = sum(parameter.size for _, parameter in tree_flatten(model.parameters()))
    if parameter_count >= 10_000_000:
        raise ValueError("v0.1 training requires a model with fewer than 10M parameters")
    optimizer = optim.AdamW(learning_rate=learning_rate)

    def loss_fn(model, features, costs, contradictions, choices, score_targets):
        output = model.forward_features(features, costs, contradictions)
        loss = nn.losses.cross_entropy(output["choice_logits"], choices).mean()
        for name in SCORE_NAMES:
            loss = loss + .25 * nn.losses.cross_entropy(output["score_logits"][name], score_targets[name]).mean()
        return loss

    value_and_grad = nn.value_and_grad(model, loss_fn)
    started = time.perf_counter()
    history = []
    for epoch in range(epochs):
        random.shuffle(examples)
        epoch_losses = []
        for start in range(0, len(examples), batch_size):
            features, costs, contradictions, choices, scores = _batch(examples[start:start + batch_size])
            loss, gradients = value_and_grad(model, features, costs, contradictions, choices, scores)
            optimizer.update(model, gradients)
            mx.eval(model.parameters(), optimizer.state, loss)
            epoch_losses.append(float(loss.item()))
        history.append({"epoch": epoch + 1, "training_loss": sum(epoch_losses) / len(epoch_losses),
                        "validation": _evaluate(model, splits.validation)})
    elapsed = time.perf_counter() - started
    output.mkdir(parents=True, exist_ok=True)
    weights = {key: value for key, value in tree_flatten(model.parameters())}
    mx.save_safetensors(str(output / "model.safetensors"), weights)
    test_metrics = _evaluate(model, splits.test)
    test_predictions = _predict_choices(model, splits.test)
    ttd_metrics = benchmark_mlx(device=device, warmup=50, iterations=300, model=model)
    record = {"architecture": "schema-onehot-mlp", "hidden_dim": 64, "parameter_count": sum(int(x.size) for x in weights.values()),
        "git_revision": None, "source_fingerprint": source_fingerprint(),
        "dataset_seed": dataset_seed, "dataset_version": splits.version, "dataset_size": size,
        "training_seed": training_seed, "epochs": epochs, "batch_size": batch_size,
        "learning_rate": learning_rate, "loss_weights": {"choice": 1.0, **{name: .25 for name in SCORE_NAMES}},
        "device": device, "platform": platform.platform(), "mlx_version": _mlx_version(),
        "training_duration_seconds": elapsed, "history": history,
        "test_metrics": test_metrics, "test_baselines": evaluate_baselines(splits.test),
        "test_comparisons": compare_learned_to_baselines(splits.test, test_predictions), "ttd": ttd_metrics,
        "checkpoint": "model.safetensors"}
    (output / "metrics.json").write_text(json.dumps(record, indent=2) + "\n")
    (output / "ttd.json").write_text(json.dumps(ttd_metrics, indent=2) + "\n")
    (output / "config.json").write_text(json.dumps({"model": "schema-onehot-mlp", "hidden_dim": 64,
        "dataset_seed": dataset_seed, "training_seed": training_seed,
        "git_revision": record["git_revision"], "source_fingerprint": record["source_fingerprint"]}, indent=2) + "\n")
    return record


def _mlx_version():
    try:
        return version("mlx")
    except PackageNotFoundError:
        return None


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path("runs/mlx"))
    parser.add_argument("--size", type=int, default=12000)
    parser.add_argument("--dataset-seed", type=int, default=17)
    parser.add_argument("--training-seed", type=int, default=23)
    parser.add_argument("--epochs", type=int, default=12)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--learning-rate", type=float, default=.001)
    parser.add_argument("--device", choices=("cpu", "gpu"), default="gpu")
    args = parser.parse_args()
    print(json.dumps(train(args.output, args.size, args.dataset_seed, args.training_seed,
                           args.epochs, args.batch_size, args.learning_rate, args.device), indent=2))


if __name__ == "__main__":
    main()
