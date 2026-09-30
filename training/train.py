"""Train one baseline family and persist weights, configuration and metrics."""

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import platform
import random
import subprocess
import sys
import time
from typing import Dict

import torch
import torch.nn.functional as F
from dataclasses import replace

from metis.heads import SCORE_NAMES
from metis.model import ModelConfig, build_model
from metis.provenance import source_fingerprint
from metis.distributions import expected_value
from metis.schema import DecisionChoice, EvidenceStrength, ProgressState, ResourceTier
from .dataset import make_splits
from .baselines import evaluate_baselines, compare_learned_to_baselines
from .metrics import classification_metrics, score_metrics, fit_temperature, abstention_metrics
from .synthetic import collate
from benchmarks.ttd import benchmark as benchmark_ttd
from metis.inference import MetisPredictor
from .metamorphic import check_metamorphic


def _consistency_loss(model, states, baseline_outputs):
    count = len(states)
    lower_budget = [replace(state, budget_pressure=min(1.0, state.budget_pressure + 0.25)) for state in states]
    higher_scarcity = [replace(state, resource_scarcity=min(1.0, state.resource_scarcity + 0.25)) for state in states]
    more_contradictions = [replace(state, active_contradictions=min(12, state.active_contradictions + 3)) for state in states]
    less_evidence = [replace(state, progress=ProgressState.UNKNOWN,
        progress_evidence=EvidenceStrength.NONE, uncertainty=min(1.0, state.uncertainty + 0.25)) for state in states]
    lower_tier = [replace(state, current_resource_tier=ResourceTier(
        max(int(ResourceTier.LOW), int(state.current_resource_tier) - 1))) for state in states]
    variants = lower_budget + higher_scarcity + more_contradictions + less_evidence + lower_tier
    changed = model(variants)
    base_choice = torch.softmax(baseline_outputs["choice_logits"], -1)
    cost_choice, scarcity_choice, decomp_choice, evidence_choice, tier_choice = [
        torch.softmax(changed["choice_logits"][i * count:(i + 1) * count], -1)
        for i in range(5)
    ]
    base_gain = expected_value(baseline_outputs["score_logits"]["expected_gain_from_stronger_cognition"])
    tier_gain = expected_value(changed["score_logits"]["expected_gain_from_stronger_cognition"][4 * count:])
    penalties = [
        F.relu(cost_choice[:, int(DecisionChoice.ESCALATE)] - base_choice[:, int(DecisionChoice.ESCALATE)]),
        F.relu(scarcity_choice[:, int(DecisionChoice.ESCALATE)] - base_choice[:, int(DecisionChoice.ESCALATE)]),
        F.relu(base_choice[:, int(DecisionChoice.DECOMPOSE)] - decomp_choice[:, int(DecisionChoice.DECOMPOSE)]),
        F.relu(base_choice[:, int(DecisionChoice.ABSTAIN)] - evidence_choice[:, int(DecisionChoice.ABSTAIN)]),
        F.relu(base_gain - tier_gain),
        F.relu(base_choice[:, int(DecisionChoice.ESCALATE)] - tier_choice[:, int(DecisionChoice.ESCALATE)]),
    ]
    return torch.stack([penalty.mean() for penalty in penalties]).mean()


def _loss(model, outputs, batch, weights: Dict[str, float]):
    total = weights["choice"] * F.cross_entropy(outputs["choice_logits"], batch["choices"])
    details = {"choice": float(F.cross_entropy(outputs["choice_logits"], batch["choices"]).detach())}
    for name in SCORE_NAMES:
        value = F.cross_entropy(outputs["score_logits"][name], batch["scores"][name])
        total = total + weights[name] * value
        details[name] = float(value.detach())
    consistency = _consistency_loss(model, batch["states"], outputs)
    total = total + weights["metamorphic"] * consistency
    details["metamorphic"] = float(consistency.detach())
    return total, details


@torch.inference_mode()
def _evaluate(model, examples, device, batch_size=256):
    model.eval()
    collected = {"choices": [], "choice_logits": [], "scores": {name: [] for name in SCORE_NAMES},
                 "score_logits": {name: [] for name in SCORE_NAMES}}
    for start in range(0, len(examples), batch_size):
        batch = collate(examples[start:start + batch_size], device)
        out = model(batch["states"])
        collected["choices"].append(batch["choices"])
        collected["choice_logits"].append(out["choice_logits"])
        for name in SCORE_NAMES:
            collected["scores"][name].append(batch["scores"][name])
            collected["score_logits"][name].append(out["score_logits"][name])
    choice_logits = torch.cat(collected["choice_logits"]) / max(float(model.choice_temperature), 1e-4)
    result = {"decision": classification_metrics(choice_logits, torch.cat(collected["choices"]))}
    result["decision"].update(abstention_metrics(choice_logits, torch.cat(collected["choices"])))
    result["scores"] = {name: score_metrics(torch.cat(collected["score_logits"][name]),
                                             torch.cat(collected["scores"][name])) for name in SCORE_NAMES}
    return result


@torch.inference_mode()
def _predict_choices(model, examples, device, batch_size=256):
    model.eval()
    choices = []
    for start in range(0, len(examples), batch_size):
        batch = collate(examples[start:start + batch_size], device)
        logits = model(batch["states"])["choice_logits"] / max(float(model.choice_temperature), 1e-4)
        choices.extend(logits.argmax(-1).cpu().tolist())
    return choices


def train(config: ModelConfig, output_dir: Path, size: int = 12000, dataset_seed: int = 17,
          training_seed: int = 23, epochs: int = 12, batch_size: int = 128,
          learning_rate: float = 0.001, device: str = "auto") -> Dict[str, object]:
    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "mps" if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available() else "cpu"
    torch.manual_seed(training_seed)
    random.seed(training_seed)
    target = torch.device(device)
    splits = make_splits(size, dataset_seed)
    model = build_model(config).to(target)
    if model.parameter_count() >= 10_000_000:
        raise ValueError("v0.1 training requires a model with fewer than 10M parameters")
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)
    loss_weights = {"choice": 1.0, **{name: 0.25 for name in SCORE_NAMES}, "metamorphic": 1.0}
    started = time.perf_counter()
    history = []
    for epoch in range(epochs):
        model.train()
        order = torch.randperm(len(splits.train)).tolist()
        weighted_loss = 0.0
        consistency_loss = 0.0
        for offset in range(0, len(order), batch_size):
            batch = collate([splits.train[i] for i in order[offset:offset + batch_size]], target)
            optimizer.zero_grad(set_to_none=True)
            outputs = model(batch["states"])
            loss, details = _loss(model, outputs, batch, loss_weights)
            loss.backward()
            optimizer.step()
            weighted_loss += loss.item() * len(batch["states"])
            consistency_loss += details["metamorphic"] * len(batch["states"])
        validation = _evaluate(model, splits.validation, target)
        history.append({"epoch": epoch + 1, "train_loss": weighted_loss / len(splits.train),
                        "metamorphic_loss": consistency_loss / len(splits.train), "validation": validation})
    duration = time.perf_counter() - started
    validation_batch = collate(splits.validation, target)
    model.eval()
    with torch.inference_mode():
        validation_logits = model(validation_batch["states"])["choice_logits"]
    validation_labels = validation_batch["choices"]
    calibration_before = classification_metrics(validation_logits, validation_labels)
    model.choice_temperature = fit_temperature(validation_logits, validation_labels)
    calibration_after = classification_metrics(validation_logits / model.choice_temperature, validation_labels)
    test_metrics = _evaluate(model, splits.test, target)
    test_predictions = _predict_choices(model, splits.test, target)
    ttd_metrics = benchmark_ttd(model, device, warmup=50, iterations=300)
    metamorphic_metrics = check_metamorphic(
        MetisPredictor(model, device), [item.state for item in splits.test[:64]])
    output_dir.mkdir(parents=True, exist_ok=True)
    checkpoint = output_dir / "model.pt"
    torch.save({"state_dict": model.cpu().state_dict(), "model_config": config.__dict__,
                "choice_temperature": model.choice_temperature}, checkpoint)
    record = {
        "created_at": datetime.now(timezone.utc).isoformat(), "git_revision": _git_revision(),
        "source_fingerprint": source_fingerprint(),
        "architecture": config.__dict__, "parameter_count": model.parameter_count(),
        "dataset_seed": dataset_seed, "dataset_version": splits.version, "dataset_size": size,
        "training_seed": training_seed, "hyperparameters": {"epochs": epochs, "batch_size": batch_size,
        "learning_rate": learning_rate}, "loss_weights": loss_weights, "device": device,
        "platform": platform.platform(), "training_duration_seconds": duration,
        "python_version": sys.version,
        "history": history, "test_metrics": test_metrics,
        "choice_calibration": {"temperature": model.choice_temperature,
                               "validation_before": calibration_before, "validation_after": calibration_after},
        "ttd": ttd_metrics, "metamorphic": metamorphic_metrics,
        "test_baselines": evaluate_baselines(splits.test),
        "test_comparisons": compare_learned_to_baselines(splits.test, test_predictions),
        "checkpoint": checkpoint.name,
    }
    (output_dir / "config.json").write_text(json.dumps({"model": config.__dict__, "loss_weights": loss_weights,
        "git_revision": record["git_revision"], "source_fingerprint": record["source_fingerprint"]}, indent=2) + "\n")
    (output_dir / "metrics.json").write_text(json.dumps(record, indent=2) + "\n")
    (output_dir / "ttd.json").write_text(json.dumps(ttd_metrics, indent=2) + "\n")
    (output_dir / "metamorphic.json").write_text(json.dumps(metamorphic_metrics, indent=2) + "\n")
    return record


def _git_revision():
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"], text=True, stderr=subprocess.DEVNULL).strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--family", choices=("linear", "mlp", "transformer", "mixer"), default="transformer")
    parser.add_argument("--output", type=Path, default=Path("runs") / datetime.now().strftime("%Y%m%d-%H%M%S"))
    parser.add_argument("--size", type=int, default=12000)
    parser.add_argument("--dataset-seed", type=int, default=17)
    parser.add_argument("--training-seed", type=int, default=23)
    parser.add_argument("--epochs", type=int, default=12)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--learning-rate", type=float, default=0.001)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--count-encoding", choices=("scalar", "bucket"), default="scalar")
    args = parser.parse_args()
    config = ModelConfig(family=args.family, count_encoding=args.count_encoding)
    print(json.dumps(train(config, args.output, args.size, args.dataset_seed, args.training_seed,
                           args.epochs, args.batch_size, args.learning_rate, args.device), indent=2))


if __name__ == "__main__":
    main()
