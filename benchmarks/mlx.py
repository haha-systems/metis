"""Apple Silicon CPU/GPU benchmark for the MLX MLP baseline."""

import argparse
import json
import statistics
import time
import platform
from pathlib import Path

import mlx.core as mx
from mlx.utils import tree_flatten, tree_unflatten

from metis.mlx_model import MLXDecisionMLP
from benchmarks.ttd import sample_state


def _percentile(values, q):
    ordered = sorted(values)
    return ordered[min(len(ordered) - 1, round((len(ordered) - 1) * q))]


def _decode(output):
    choice = mx.softmax(output["choice_logits"], axis=-1)
    score_outputs = {}
    buckets = mx.arange(11, dtype=mx.float32) / 10.0
    for name, logits in output["score_logits"].items():
        probs = mx.softmax(logits, axis=-1)
        mean = (probs * buckets).sum(axis=-1)
        entropy = -(probs * mx.log(mx.maximum(probs, 1e-12))).sum(axis=-1)
        score_outputs[name] = (mean, entropy)
    choice_entropy = -(choice * mx.log(mx.maximum(choice, 1e-12))).sum(axis=-1)
    return choice, choice_entropy, score_outputs


def _materialize(decoded):
    choice, choice_entropy, score_outputs = decoded
    probabilities = choice.tolist()[0]
    score_values = {
        name: {"mean": mean.tolist()[0], "entropy": entropy.tolist()[0]}
        for name, (mean, entropy) in score_outputs.items()
    }
    return {"choice": max(range(len(probabilities)), key=probabilities.__getitem__),
            "choice_probabilities": probabilities, "uncertainty": choice_entropy.tolist()[0],
            "scores": score_values}


def benchmark(device="gpu", warmup=100, iterations=1000, model=None):
    mx.set_default_device(mx.gpu if device == "gpu" else mx.cpu)
    model = model or MLXDecisionMLP()
    model.eval()
    state = sample_state()
    # Materialize initial weights and warm runtime caches before measuring.
    mx.eval(model.parameters())
    for _ in range(warmup):
        output = model([state])
        decoded = _decode(output)
        mx.eval(decoded)
        _materialize(decoded)
    samples = []
    for _ in range(iterations):
        started = time.perf_counter_ns()
        output = model([state])
        decoded = _decode(output)
        mx.eval(decoded)
        _materialize(decoded)
        samples.append((time.perf_counter_ns() - started) / 1e6)
    return {
        "backend": "mlx", "device": device,
        "parameter_count": sum(int(value.size) for _, value in tree_flatten(model.parameters())),
        "platform": platform.platform(),
        "batch": 1, "iterations": iterations,
        "ttd": {"p50_ms": _percentile(samples, .5), "p95_ms": _percentile(samples, .95),
                "p99_ms": _percentile(samples, .99), "mean_ms": statistics.mean(samples),
                "stddev_ms": statistics.pstdev(samples)},
        "includes": ["schema encoding", "model forward", "probability decoding", "output materialization"],
        "target_p95_ms": 5.0,
        "meets_v01_latency_target": _percentile(samples, .95) < 5.0,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--device", choices=("cpu", "gpu"), default="gpu")
    parser.add_argument("--warmup", type=int, default=100)
    parser.add_argument("--iterations", type=int, default=1000)
    args = parser.parse_args()
    mx.set_default_device(mx.gpu if args.device == "gpu" else mx.cpu)
    model = MLXDecisionMLP()
    if args.checkpoint:
        model.update(tree_unflatten(list(mx.load(str(args.checkpoint)).items())))
    print(json.dumps(benchmark(args.device, args.warmup, args.iterations, model=model), indent=2))


if __name__ == "__main__":
    main()
