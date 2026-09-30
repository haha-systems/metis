"""Batch throughput benchmark, reported independently from batch-1 TTD."""

import argparse
import json
import platform
import time
from pathlib import Path

import torch

from metis.model import ModelConfig, build_model
from .ttd import sample_state
from .ttd import _synchronize
from metis.inference import MetisPredictor


def benchmark(model, device="cpu", batches=(1, 8, 32, 128), iterations=100):
    predictor = MetisPredictor(model, device)
    state = sample_state()
    results = {}
    with torch.inference_mode():
        for batch_size in batches:
            states = [state] * batch_size
            for _ in range(10):
                predictor.predict_batch(states)
            _synchronize(device)
            started = time.perf_counter()
            for _ in range(iterations):
                predictor.predict_batch(states)
            _synchronize(device)
            elapsed = time.perf_counter() - started
            results[str(batch_size)] = {
                "decisions_per_second": batch_size * iterations / elapsed,
                "batch_latency_ms": elapsed * 1000 / iterations,
                "latency_per_decision_ms": elapsed * 1000 / (batch_size * iterations),
                "includes": ["schema encoding", "model forward", "output decoding"],
            }
    return results


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--family", choices=("linear", "mlp", "transformer", "mixer"), default="mlp")
    parser.add_argument("--count-encoding", choices=("scalar", "bucket"), default="scalar")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--iterations", type=int, default=100)
    args = parser.parse_args()
    if args.checkpoint:
        saved = torch.load(args.checkpoint, map_location=args.device, weights_only=False)
        model = build_model(ModelConfig(**saved["model_config"]))
        model.load_state_dict(saved["state_dict"])
        model.choice_temperature = saved.get("choice_temperature", 1.0)
    else:
        model = build_model(ModelConfig(family=args.family, count_encoding=args.count_encoding))
    print(json.dumps({"backend": "pytorch", "device": args.device,
        "platform": platform.platform(), "pytorch_version": torch.__version__,
        "model": model.config_dict(), "parameter_count": model.parameter_count(),
        "results": benchmark(model, args.device, iterations=args.iterations)}, indent=2))


if __name__ == "__main__":
    main()
