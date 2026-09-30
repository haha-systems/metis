"""MLX batch throughput benchmark using saved MLX checkpoint weights."""

import argparse
import json
import platform
import time
from pathlib import Path

import mlx.core as mx
from mlx.utils import tree_flatten, tree_unflatten

from metis.mlx_model import MLXDecisionMLP
from benchmarks.mlx import _decode, _materialize
from benchmarks.ttd import sample_state


def benchmark(model, device="gpu", batches=(1, 8, 32, 128), iterations=100):
    mx.set_default_device(mx.gpu if device == "gpu" else mx.cpu)
    model.eval()
    state = sample_state()
    results = {}
    for batch_size in batches:
        states = [state] * batch_size
        for _ in range(10):
            decoded = _decode(model(states))
            mx.eval(decoded)
            _materialize(decoded)
        started = time.perf_counter()
        for _ in range(iterations):
            decoded = _decode(model(states))
            mx.eval(decoded)
            _materialize(decoded)
        elapsed = time.perf_counter() - started
        results[str(batch_size)] = {
            "decisions_per_second": batch_size * iterations / elapsed,
            "batch_latency_ms": elapsed * 1000 / iterations,
            "latency_per_decision_ms": elapsed * 1000 / (batch_size * iterations),
        }
    return results


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--device", choices=("cpu", "gpu"), default="gpu")
    parser.add_argument("--iterations", type=int, default=100)
    args = parser.parse_args()
    mx.set_default_device(mx.gpu if args.device == "gpu" else mx.cpu)
    model = MLXDecisionMLP()
    model.update(tree_unflatten(list(mx.load(str(args.checkpoint)).items())))
    print(json.dumps({"backend": "mlx", "device": args.device, "platform": platform.platform(),
        "parameter_count": sum(int(value.size) for _, value in tree_flatten(model.parameters())),
        "includes": ["schema encoding", "model forward", "probability decoding", "output materialization"],
        "results": benchmark(model, args.device, iterations=args.iterations)}, indent=2))


if __name__ == "__main__":
    main()
