"""Benchmark a trained PyTorch Transformer checkpoint through MLX."""

import argparse
import hashlib
import json
from pathlib import Path
import platform
import time

import mlx.core as mx
from mlx.utils import tree_flatten
import torch

from benchmarks.mlx import benchmark as benchmark_ttd
from benchmarks.mlx_throughput import benchmark as benchmark_throughput
from metis.mlx_transformer import MLXTransformerCheckpoint


def run(checkpoint: Path, device: str, warmup: int, iterations: int,
        throughput_iterations: int, output: Path):
    mx.set_default_device(mx.gpu if device == "gpu" else mx.cpu)
    started = time.perf_counter()
    saved = torch.load(checkpoint, map_location="cpu", weights_only=False)
    model = MLXTransformerCheckpoint(saved["model_config"])
    model.load_torch_state_dict(saved["state_dict"])
    model.choice_temperature = float(saved.get("choice_temperature", 1.0))
    cold_load_ms = (time.perf_counter() - started) * 1000

    ttd = benchmark_ttd(device=device, warmup=warmup, iterations=iterations, model=model)
    throughput = benchmark_throughput(model, device=device, iterations=throughput_iterations)
    checkpoint_sha256 = hashlib.sha256(checkpoint.read_bytes()).hexdigest()
    result = {
        "backend": "mlx",
        "device": device,
        "platform": platform.platform(),
        "python_version": platform.python_version(),
        "mlx_version": mx.__version__,
        "torch_checkpoint_format": "pytorch-state-dict-converted-to-mlx",
        "checkpoint": checkpoint.name,
        "checkpoint_sha256": checkpoint_sha256,
        "model": saved["model_config"],
        "parameter_count": sum(int(value.size) for _, value in tree_flatten(model.parameters())),
        "choice_temperature": model.choice_temperature,
        "cold_load_ms": cold_load_ms,
        "ttd": ttd,
        "throughput": throughput,
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2) + "\n")
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--device", choices=("cpu", "gpu"), default="gpu")
    parser.add_argument("--warmup", type=int, default=100)
    parser.add_argument("--iterations", type=int, default=1000)
    parser.add_argument("--throughput-iterations", type=int, default=100)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    output = args.output or args.checkpoint.with_name(f"mlx_same_checkpoint_{args.device}.json")
    print(json.dumps(run(args.checkpoint, args.device, args.warmup, args.iterations,
                         args.throughput_iterations, output), indent=2))


if __name__ == "__main__":
    main()
