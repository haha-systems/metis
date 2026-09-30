"""Warm batch-1 end-to-end TTD benchmark. Model loading is measured separately."""

import argparse
import json
import platform
import statistics
import time
from pathlib import Path

import torch

from metis.inference import MetisPredictor
from metis.model import ModelConfig, build_model
from metis.schema import CognitiveState, EvidenceStrength, Phase, ProgressState, ResourceTier


def sample_state():
    return CognitiveState(Phase.EXECUTE, ResourceTier.MEDIUM, 2, 1, 1, 0, ProgressState.LOW,
        EvidenceStrength.VERIFIED, True, False, 4, 1, True, True, 0.45, 0.62, 0.25, 0.2)


def percentile(values, quantile):
    ordered = sorted(values)
    return ordered[min(len(ordered) - 1, max(0, int(round((len(ordered) - 1) * quantile))))]


def summarize(values):
    return {"p50_ms": percentile(values, .50), "p95_ms": percentile(values, .95),
            "p99_ms": percentile(values, .99), "mean_ms": statistics.mean(values),
            "stddev_ms": statistics.pstdev(values)}


def _synchronize(device):
    if device.startswith("cuda"):
        torch.cuda.synchronize()
    elif device.startswith("mps"):
        torch.mps.synchronize()


@torch.inference_mode()
def benchmark(model, device="cpu", warmup=100, iterations=1000):
    predictor = MetisPredictor(model, device)
    state = sample_state()
    for _ in range(warmup):
        predictor.predict(state)
    _synchronize(device)
    samples = []
    for _ in range(iterations):
        start = time.perf_counter_ns()
        predictor.predict(state)
        _synchronize(device)
        samples.append((time.perf_counter_ns() - start) / 1e6)
    # Measure the components in isolation as well as full TTD. These stage
    # distributions are diagnostic and need not add exactly due to timer overhead.
    state_batch = [state]
    encoded = model.encoder(state_batch, ignored_numeric_fields={
        "budget_pressure", "resource_scarcity", "active_contradictions"})
    outputs = model.forward_encoded(encoded, state_batch)
    encode_samples, forward_samples, postprocess_samples = [], [], []
    for _ in range(iterations):
        start = time.perf_counter_ns()
        tokens = model.encoder(state_batch, ignored_numeric_fields={
            "budget_pressure", "resource_scarcity", "active_contradictions"})
        _synchronize(device)
        encode_samples.append((time.perf_counter_ns() - start) / 1e6)

        start = time.perf_counter_ns()
        result = model.forward_encoded(tokens, state_batch)
        _synchronize(device)
        forward_samples.append((time.perf_counter_ns() - start) / 1e6)

        start = time.perf_counter_ns()
        predictor.decode_outputs(outputs)
        postprocess_samples.append((time.perf_counter_ns() - start) / 1e6)
    return {"ttd": summarize(samples), "encode": summarize(encode_samples),
            "forward": summarize(forward_samples), "postprocess": summarize(postprocess_samples),
            "iterations": iterations, "batch": 1}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--family", choices=("linear", "mlp", "transformer", "mixer"), default="mlp")
    parser.add_argument("--count-encoding", choices=("scalar", "bucket"), default="scalar")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--warmup", type=int, default=100)
    parser.add_argument("--iterations", type=int, default=1000)
    args = parser.parse_args()
    started = time.perf_counter()
    if args.checkpoint:
        saved = torch.load(args.checkpoint, map_location=args.device, weights_only=False)
        config = ModelConfig(**saved["model_config"])
        model = build_model(config)
        model.load_state_dict(saved["state_dict"])
        model.choice_temperature = saved.get("choice_temperature", 1.0)
    else:
        model = build_model(ModelConfig(family=args.family, count_encoding=args.count_encoding))
    cold_load = (time.perf_counter() - started) * 1000
    result = benchmark(model, args.device, args.warmup, args.iterations)
    result.update({"model": model.config_dict(), "parameter_count": model.parameter_count(),
        "backend": "pytorch", "device": args.device, "platform": platform.platform(),
        "pytorch_version": torch.__version__,
        "torch_cpu_threads": torch.get_num_threads(),
        "cold_load_ms": cold_load, "target_p95_ms": 5.0,
        "meets_v01_latency_target": result["ttd"]["p95_ms"] < 5.0})
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
