# Metis v0.1

Metis is a small, non-autoregressive model for decisions over a fixed structured QAC state. The initial experiment predicts one of `CONTINUE`, `ESCALATE`, `DECOMPOSE`, or `ABSTAIN` together with distributional estimates for resource adequacy, stronger-cognition gain, decomposition need, and evidence sufficiency.

## Install

```sh
python -m venv .venv
. .venv/bin/activate
python -m pip install -e '.[dev]'
```

The core dependency is PyTorch. Apple Silicon MLX support is optional:

```sh
python -m pip install -e '.[mlx]'
```

## Train and evaluate

The synthetic generator produces examples from latent task difficulty, resource capability, evidence reliability, expected gain, decomposition need, and budget conditions. A small deterministic share of each dataset is reserved for designed counterexamples. Fixed seeds reproduce generated examples and splits.

```sh
metis-train --family linear --output runs/linear --dataset-seed 17
metis-train --family mlp --output runs/mlp --dataset-seed 17
metis-train --family transformer --output runs/transformer --dataset-seed 17
metis-train --family mixer --output runs/mixer --dataset-seed 17
metis-evaluate runs/transformer/model.pt --seed 17
```

Each training run writes `config.json`, `metrics.json`, `ttd.json`, `metamorphic.json`, and `model.pt`. Metrics include multi-task validation history, held-out classification and score metrics, validation-fitted choice temperature, deterministic/heuristic baseline comparisons with paired bootstrap intervals, metamorphic pass rates, training metadata, and a warm batch-1 TTD report. Loss weights are recorded explicitly.

## Inference and replay

```python
from metis import CognitiveState, MetisPredictor

predictor = MetisPredictor(model, device="cpu")
decision = predictor.predict(state)
print(decision.choice, decision.choice_probabilities, decision.scores)
```

`metis-replay CHECKPOINT INPUT.jsonl OUTPUT.jsonl` runs an offline replay over user-provided records. A record may contain the state directly or under `state`, with categorical enum values written by name. An optional `historical_choice` is preserved in each output for later comparison. No production routing or data export is included.

The exact record contract and the current Ghost trace coverage are documented in [docs/ghost-replay.md](docs/ghost-replay.md).

## Benchmarks

```sh
metis-ttd --checkpoint runs/transformer/model.pt --device cpu
metis-throughput --checkpoint runs/transformer/model.pt --device cpu
metis-mlx-train --output runs/mlx --dataset-seed 17
python -m benchmarks.mlx --device gpu
python -m benchmarks.mlx_throughput --checkpoint runs/mlx/model.safetensors --device gpu
metis-mlx-checkpoint --checkpoint runs/transformer-full/model.pt --device cpu
metis-mlx-checkpoint --checkpoint runs/transformer-full/model.pt --device gpu
```

TTD includes state encoding, model inference, and output decoding/calibration; checkpoint loading is reported separately. The report includes p50/p95/p99, mean, and standard deviation, with isolated encoding, forward, and postprocess timings. Throughput results remain separate at batch sizes 1, 8, 32, and 128. `metis-mlx-checkpoint` converts a PyTorch Transformer checkpoint into MLX layers and benchmarks the exact same weights on Apple CPU or GPU, for a direct comparison with the Colab CUDA and PyTorch CPU results. The separately trained MLX MLP remains an independent architecture baseline.

## Metamorphic diagnostics

`training.metamorphic.check_metamorphic` evaluates directional counterfactual properties over a state sample: higher budget pressure should not raise escalation probability, stronger decomposition need should not lower decomposition probability, weaker evidence should raise abstention probability or uncertainty, and changing unknown progress to verified low progress should affect the distribution. These are reported pass rates over the learned model rather than hard-coded policy overrides.

## Initial seeded results

The 2026-09-23 run used the same 12,000-example `synthetic-v1` dataset (seed 17), training seed 23, and 12 epochs for each PyTorch family. The paired 95% bootstrap interval for learned accuracy improvement over the heuristic was fully above zero for all four families. The Transformer gave 75.4% held-out accuracy, 72,819 parameters, 0.62 ms warm batch-1 CPU p95 TTD, and a 30.8–38.3 percentage-point accuracy improvement interval. The Mixer gave 74.8% accuracy with 42,055 parameters. The chosen Transformer test ECE was 0.020 and abstention recall was 0.882.

The controlled count-encoding experiment used the same source fingerprint, data split, and training seeds. Scalar encoding scored 75.4% accuracy with 72,819 parameters and 0.624 ms CPU p95 TTD; learned buckets scored 74.4% with 79,347 parameters and 0.632 ms. Scalar encoding is the selected default. Full metrics and batch throughput are in [`runs/count-encoding-comparison.json`](runs/count-encoding-comparison.json).

The separately trained equivalent-schema MLX MLP reached 75.7% held-out accuracy with 9,267 parameters. On the same Apple Silicon host, measured p95 TTD was 0.15 ms on MLX CPU and 0.52 ms on MLX GPU. The local 72,819-parameter PyTorch Transformer checkpoint converted to MLX measured 0.55 ms CPU p95 and 1.04 ms GPU p95. Its PyTorch CPU p95 was 0.45 ms and MPS p95 was 18.45 ms. Throughput at batches 1, 8, 32, and 128 is recorded separately.

The Colab artifact provides a cross-hardware comparison using its exact trained checkpoint (SHA-256 is in the record). The 72,819-parameter T4 model reached 74.9% held-out accuracy and improved on the heuristic by 34.1 percentage points (95% paired bootstrap CI: 30.5–37.8 points). Its batch-1 p95 TTD was 5.67 ms on the Tesla T4 and 2.17 ms on Colab CPU; the T4 misses the 5 ms latency target. The same checkpoint measured 0.47 ms on local PyTorch CPU, 1.15 ms on MLX CPU, and 1.19 ms on MLX GPU.

The larger 804,787-parameter Transformer used the same Colab source fingerprint, dataset split, and training seeds. It reached 73.9% accuracy and 57.8% macro-F1, compared with 74.9% and 56.6% for the small model. Its T4 p95 TTD rose to 6.87 ms and batch-128 throughput fell from 20,018 to 12,294 decisions/s. On Apple Silicon the large checkpoint measured 1.02 ms p95 on PyTorch CPU, 1.12 ms on MLX CPU, and 1.80 ms on MLX GPU. Full results are in [`runs/colab-t4-model-size-comparison.json`](runs/colab-t4-model-size-comparison.json) and [`runs/t4-large-cross-platform-comparison.json`](runs/t4-large-cross-platform-comparison.json). The Colab and current-checkout source fingerprints differ; the configs, dataset seeds, and training seeds match. Combined measurements are in [`runs/hardware-comparison.json`](runs/hardware-comparison.json).

These are synthetic-domain results from one Apple Silicon host and one Colab T4 session. No real Ghost/QAC replay has been run because no complete decision-time JSONL export is available.

Run the project tests with `python -m pytest`. Benchmarks and tests require the declared dependencies; MLX requires Apple Silicon and its optional package.

## Scope and current research status

The decision head uses learned, sign-constrained terms for escalation costs and decomposition pressure. These keep the cost and contradiction counterfactuals monotone while leaving their magnitudes learned from data. Consistency penalties train the other paired properties; all measured pass rates are recorded in each run.

A runnable Google Colab notebook is available at [notebooks/metis_t4_benchmark.ipynb](notebooks/metis_t4_benchmark.ipynb), with a ready-to-upload [source archive](notebooks/metis-colab-source.zip). It produces the checkpoint, quality metrics, TTD, throughput, and hardware metadata. Synthetic results alone are not evidence of production suitability.
