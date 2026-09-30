import pytest
import torch

mx = pytest.importorskip("mlx.core")

from benchmarks.ttd import sample_state
from metis.model import ModelConfig, build_model
from metis.mlx_transformer import MLXTransformerCheckpoint
from metis.schema import EvidenceStrength, ProgressState
from dataclasses import replace


def test_mlx_checkpoint_matches_pytorch_transformer_outputs():
    torch.manual_seed(11)
    config = ModelConfig(family="transformer", embedding_dim=16, layers=1,
                         attention_heads=4, ffn_multiplier=2, count_encoding="scalar")
    torch_model = build_model(config).eval()
    torch_model.choice_temperature = 0.83
    mlx_model = MLXTransformerCheckpoint(config.__dict__).load_torch_state_dict(torch_model.state_dict())
    mlx_model.choice_temperature = torch_model.choice_temperature
    states = [
        sample_state(),
        replace(sample_state(), progress=ProgressState.UNKNOWN,
                progress_evidence=EvidenceStrength.NONE, budget_pressure=0.9,
                resource_scarcity=0.7, active_contradictions=8),
    ]
    with torch.inference_mode():
        expected = torch_model(states)
    actual = mlx_model(states)
    mx.eval(actual)
    # Attention kernels can differ slightly across runtimes; decisions and score
    # distributions should still agree closely for the exact converted weights.
    actual_choice = torch.tensor(actual["choice_logits"].tolist())
    expected_choice = expected["choice_logits"] / torch_model.choice_temperature
    assert torch.allclose(actual_choice, expected_choice, atol=1e-2, rtol=1e-3)
    for name in expected["score_logits"]:
        actual_scores = torch.tensor(actual["score_logits"][name].tolist())
        assert torch.allclose(actual_scores, expected["score_logits"][name], atol=1e-4, rtol=1e-4)
