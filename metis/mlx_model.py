"""MLX MLP baseline over the same fixed schema and output vocabulary.

The MLX baseline uses an explicit one-hot typed representation so the same
conceptual decision function can be trained and benchmarked on Apple Silicon.
It is kept separate from PyTorch modules because MLX has its own array/runtime.
"""

from typing import List

try:
    import mlx.core as mx
    import mlx.nn as nn
except ImportError as exc:  # pragma: no cover - depends on Apple Silicon setup
    raise ImportError("Install the optional 'mlx' dependency on Apple Silicon to use Metis MLX") from exc

from metis.encoding import BOOLEAN_FIELDS, CATEGORICAL_FIELDS, NUMERIC_FIELDS, NUMERIC_MAX
from metis.schema import CognitiveState


FEATURE_DIM = sum(CATEGORICAL_FIELDS.values()) + len(BOOLEAN_FIELDS) + len(NUMERIC_FIELDS)


def encode_states(states: List[CognitiveState]):
    rows = []
    for state in states:
        row = []
        for name, size in CATEGORICAL_FIELDS.items():
            one_hot = [0.0] * size
            one_hot[int(getattr(state, name))] = 1.0
            row.extend(one_hot)
        row.extend(float(getattr(state, name)) for name in BOOLEAN_FIELDS)
        row.extend(0.0 if name in {"budget_pressure", "resource_scarcity", "active_contradictions"} else
                   min(float(getattr(state, name)) / NUMERIC_MAX.get(name, 1.0), 1.0)
                   for name in NUMERIC_FIELDS)
        rows.append(row)
    return mx.array(rows, dtype=mx.float32)


class MLXDecisionMLP(nn.Module):
    def __init__(self, hidden_dim: int = 64):
        super().__init__()
        self.hidden1 = nn.Linear(FEATURE_DIM, hidden_dim)
        self.hidden2 = nn.Linear(hidden_dim, hidden_dim)
        self.choice = nn.Linear(hidden_dim, 4)
        self.adequacy = nn.Linear(hidden_dim, 11)
        self.expected_gain = nn.Linear(hidden_dim, 11)
        self.decomposition_need = nn.Linear(hidden_dim, 11)
        self.evidence_sufficiency = nn.Linear(hidden_dim, 11)
        self.escalation_cost_coefficients = mx.array([-2.0, -2.0], dtype=mx.float32)
        self.decomposition_sensitivity = mx.array(-2.0, dtype=mx.float32)

    def forward_features(self, features, costs, contradictions):
        hidden = nn.gelu(self.hidden1(features))
        hidden = nn.gelu(self.hidden2(hidden))
        choice_logits = self.choice(hidden)
        cost_penalty = (costs * mx.log(1.0 + mx.exp(self.escalation_cost_coefficients))).sum(axis=-1)
        selector = mx.array([0.0, 1.0, 0.0, 0.0], dtype=choice_logits.dtype)
        choice_logits = choice_logits - cost_penalty[:, None] * selector[None, :]
        normalized_contradictions = contradictions / 12.0
        decomp_bonus = mx.log(1.0 + mx.exp(self.decomposition_sensitivity)) * normalized_contradictions
        decomp_selector = mx.array([0.0, 0.0, 1.0, 0.0], dtype=choice_logits.dtype)
        choice_logits = choice_logits + decomp_bonus[:, None] * decomp_selector[None, :]
        score_logits = {
            "current_resource_adequacy": self.adequacy(hidden),
            "expected_gain_from_stronger_cognition": self.expected_gain(hidden),
            "decomposition_need": self.decomposition_need(hidden) +
                decomp_bonus[:, None] * (mx.arange(11, dtype=hidden.dtype) / 10.0)[None, :],
            "evidence_sufficiency": self.evidence_sufficiency(hidden),
        }
        return {
            "choice_logits": choice_logits,
            "score_logits": score_logits,
        }

    def __call__(self, states):
        costs = mx.array([[state.budget_pressure, state.resource_scarcity] for state in states], dtype=mx.float32)
        contradictions = mx.array([state.active_contradictions for state in states], dtype=mx.float32)
        return self.forward_features(encode_states(states), costs, contradictions)
