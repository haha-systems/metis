"""MLX inference for the same Transformer checkpoint used by PyTorch."""

from typing import Sequence

try:
    import mlx.core as mx
    import mlx.nn as nn
except ImportError as exc:  # pragma: no cover - depends on Apple Silicon setup
    raise ImportError("Install the optional 'mlx' dependency on Apple Silicon") from exc

from metis.encoding import BOOLEAN_FIELDS, CATEGORICAL_FIELDS, NUMERIC_FIELDS, NUMERIC_MAX
from metis.schema import CognitiveState


def _gelu(value):
    return 0.5 * value * (1.0 + mx.erf(value / 1.4142135623730951))


class _TransformerBlock(nn.Module):
    def __init__(self, width: int, heads: int, ffn_width: int):
        super().__init__()
        self.width = width
        self.heads = heads
        self.head_width = width // heads
        self.norm1 = nn.LayerNorm(width, eps=1e-5)
        self.qkv = nn.Linear(width, 3 * width)
        self.attention_output = nn.Linear(width, width)
        self.norm2 = nn.LayerNorm(width, eps=1e-5)
        self.linear1 = nn.Linear(width, ffn_width)
        self.linear2 = nn.Linear(ffn_width, width)

    def __call__(self, tokens):
        batch, sequence, _ = tokens.shape
        normalized = self.norm1(tokens)
        query, key, value = mx.split(self.qkv(normalized), 3, axis=-1)
        query = query.reshape(batch, sequence, self.heads, self.head_width).transpose(0, 2, 1, 3)
        key = key.reshape(batch, sequence, self.heads, self.head_width).transpose(0, 2, 1, 3)
        value = value.reshape(batch, sequence, self.heads, self.head_width).transpose(0, 2, 1, 3)
        weights = mx.softmax((query @ key.swapaxes(-1, -2)) / (self.head_width ** 0.5), axis=-1)
        attended = (weights @ value).transpose(0, 2, 1, 3).reshape(batch, sequence, self.width)
        tokens = tokens + self.attention_output(attended)
        return tokens + self.linear2(_gelu(self.linear1(self.norm2(tokens))))


class MLXTransformerCheckpoint(nn.Module):
    """Run a converted PyTorch Transformer checkpoint through MLX."""

    def __init__(self, config):
        super().__init__()
        self.config = dict(config)
        if self.config.get("family") != "transformer":
            raise ValueError("MLXTransformerCheckpoint requires a transformer checkpoint")
        width = int(self.config["embedding_dim"])
        layers = int(self.config["layers"])
        heads = int(self.config["attention_heads"])
        multiplier = int(self.config["ffn_multiplier"])
        if width % heads:
            raise ValueError("embedding_dim must be divisible by attention_heads")
        self.width = width
        self.count_encoding = self.config["count_encoding"]
        self.field_embeddings = nn.Embedding(len(NUMERIC_FIELDS) + len(BOOLEAN_FIELDS) + len(CATEGORICAL_FIELDS), width)
        self.type_embeddings = nn.Embedding(4, width)
        self.numeric_projection = nn.Linear(1, width, bias=False)
        self.boolean_embeddings = nn.Embedding(2, width)
        self.category_embeddings = {name: nn.Embedding(size, width)
                                    for name, size in CATEGORICAL_FIELDS.items()}
        self.count_buckets = {name: nn.Embedding(17, width) for name in NUMERIC_FIELDS[:6]} \
            if self.count_encoding == "bucket" else {}
        self.blocks = [_TransformerBlock(width, heads, width * multiplier) for _ in range(layers)]
        self.norm = nn.LayerNorm(width, eps=1e-5)
        self.choice = nn.Linear(width, 4)
        self.scores = {name: nn.Linear(width, 11) for name in (
            "current_resource_adequacy", "expected_gain_from_stronger_cognition",
            "decomposition_need", "evidence_sufficiency")}
        self.escalation_cost_coefficients = mx.zeros((2,), dtype=mx.float32)
        self.decomposition_sensitivity = mx.array(0.0, dtype=mx.float32)
        self.choice_temperature = 1.0

    def _encode(self, states: Sequence[CognitiveState]):
        batch = len(states)
        tokens = []
        field_id = 0
        for name in NUMERIC_FIELDS:
            if name in {"budget_pressure", "resource_scarcity", "active_contradictions"}:
                content = mx.zeros((batch, self.width), dtype=mx.float32)
            else:
                values = mx.array([[min(float(getattr(state, name)) / NUMERIC_MAX.get(name, 1.0), 1.0)]
                                   for state in states], dtype=mx.float32)
                if self.count_encoding == "bucket" and name in self.count_buckets:
                    ids = mx.round(values.squeeze(-1) * 16).astype(mx.int32)
                    content = self.count_buckets[name](ids)
                else:
                    content = self.numeric_projection(values)
            field = mx.broadcast_to(self.field_embeddings(mx.array(field_id)), (batch, self.width))
            type_value = mx.broadcast_to(self.type_embeddings(mx.array(0)), (batch, self.width))
            tokens.append(content + field + type_value)
            field_id += 1
        for name in BOOLEAN_FIELDS:
            ids = mx.array([int(getattr(state, name)) for state in states], dtype=mx.int32)
            field = mx.broadcast_to(self.field_embeddings(mx.array(field_id)), (batch, self.width))
            type_value = mx.broadcast_to(self.type_embeddings(mx.array(1)), (batch, self.width))
            tokens.append(self.boolean_embeddings(ids) + field + type_value)
            field_id += 1
        for name in CATEGORICAL_FIELDS:
            ids = mx.array([int(getattr(state, name)) for state in states], dtype=mx.int32)
            field = mx.broadcast_to(self.field_embeddings(mx.array(field_id)), (batch, self.width))
            type_value = mx.broadcast_to(self.type_embeddings(mx.array(2)), (batch, self.width))
            tokens.append(self.category_embeddings[name](ids) + field + type_value)
            field_id += 1
        return mx.stack(tokens, axis=1)

    def __call__(self, states: Sequence[CognitiveState]):
        tokens = self._encode(states)
        for block in self.blocks:
            tokens = block(tokens)
        representation = self.norm(tokens.mean(axis=1))
        choice_logits = self.choice(representation)
        costs = mx.array([[state.budget_pressure, state.resource_scarcity] for state in states], dtype=mx.float32)
        cost_coefficients = mx.log(1.0 + mx.exp(self.escalation_cost_coefficients))
        cost_penalty = (costs * cost_coefficients).sum(axis=-1)
        choice_logits = choice_logits - cost_penalty[:, None] * mx.array([0.0, 1.0, 0.0, 0.0])
        contradictions = mx.array([state.active_contradictions / 12.0 for state in states], dtype=mx.float32)
        decomp_bonus = mx.log(1.0 + mx.exp(self.decomposition_sensitivity)) * contradictions
        choice_logits = choice_logits + decomp_bonus[:, None] * mx.array([0.0, 0.0, 1.0, 0.0])
        choice_logits = choice_logits / max(float(self.choice_temperature), 1e-4)
        score_logits = {name: head(representation) for name, head in self.scores.items()}
        buckets = mx.arange(11, dtype=mx.float32) / 10.0
        score_logits["decomposition_need"] = score_logits["decomposition_need"] + decomp_bonus[:, None] * buckets[None, :]
        return {"choice_logits": choice_logits, "score_logits": score_logits}

    def load_torch_state_dict(self, state_dict):
        """Copy PyTorch tensors into matching MLX layers."""
        def array(key):
            return mx.array(state_dict[key].detach().cpu().tolist(), dtype=mx.float32)

        self.field_embeddings.weight = array("encoder.field_embeddings.weight")
        self.type_embeddings.weight = array("encoder.type_embeddings.weight")
        self.numeric_projection.weight = array("encoder.numeric_projection.weight")
        self.boolean_embeddings.weight = array("encoder.boolean_embeddings.weight")
        for name in CATEGORICAL_FIELDS:
            self.category_embeddings[name].weight = array(f"encoder.category_embeddings.{name}.weight")
        if self.count_encoding == "bucket":
            for name in self.count_buckets:
                self.count_buckets[name].weight = array(f"encoder.count_buckets.{name}.weight")
        for index, block in enumerate(self.blocks):
            prefix = f"backbone.layers.{index}."
            block.norm1.weight = array(prefix + "norm1.weight")
            block.norm1.bias = array(prefix + "norm1.bias")
            block.qkv.weight = array(prefix + "self_attn.in_proj_weight")
            block.qkv.bias = array(prefix + "self_attn.in_proj_bias")
            block.attention_output.weight = array(prefix + "self_attn.out_proj.weight")
            block.attention_output.bias = array(prefix + "self_attn.out_proj.bias")
            block.norm2.weight = array(prefix + "norm2.weight")
            block.norm2.bias = array(prefix + "norm2.bias")
            block.linear1.weight = array(prefix + "linear1.weight")
            block.linear1.bias = array(prefix + "linear1.bias")
            block.linear2.weight = array(prefix + "linear2.weight")
            block.linear2.bias = array(prefix + "linear2.bias")
        self.norm.weight = array("norm.weight")
        self.norm.bias = array("norm.bias")
        self.choice.weight = array("heads.choice.weight")
        self.choice.bias = array("heads.choice.bias")
        for name in self.scores:
            self.scores[name].weight = array(f"heads.scores.{name}.weight")
            self.scores[name].bias = array(f"heads.scores.{name}.bias")
        self.escalation_cost_coefficients = array("escalation_cost_coefficients")
        self.decomposition_sensitivity = array("decomposition_sensitivity")
        mx.eval(self.parameters())
        return self
