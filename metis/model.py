"""Small non-autoregressive model families used in the baseline sweep."""

from dataclasses import asdict, dataclass
from typing import Dict, Sequence

import torch
from torch import nn
import torch.nn.functional as F

from .encoding import StateEncoder, TOKEN_COUNT
from .heads import DecisionHeads
from .schema import CognitiveState
from .schema import DecisionChoice


@dataclass(frozen=True)
class ModelConfig:
    family: str = "transformer"
    embedding_dim: int = 64
    layers: int = 2
    attention_heads: int = 4
    ffn_multiplier: int = 2
    count_encoding: str = "scalar"

    def __post_init__(self) -> None:
        if self.family not in {"linear", "mlp", "transformer", "mixer"}:
            raise ValueError("family must be linear, mlp, transformer, or mixer")
        if self.embedding_dim < 8 or self.layers < 1:
            raise ValueError("embedding_dim must be >= 8 and layers must be positive")
        if self.attention_heads < 1 or self.ffn_multiplier < 1:
            raise ValueError("attention_heads and ffn_multiplier must be positive")
        if self.family == "transformer" and self.embedding_dim % self.attention_heads:
            raise ValueError("embedding_dim must be divisible by attention_heads")


class MetisModel(nn.Module):
    def __init__(self, config: ModelConfig = ModelConfig()) -> None:
        super().__init__()
        self.config = config
        self.choice_temperature = 1.0
        self.encoder = StateEncoder(config.embedding_dim, config.count_encoding)
        width = config.embedding_dim
        if config.family == "linear":
            self.backbone = nn.Sequential(nn.Flatten(), nn.Linear(TOKEN_COUNT * width, width))
        elif config.family == "mlp":
            self.backbone = nn.Sequential(
                nn.Flatten(), nn.Linear(TOKEN_COUNT * width, width * config.ffn_multiplier), nn.GELU(),
                nn.Linear(width * config.ffn_multiplier, width), nn.GELU(),
            )
        elif config.family == "transformer":
            layer = nn.TransformerEncoderLayer(
                d_model=width, nhead=config.attention_heads,
                dim_feedforward=width * config.ffn_multiplier, batch_first=True,
                activation="gelu", norm_first=True,
            )
            self.backbone = nn.TransformerEncoder(layer, num_layers=config.layers, enable_nested_tensor=False)
        else:
            self.backbone = nn.ModuleList([
                nn.Sequential(nn.LayerNorm(TOKEN_COUNT), nn.Linear(TOKEN_COUNT, TOKEN_COUNT * config.ffn_multiplier),
                              nn.GELU(), nn.Linear(TOKEN_COUNT * config.ffn_multiplier, TOKEN_COUNT))
                for _ in range(config.layers)
            ])
            self.mixer_channel = nn.ModuleList([
                nn.Sequential(nn.LayerNorm(width), nn.Linear(width, width * config.ffn_multiplier), nn.GELU(),
                              nn.Linear(width * config.ffn_multiplier, width)) for _ in range(config.layers)
            ])
        self.norm = nn.Identity() if config.family == "linear" else nn.LayerNorm(width)
        self.heads = DecisionHeads(width)
        # Nonnegative learned costs provide a monotone path to the ESCALATE logit.
        self.escalation_cost_coefficients = nn.Parameter(torch.full((2,), -2.0))
        self.decomposition_sensitivity = nn.Parameter(torch.tensor(-2.0))

    def forward_encoded(self, tokens: torch.Tensor, states: Sequence[CognitiveState] = None) -> Dict[str, object]:
        if self.config.family == "transformer":
            tokens = self.backbone(tokens)
        elif self.config.family == "mixer":
            for token_mixing, channel_mixing in zip(self.backbone, self.mixer_channel):
                tokens = tokens + token_mixing(tokens.transpose(1, 2)).transpose(1, 2)
                tokens = tokens + channel_mixing(tokens)
        representation = self.norm(tokens.mean(dim=1))
        outputs = self.heads(representation)
        if states is not None:
            costs = torch.tensor([[state.budget_pressure, state.resource_scarcity] for state in states],
                                 dtype=representation.dtype, device=representation.device)
            penalty = (costs * F.softplus(self.escalation_cost_coefficients)).sum(dim=-1)
            escalation_selector = F.one_hot(
                torch.full((len(states),), int(DecisionChoice.ESCALATE), device=representation.device),
                num_classes=len(DecisionChoice)).to(representation.dtype)
            outputs["choice_logits"] = outputs["choice_logits"] - escalation_selector * penalty.unsqueeze(-1)
            contradictions = torch.tensor([state.active_contradictions / 12.0 for state in states],
                                         dtype=representation.dtype, device=representation.device)
            decomp_bonus = F.softplus(self.decomposition_sensitivity) * contradictions
            decomp_selector = F.one_hot(
                torch.full((len(states),), 2, device=representation.device),
                num_classes=4).to(representation.dtype)
            outputs["choice_logits"] = outputs["choice_logits"] + decomp_selector * decomp_bonus.unsqueeze(-1)
            score_logits = outputs["score_logits"]["decomposition_need"]
            buckets = torch.arange(11, dtype=representation.dtype, device=representation.device) / 10.0
            outputs["score_logits"]["decomposition_need"] = score_logits + (
                decomp_bonus.unsqueeze(-1) * buckets.unsqueeze(0))
        return outputs

    def forward(self, states: Sequence[CognitiveState]) -> Dict[str, object]:
        tokens = self.encoder(states, ignored_numeric_fields={
            "budget_pressure", "resource_scarcity", "active_contradictions"})
        return self.forward_encoded(tokens, states)

    def parameter_count(self) -> int:
        return sum(parameter.numel() for parameter in self.parameters())

    def config_dict(self) -> Dict[str, object]:
        return asdict(self.config)


def build_model(config: ModelConfig = ModelConfig()) -> MetisModel:
    return MetisModel(config)
