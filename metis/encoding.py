"""Typed field encoding. UNKNOWN progress remains a distinct categorical value."""

from typing import List, Sequence

import torch
from torch import nn

from .schema import CognitiveState, EvidenceStrength, Phase, ProgressState, ResourceTier


NUMERIC_FIELDS = (
    "phase_attempts", "consecutive_phase_attempts", "failed_attempts", "reopen_count",
    "open_unknowns", "active_contradictions", "uncertainty", "novelty",
    "budget_pressure", "resource_scarcity",
)
BOOLEAN_FIELDS = ("artifact_valid", "artifact_accepted", "has_leading_hypothesis", "has_active_frame")
CATEGORICAL_FIELDS = {
    "phase": len(Phase), "current_resource_tier": len(ResourceTier),
    "progress": len(ProgressState), "progress_evidence": len(EvidenceStrength),
}
NUMERIC_MAX = {
    "phase_attempts": 12.0, "consecutive_phase_attempts": 8.0,
    "failed_attempts": 10.0, "reopen_count": 8.0, "open_unknowns": 20.0,
    "active_contradictions": 12.0,
}
TOKEN_COUNT = len(NUMERIC_FIELDS) + len(BOOLEAN_FIELDS) + len(CATEGORICAL_FIELDS)


class StateEncoder(nn.Module):
    """Encode each typed field into one token with identity and type information."""

    def __init__(self, embedding_dim: int = 64, count_encoding: str = "scalar") -> None:
        super().__init__()
        if count_encoding not in {"scalar", "bucket"}:
            raise ValueError("count_encoding must be 'scalar' or 'bucket'")
        self.embedding_dim = embedding_dim
        self.count_encoding = count_encoding
        self.field_embeddings = nn.Embedding(TOKEN_COUNT, embedding_dim)
        self.type_embeddings = nn.Embedding(4, embedding_dim)
        self.numeric_projection = nn.Linear(1, embedding_dim, bias=False)
        self.boolean_embeddings = nn.Embedding(2, embedding_dim)
        self.category_embeddings = nn.ModuleDict({
            key: nn.Embedding(size, embedding_dim) for key, size in CATEGORICAL_FIELDS.items()
        })
        self.count_buckets = nn.ModuleDict({
            key: nn.Embedding(17, embedding_dim) for key in NUMERIC_FIELDS[:6]
        }) if count_encoding == "bucket" else nn.ModuleDict()

    def forward(self, states: Sequence[CognitiveState], ignored_numeric_fields=()) -> torch.Tensor:
        device = self.field_embeddings.weight.device
        batch = len(states)
        tokens: List[torch.Tensor] = []
        field_id = 0
        for name in NUMERIC_FIELDS:
            if name in ignored_numeric_fields:
                content = torch.zeros((batch, self.embedding_dim), dtype=torch.float32, device=device)
            else:
                values = torch.tensor([
                    min(float(getattr(state, name)) / NUMERIC_MAX.get(name, 1.0), 1.0)
                    for state in states
                ], dtype=torch.float32, device=device).unsqueeze(-1)
                if self.count_encoding == "bucket" and name in self.count_buckets:
                    ids = (values.squeeze(-1) * 16).round().long()
                    content = self.count_buckets[name](ids)
                else:
                    content = self.numeric_projection(values)
            tokens.append(content + self.field_embeddings.weight[field_id] + self.type_embeddings.weight[0])
            field_id += 1
        for name in BOOLEAN_FIELDS:
            ids = torch.tensor([int(getattr(state, name)) for state in states], device=device)
            tokens.append(self.boolean_embeddings(ids) + self.field_embeddings.weight[field_id] + self.type_embeddings.weight[1])
            field_id += 1
        for name in CATEGORICAL_FIELDS:
            ids = torch.tensor([int(getattr(state, name)) for state in states], device=device)
            tokens.append(self.category_embeddings[name](ids) + self.field_embeddings.weight[field_id] + self.type_embeddings.weight[2])
            field_id += 1
        if batch == 0:
            return torch.empty((0, TOKEN_COUNT, self.embedding_dim), device=device)
        return torch.stack(tokens, dim=1)
