"""Multi-task heads for bounded decisions and distributional scores."""

from typing import Dict

from torch import nn

from .schema import DecisionChoice

SCORE_NAMES = ("current_resource_adequacy", "expected_gain_from_stronger_cognition",
               "decomposition_need", "evidence_sufficiency")


class DecisionHeads(nn.Module):
    def __init__(self, hidden_dim: int) -> None:
        super().__init__()
        self.choice = nn.Linear(hidden_dim, len(DecisionChoice))
        self.scores = nn.ModuleDict({name: nn.Linear(hidden_dim, 11) for name in SCORE_NAMES})

    def forward(self, representation) -> Dict[str, object]:
        return {"choice_logits": self.choice(representation),
                "score_logits": {name: head(representation) for name, head in self.scores.items()}}

