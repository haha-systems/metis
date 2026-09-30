"""Single-pass inference and output decoding."""

from dataclasses import dataclass
import math
from typing import Dict, Mapping, Sequence

import torch

from .distributions import expected_value, entropy, probabilities
from .heads import SCORE_NAMES
from .model import MetisModel
from .schema import CognitiveState, DecisionChoice


@dataclass(frozen=True)
class Decision:
    choice: DecisionChoice
    choice_probabilities: Mapping[str, float]
    scores: Mapping[str, Mapping[str, object]]
    uncertainty: float


class MetisPredictor:
    def __init__(self, model: MetisModel, device: str = "cpu") -> None:
        self.device = torch.device(device)
        self.model = model.to(self.device).eval()

    @torch.inference_mode()
    def predict_batch(self, states: Sequence[CognitiveState]) -> Sequence[Decision]:
        return self.decode_outputs(self.model(states))

    @torch.inference_mode()
    def decode_outputs(self, outputs) -> Sequence[Decision]:
        temperature = max(float(self.model.choice_temperature), 1e-4)
        choices = torch.softmax(outputs["choice_logits"] / temperature, dim=-1)
        score_logits = outputs["score_logits"]
        score_probs = {name: probabilities(score_logits[name]) for name in SCORE_NAMES}
        score_means = {name: expected_value(score_logits[name]) for name in SCORE_NAMES}
        score_entropies = {name: entropy(score_logits[name]) for name in SCORE_NAMES}
        choices_cpu = choices.cpu().tolist()
        score_probs_cpu = {name: values.cpu().tolist() for name, values in score_probs.items()}
        score_means_cpu = {name: values.cpu().tolist() for name, values in score_means.items()}
        score_entropies_cpu = {name: values.cpu().tolist() for name, values in score_entropies.items()}
        result = []
        for row, choice_row in enumerate(choices_cpu):
            scores: Dict[str, Dict[str, object]] = {}
            for name in SCORE_NAMES:
                scores[name] = {
                    "mean": float(score_means_cpu[name][row]),
                    "entropy": float(score_entropies_cpu[name][row]),
                    "probabilities": [float(value) for value in score_probs_cpu[name][row]],
                }
            clipped = [max(float(value), 1e-12) for value in choice_row]
            uncertainty = -sum(value * math.log(value) for value in clipped) / math.log(len(DecisionChoice))
            result.append(Decision(
                choice=DecisionChoice(max(range(len(choice_row)), key=choice_row.__getitem__)),
                choice_probabilities={member.name.lower(): float(choice_row[int(member)]) for member in DecisionChoice},
                scores=scores, uncertainty=float(uncertainty),
            ))
        return result

    def predict(self, state: CognitiveState) -> Decision:
        return self.predict_batch([state])[0]


def predict(model: MetisModel, state: CognitiveState, device: str = "cpu") -> Decision:
    return MetisPredictor(model, device).predict(state)
