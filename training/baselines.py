"""Deterministic reference policies for synthetic-data comparison."""

from typing import Dict, Sequence
import random

import torch

from metis.schema import DecisionChoice, ProgressState
from .metrics import classification_metrics
from .synthetic import Example


def heuristic_choice(state):
    if state.progress == ProgressState.UNKNOWN and state.progress_evidence.value == 0 and state.uncertainty > 0.68:
        return DecisionChoice.ABSTAIN
    if state.active_contradictions >= 3 and state.open_unknowns >= 5:
        return DecisionChoice.DECOMPOSE
    if state.failed_attempts >= 3 and state.progress == ProgressState.LOW and state.progress_evidence.value >= 3:
        return DecisionChoice.ESCALATE
    if state.resource_scarcity + state.budget_pressure > 1.5:
        return DecisionChoice.CONTINUE
    return DecisionChoice.CONTINUE


def _regret(examples: Sequence[Example], choices: Sequence[int]) -> float:
    return sum(max(item.utilities.values()) - item.utilities[DecisionChoice(choice).name]
               for item, choice in zip(examples, choices)) / max(1, len(examples))


def evaluate_baselines(examples: Sequence[Example]) -> Dict[str, object]:
    labels = torch.tensor([example.choice for example in examples])
    policies = {
        "always_continue": [int(DecisionChoice.CONTINUE)] * len(examples),
        "heuristic": [int(heuristic_choice(example.state)) for example in examples],
    }
    result = {}
    for name, choices in policies.items():
        logits = torch.nn.functional.one_hot(torch.tensor(choices), 4).float() * 8.0
        result[name] = {"decision": classification_metrics(logits, labels), "regret": _regret(examples, choices)}
    return result


def compare_learned_to_baselines(examples: Sequence[Example], learned_choices: Sequence[int],
                                seed: int = 31, resamples: int = 1000) -> Dict[str, object]:
    """Paired bootstrap intervals for accuracy and regret differences on held-out rows."""
    labels = [example.choice for example in examples]
    learned_correct = [int(int(choice) == label) for choice, label in zip(learned_choices, labels)]
    learned_regrets = [max(item.utilities.values()) - item.utilities[DecisionChoice(choice).name]
                       for item, choice in zip(examples, learned_choices)]
    policies = {
        "always_continue": [int(DecisionChoice.CONTINUE)] * len(examples),
        "heuristic": [int(heuristic_choice(item.state)) for item in examples],
    }
    rng = random.Random(seed)
    output = {}
    for name, choices in policies.items():
        accuracy_delta = [learned_ok - int(choice == label)
                          for learned_ok, choice, label in zip(learned_correct, choices, labels)]
        regret_delta = [max(item.utilities.values()) - item.utilities[DecisionChoice(choice).name] - learned_regret
                        for item, choice, learned_regret in zip(examples, choices, learned_regrets)]
        accuracy_samples, regret_samples = [], []
        for _ in range(resamples):
            indices = [rng.randrange(len(examples)) for _ in examples]
            accuracy_samples.append(sum(accuracy_delta[index] for index in indices) / len(indices))
            regret_samples.append(sum(regret_delta[index] for index in indices) / len(indices))
        accuracy_samples.sort()
        regret_samples.sort()
        output[name] = {
            "accuracy_delta_learned_minus_baseline": sum(accuracy_delta) / len(examples),
            "accuracy_delta_95pct_ci": [accuracy_samples[int(.025 * resamples)], accuracy_samples[min(resamples - 1, int(.975 * resamples))]],
            "regret_reduction_baseline_minus_learned": sum(regret_delta) / len(examples),
            "regret_reduction_95pct_ci": [regret_samples[int(.025 * resamples)], regret_samples[min(resamples - 1, int(.975 * resamples))]],
            "resamples": resamples,
        }
    return output
