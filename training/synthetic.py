"""Deterministic synthetic decision environment with latent utilities."""

from dataclasses import dataclass
import random
from typing import Dict, List, Sequence, Union

import torch

from metis.schema import (
    CognitiveState, DecisionChoice, EvidenceStrength, Phase, ProgressState, ResourceTier,
)


@dataclass(frozen=True)
class Example:
    state: CognitiveState
    choice: int
    scores: Dict[str, float]
    utilities: Dict[str, float]


def _clip(value: float) -> float:
    return min(1.0, max(0.0, value))


def utility_model(state: CognitiveState, gain: float, decomposition_need: float,
                  adequacy: float, evidence_sufficiency: float) -> Dict[str, float]:
    """Expected latent utility for each action; labels depend on latent conditions."""
    evidence_gap = 1.0 - evidence_sufficiency
    progress_signal = 0.0 if state.progress == ProgressState.UNKNOWN else (
        -0.18 if state.progress == ProgressState.LOW else 0.12
    )
    continue_u = 0.52 * adequacy + progress_signal - 0.22 * state.uncertainty - 0.06 * state.phase_attempts
    escalate_u = 0.62 * gain - 0.23 * state.resource_scarcity - 0.20 * state.budget_pressure
    decompose_u = 0.65 * decomposition_need - 0.13 * state.budget_pressure + 0.08 * state.open_unknowns / 20.0
    abstain_u = 0.55 * evidence_gap - 0.12 * (1.0 - state.uncertainty)
    if state.phase == Phase.CLOSE:
        continue_u += 0.1 * state.artifact_accepted
    return {
        "CONTINUE": continue_u, "ESCALATE": escalate_u,
        "DECOMPOSE": decompose_u, "ABSTAIN": abstain_u,
    }


def generate_example(rng: random.Random) -> Example:
    difficulty = rng.random()
    capability = rng.random()
    match = 1.0 - abs(difficulty - capability)
    gain = _clip(0.12 + 0.72 * max(0.0, difficulty - capability) + rng.gauss(0, 0.10))
    decomposition_need = _clip(0.08 + 0.72 * difficulty * rng.random() + 0.22 * rng.random())
    evidence_sufficiency = _clip(0.12 + 0.78 * rng.random())
    evidence_reliability = _clip(evidence_sufficiency * rng.uniform(0.55, 1.0))
    budget = rng.random()
    scarcity = rng.random()

    # Observations are noisy and intentionally do not reveal latent variables directly.
    attempts = min(12, max(0, round(1 + 6 * difficulty + rng.gauss(0, 1.8))))
    failed = min(attempts, max(0, round(attempts * (difficulty - capability + 0.5) + rng.gauss(0, 1.3))))
    unknown_progress = rng.random() < (0.10 + 0.55 * (1.0 - evidence_reliability))
    if unknown_progress:
        progress = ProgressState.UNKNOWN
        progress_evidence = EvidenceStrength.NONE
    else:
        observed_progress = _clip(match + rng.gauss(0, 0.35))
        progress = ProgressState.MEANINGFUL if observed_progress > 0.55 else ProgressState.LOW
        level = evidence_reliability
        progress_evidence = (EvidenceStrength.VERIFIED if level > 0.76 else
                             EvidenceStrength.INFERRED if level > 0.52 else EvidenceStrength.REPORTED)
    valid = rng.random() < _clip(0.9 - 0.48 * difficulty + 0.2 * capability)
    accepted = valid and rng.random() < _clip(0.25 + 0.52 * capability - 0.3 * difficulty)
    tier = max(0, min(3, int(capability * 4) + rng.choice((-1, 0, 0, 0, 1))))
    state = CognitiveState(
        phase=rng.choice(list(Phase)), current_resource_tier=ResourceTier(tier),
        phase_attempts=attempts, consecutive_phase_attempts=min(attempts, rng.randint(0, max(0, attempts))),
        failed_attempts=failed, reopen_count=rng.randint(0, min(5, attempts)),
        progress=progress, progress_evidence=progress_evidence, artifact_valid=valid,
        artifact_accepted=accepted, open_unknowns=min(20, max(0, round(20 * (1.0 - evidence_sufficiency) + rng.gauss(0, 2)))),
        active_contradictions=min(12, max(0, round(8 * difficulty * (1.0 - evidence_reliability) + rng.gauss(0, 1.5)))),
        has_leading_hypothesis=rng.random() < _clip(0.25 + capability * 0.55),
        has_active_frame=rng.random() < _clip(0.2 + decomposition_need * 0.5),
        uncertainty=_clip(0.72 * (1.0 - evidence_sufficiency) + 0.25 * rng.random()),
        novelty=_clip(0.62 * difficulty + 0.38 * rng.random()),
        budget_pressure=budget, resource_scarcity=scarcity,
    )
    utilities = utility_model(state, gain, decomposition_need, match, evidence_sufficiency)
    # Tie-breaking is stable by enum order for reproducible generated data.
    choice = max(DecisionChoice, key=lambda item: utilities[item.name])
    return Example(state, int(choice), {
        "current_resource_adequacy": match,
        "expected_gain_from_stronger_cognition": gain,
        "decomposition_need": decomposition_need,
        "evidence_sufficiency": evidence_sufficiency,
    }, utilities)


def _designed_counterexamples() -> List[Example]:
    """Cases that deliberately break attempt-count and acceptance shortcuts."""
    base = CognitiveState(
        phase=Phase.EXECUTE, current_resource_tier=ResourceTier.MEDIUM,
        phase_attempts=3, consecutive_phase_attempts=3, failed_attempts=3, reopen_count=0,
        progress=ProgressState.LOW, progress_evidence=EvidenceStrength.VERIFIED,
        artifact_valid=True, artifact_accepted=False, open_unknowns=4, active_contradictions=1,
        has_leading_hypothesis=True, has_active_frame=False, uncertainty=0.25, novelty=0.6,
        budget_pressure=0.1, resource_scarcity=0.1,
    )
    scenarios = [
        # Same three failures; stronger cognition has high expected value.
        (base, 0.95, 0.12, 0.2, 0.9),
        # Unknown is missing information, not observed low progress.
        (replace_state(base, progress=ProgressState.UNKNOWN, progress_evidence=EvidenceStrength.NONE,
            uncertainty=0.8, failed_attempts=3), 0.12, 0.12, 0.5, 0.18),
        # Decomposition dominates even after repeated failures.
        (replace_state(base, active_contradictions=5, open_unknowns=12), 0.25, 0.98, 0.2, 0.7),
        # Severe budget pressure and scarcity can make continuing preferable.
        (replace_state(base, progress=ProgressState.MEANINGFUL, progress_evidence=EvidenceStrength.VERIFIED,
            budget_pressure=1.0, resource_scarcity=0.9, phase_attempts=3), 0.88, 0.12, 0.9, 0.85),
        # One attempt can be enough when evidence of resource mismatch is strong.
        (replace_state(base, phase_attempts=1, consecutive_phase_attempts=1, failed_attempts=1,
            progress=ProgressState.LOW, budget_pressure=0.05, resource_scarcity=0.05), 0.98, 0.12, 0.08, 0.9),
        # Accepted artifacts are compatible with verified low progress.
        (replace_state(base, artifact_accepted=True, progress=ProgressState.LOW,
            progress_evidence=EvidenceStrength.VERIFIED), 0.86, 0.2, 0.22, 0.88),
    ]
    examples = []
    for state, gain, decompose, adequacy, evidence in scenarios:
        utilities = utility_model(state, gain, decompose, adequacy, evidence)
        choice = max(DecisionChoice, key=lambda item: utilities[item.name])
        examples.append(Example(state, int(choice), {
            "current_resource_adequacy": adequacy,
            "expected_gain_from_stronger_cognition": gain,
            "decomposition_need": decompose,
            "evidence_sufficiency": evidence,
        }, utilities))
    return examples


def replace_state(state: CognitiveState, **changes) -> CognitiveState:
    from dataclasses import replace
    return replace(state, **changes)


def generate_dataset(size: int, seed: int = 0) -> List[Example]:
    if size < 1:
        raise ValueError("size must be positive")
    rng = random.Random(seed)
    designed = _designed_counterexamples()
    # Reserve a small fixed share for counterexamples, shuffled with generated cases.
    count = min(len(designed), max(1, size // 50))
    items = [generate_example(rng) for _ in range(size - count)] + designed[:count]
    rng.shuffle(items)
    return items


def score_bucket(value: float) -> int:
    return min(10, max(0, round(value * 10)))


def collate(examples: Sequence[Example], device: Union[torch.device, str] = "cpu"):
    return {
        "states": [item.state for item in examples],
        "choices": torch.tensor([item.choice for item in examples], dtype=torch.long, device=device),
        "scores": {
            name: torch.tensor([score_bucket(item.scores[name]) for item in examples], dtype=torch.long, device=device)
            for name in examples[0].scores
        },
    }
