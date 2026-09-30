"""Paired counterfactual generators and diagnostics for learned response shape."""

from dataclasses import replace
from typing import Dict, Sequence

from metis.inference import MetisPredictor
from metis.schema import CognitiveState, EvidenceStrength, ProgressState
from metis.schema import ResourceTier


def counterfactual_pairs(state: CognitiveState):
    """Return named paired states changing one decision-relevant factor at a time."""
    return {
        "higher_cost": (state, replace(state, budget_pressure=min(1.0, state.budget_pressure + 0.25))),
        "higher_decomposition_need": (state, replace(state,
            active_contradictions=min(12, state.active_contradictions + 3))),
        "unknown_to_verified_low": (state, replace(state, progress=ProgressState.LOW,
            progress_evidence=EvidenceStrength.VERIFIED)),
        "lower_resource_tier": (state, replace(state, current_resource_tier=ResourceTier(
            max(int(ResourceTier.LOW), int(state.current_resource_tier) - 1)))),
        "remove_evidence": (state, replace(state, progress=ProgressState.UNKNOWN,
            progress_evidence=EvidenceStrength.NONE, uncertainty=min(1.0, state.uncertainty + 0.25))),
    }


def check_metamorphic(predictor: MetisPredictor, states: Sequence[CognitiveState]) -> Dict[str, object]:
    """Score directional invariants. Results are diagnostics, not guarantees per sample."""
    checks = {"higher_cost_escalation_nonincrease": [], "higher_contradictions_decompose_non_decrease": [],
              "evidence_removal_abstention_or_uncertainty": [], "progress_change_material": []}
    gain_checks = []
    for state in states:
        pairs = counterfactual_pairs(state)
        original = predictor.predict(pairs["higher_cost"][0])
        costly = predictor.predict(pairs["higher_cost"][1])
        checks["higher_cost_escalation_nonincrease"].append(
            costly.choice_probabilities["escalate"] <= original.choice_probabilities["escalate"] + 1e-6)
        base, more_need = pairs["higher_decomposition_need"]
        p0, p1 = predictor.predict(base), predictor.predict(more_need)
        checks["higher_contradictions_decompose_non_decrease"].append(
            p1.choice_probabilities["decompose"] >= p0.choice_probabilities["decompose"] - 1e-6)
        base, less_evidence = pairs["remove_evidence"]
        p0, p1 = predictor.predict(base), predictor.predict(less_evidence)
        checks["evidence_removal_abstention_or_uncertainty"].append(
            p1.choice_probabilities["abstain"] >= p0.choice_probabilities["abstain"] - 1e-6 or
            p1.uncertainty >= p0.uncertainty - 1e-6)
        base, low = pairs["unknown_to_verified_low"]
        p0, p1 = predictor.predict(base), predictor.predict(low)
        checks["progress_change_material"].append(
            max(abs(p1.choice_probabilities[k] - p0.choice_probabilities[k]) for k in p0.choice_probabilities) > 0.02)
        base, lower_resource = pairs["lower_resource_tier"]
        p0, p1 = predictor.predict(base), predictor.predict(lower_resource)
        gain0 = p0.scores["expected_gain_from_stronger_cognition"]["mean"]
        gain1 = p1.scores["expected_gain_from_stronger_cognition"]["mean"]
        if gain1 > gain0 + 1e-6:
            gain_checks.append(p1.choice_probabilities["escalate"] >= p0.choice_probabilities["escalate"] - 1e-6)
    checks["higher_predicted_gain_escalation_non_decrease"] = gain_checks
    return {name: {"pass_rate": sum(values) / max(1, len(values)), "passed": sum(values), "total": len(values)}
            for name, values in checks.items()}
