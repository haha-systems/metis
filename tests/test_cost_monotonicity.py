from dataclasses import replace

from metis.schema import CognitiveState
from training.synthetic import utility_model
from benchmarks.ttd import sample_state
from metis.inference import MetisPredictor
from metis.model import ModelConfig, build_model


def test_higher_escalation_cost_cannot_raise_ground_truth_escalation_utility():
    state = sample_state()
    kwargs = (0.8, 0.2, 0.4, 0.7)
    before = utility_model(state, *kwargs)["ESCALATE"]
    after = utility_model(replace(state, budget_pressure=min(1, state.budget_pressure + .2)), *kwargs)["ESCALATE"]
    assert after <= before


def test_model_escalation_probability_is_monotone_in_budget_and_scarcity():
    state = sample_state()
    predictor = MetisPredictor(build_model(ModelConfig(family="mlp", embedding_dim=16)))
    original = predictor.predict(state).choice_probabilities["escalate"]
    higher_budget = predictor.predict(replace(state, budget_pressure=min(1.0, state.budget_pressure + .3)))
    higher_scarcity = predictor.predict(replace(state, resource_scarcity=min(1.0, state.resource_scarcity + .3)))
    assert higher_budget.choice_probabilities["escalate"] <= original + 1e-7
    assert higher_scarcity.choice_probabilities["escalate"] <= original + 1e-7
