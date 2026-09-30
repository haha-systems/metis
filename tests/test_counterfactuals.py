from training.synthetic import _designed_counterexamples
from metis.schema import DecisionChoice, ProgressState
from dataclasses import replace
from metis.inference import MetisPredictor
from metis.model import ModelConfig, build_model
from benchmarks.ttd import sample_state


def test_designed_counterexamples_cover_three_failures_with_distinct_outcomes():
    cases = _designed_counterexamples()
    repeated = [item for item in cases if item.state.phase_attempts == 3]
    choices = {DecisionChoice(item.choice) for item in repeated}
    assert {DecisionChoice.ESCALATE, DecisionChoice.DECOMPOSE}.issubset(choices)
    assert any(item.state.progress == ProgressState.UNKNOWN for item in repeated)


def test_model_decomposition_probability_is_monotone_in_contradictions():
    state = sample_state()
    predictor = MetisPredictor(build_model(ModelConfig(family="transformer", embedding_dim=16)))
    original = predictor.predict(state)
    more_contradictions = predictor.predict(replace(state, active_contradictions=min(12, state.active_contradictions + 3)))
    assert more_contradictions.choice_probabilities["decompose"] >= original.choice_probabilities["decompose"] - 1e-7
    assert more_contradictions.scores["decomposition_need"]["mean"] >= original.scores["decomposition_need"]["mean"] - 1e-7
