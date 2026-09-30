from metis.inference import MetisPredictor
from metis.model import ModelConfig, build_model
from metis.heads import SCORE_NAMES
from benchmarks.ttd import sample_state


def test_inference_returns_normalized_choice_and_score_distributions():
    predictor = MetisPredictor(build_model(ModelConfig(family="mlp", embedding_dim=16)))
    decision = predictor.predict(sample_state())
    assert abs(sum(decision.choice_probabilities.values()) - 1.0) < 1e-6
    assert set(decision.scores) == set(SCORE_NAMES)
    assert all(abs(sum(score["probabilities"]) - 1.0) < 1e-6 for score in decision.scores.values())
