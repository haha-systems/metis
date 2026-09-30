from metis.heads import SCORE_NAMES
from metis.model import ModelConfig, build_model
from metis.schema import DecisionChoice
from benchmarks.ttd import sample_state


def test_model_has_one_decision_and_distributional_score_heads():
    model = build_model(ModelConfig(family="mlp", embedding_dim=16, ffn_multiplier=2))
    output = model([sample_state()])
    assert output["choice_logits"].shape == (1, len(DecisionChoice))
    assert set(output["score_logits"]) == set(SCORE_NAMES)
    assert all(value.shape == (1, 11) for value in output["score_logits"].values())

