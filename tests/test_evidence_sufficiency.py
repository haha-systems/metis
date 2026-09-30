from dataclasses import replace

from benchmarks.ttd import sample_state
from metis.schema import EvidenceStrength, ProgressState
from training.synthetic import utility_model


def test_removing_evidence_increases_ground_truth_abstention_utility():
    state = sample_state()
    before = utility_model(state, .5, .3, .6, .8)["ABSTAIN"]
    weaker = replace(state, progress=ProgressState.UNKNOWN, progress_evidence=EvidenceStrength.NONE,
                     uncertainty=min(1.0, state.uncertainty + .2))
    after = utility_model(weaker, .5, .3, .6, .5)["ABSTAIN"]
    assert after > before

