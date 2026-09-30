import torch

from metis.encoding import StateEncoder
from benchmarks.ttd import sample_state
from metis.schema import EvidenceStrength, ProgressState


def test_unknown_progress_has_distinct_encoded_token():
    encoder = StateEncoder(embedding_dim=16)
    low = sample_state()
    unknown = type(low)(**{**low.__dict__, "progress": ProgressState.UNKNOWN,
                            "progress_evidence": EvidenceStrength.NONE})
    assert not torch.equal(encoder([low]), encoder([unknown]))

