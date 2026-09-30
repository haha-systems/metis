import pytest

from metis.schema import CognitiveState, EvidenceStrength, Phase, ProgressState, ResourceTier
from benchmarks.ttd import sample_state


def test_schema_round_trip_uses_explicit_enum_names():
    state = sample_state()
    restored = CognitiveState.from_dict(state.to_dict())
    assert restored == state


def test_accepted_artifact_must_be_valid():
    state = sample_state()
    with pytest.raises(ValueError):
        CognitiveState(**{**state.__dict__, "artifact_valid": False, "artifact_accepted": True})

