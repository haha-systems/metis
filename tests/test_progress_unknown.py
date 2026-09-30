from metis.schema import EvidenceStrength, ProgressState
from training.synthetic import _designed_counterexamples


def test_designed_unknown_progress_case_keeps_evidence_missing():
    cases = _designed_counterexamples()
    unknown = [item for item in cases if item.state.progress == ProgressState.UNKNOWN]
    assert unknown
    assert all(item.state.progress_evidence == EvidenceStrength.NONE for item in unknown)

