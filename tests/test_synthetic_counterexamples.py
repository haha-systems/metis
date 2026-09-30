from metis.schema import DecisionChoice, EvidenceStrength, ProgressState
from training.synthetic import _designed_counterexamples, generate_dataset


def test_designed_cases_defeat_attempt_count_and_acceptance_shortcuts():
    escalate, unknown, decompose, costly_continue, early_escalate, accepted_low = _designed_counterexamples()

    assert escalate.state.failed_attempts == 3
    assert escalate.state.progress == ProgressState.LOW
    assert DecisionChoice(escalate.choice) == DecisionChoice.ESCALATE

    assert unknown.state.failed_attempts == 3
    assert unknown.state.progress == ProgressState.UNKNOWN
    assert unknown.state.progress_evidence == EvidenceStrength.NONE
    assert DecisionChoice(unknown.choice) in {DecisionChoice.ABSTAIN, DecisionChoice.CONTINUE}

    assert decompose.state.failed_attempts == 3
    assert DecisionChoice(decompose.choice) == DecisionChoice.DECOMPOSE

    assert costly_continue.state.failed_attempts == 3
    assert costly_continue.state.budget_pressure == 1.0
    assert DecisionChoice(costly_continue.choice) == DecisionChoice.CONTINUE

    assert early_escalate.state.phase_attempts == 1
    assert early_escalate.state.failed_attempts == 1
    assert DecisionChoice(early_escalate.choice) == DecisionChoice.ESCALATE

    assert accepted_low.state.artifact_valid and accepted_low.state.artifact_accepted
    assert accepted_low.state.progress == ProgressState.LOW
    assert accepted_low.state.progress_evidence == EvidenceStrength.VERIFIED

    observed = {example.state for example in generate_dataset(1000, seed=7)}
    assert all(example.state in observed for example in
               (escalate, unknown, decompose, costly_continue, early_escalate, accepted_low))
