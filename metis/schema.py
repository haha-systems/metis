"""Fixed, versioned input and output vocabulary for Metis v0.1."""

from dataclasses import asdict, dataclass
from enum import IntEnum
import math
from typing import Any, Dict


class Phase(IntEnum):
    TRIAGE = 0
    ABDUCE = 1
    FRAME = 2
    EXECUTE = 3
    CLOSE = 4


class ResourceTier(IntEnum):
    LOW = 0
    MEDIUM = 1
    HIGH = 2
    VERY_HIGH = 3


class ProgressState(IntEnum):
    UNKNOWN = 0
    LOW = 1
    MEANINGFUL = 2


class EvidenceStrength(IntEnum):
    NONE = 0
    REPORTED = 1
    INFERRED = 2
    VERIFIED = 3


class DecisionChoice(IntEnum):
    CONTINUE = 0
    ESCALATE = 1
    DECOMPOSE = 2
    ABSTAIN = 3


@dataclass(frozen=True)
class CognitiveState:
    phase: Phase
    current_resource_tier: ResourceTier
    phase_attempts: int
    consecutive_phase_attempts: int
    failed_attempts: int
    reopen_count: int
    progress: ProgressState
    progress_evidence: EvidenceStrength
    artifact_valid: bool
    artifact_accepted: bool
    open_unknowns: int
    active_contradictions: int
    has_leading_hypothesis: bool
    has_active_frame: bool
    uncertainty: float
    novelty: float
    budget_pressure: float
    resource_scarcity: float

    def __post_init__(self) -> None:
        for name in ("phase_attempts", "consecutive_phase_attempts", "failed_attempts",
                     "reopen_count", "open_unknowns", "active_contradictions"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or value < 0:
                raise ValueError(f"{name} must be a non-negative integer")
        for name in ("uncertainty", "novelty", "budget_pressure", "resource_scarcity"):
            value = getattr(self, name)
            if not math.isfinite(float(value)) or not 0.0 <= float(value) <= 1.0:
                raise ValueError(f"{name} must be within [0, 1]")
        for name, enum_type in (("phase", Phase), ("current_resource_tier", ResourceTier),
                                ("progress", ProgressState), ("progress_evidence", EvidenceStrength)):
            if not isinstance(getattr(self, name), enum_type):
                raise TypeError(f"{name} must be a {enum_type.__name__}")
        for name in ("artifact_valid", "artifact_accepted", "has_leading_hypothesis", "has_active_frame"):
            if not isinstance(getattr(self, name), bool):
                raise TypeError(f"{name} must be a boolean")
        if self.artifact_accepted and not self.artifact_valid:
            raise ValueError("an accepted artifact must be valid")

    def to_dict(self) -> Dict[str, Any]:
        result = asdict(self)
        for name in ("phase", "current_resource_tier", "progress", "progress_evidence"):
            result[name] = getattr(self, name).name
        return result

    @classmethod
    def from_dict(cls, value: Dict[str, Any]) -> "CognitiveState":
        data = dict(value)
        enum_fields = {
            "phase": Phase, "current_resource_tier": ResourceTier,
            "progress": ProgressState, "progress_evidence": EvidenceStrength,
        }
        for field, enum_type in enum_fields.items():
            raw = data[field]
            data[field] = enum_type[raw] if isinstance(raw, str) else enum_type(raw)
        return cls(**data)
