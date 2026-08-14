"""Core records and current-source guide binding for trapped VMC workflows."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

from hrdmc.system.guide_registry import load_validated_reduced_tg_guide
from hrdmc.system.settings import TrappedCase


@dataclass(frozen=True)
class GuideParameterBinding:
    case_id: str
    guide_family: str
    relative_alpha: float | None
    def factory_parameters(self) -> dict[str, str | float | None]:
        return {
            "guide_family": self.guide_family,
            "relative_alpha": self.relative_alpha,
        }
    def to_dict(self) -> dict[str, Any]:
        """Return the case and guide parameters."""
        return {
            "case_id": self.case_id,
            "guide_parameters": self.factory_parameters(),
        }
def bind_current_reduced_tg_guide(*, expected_case: TrappedCase) -> GuideParameterBinding:
    """Bind the exact current reduced-TG guide for a zero-diameter case."""
    if expected_case.rod_length != 0.0:
        raise ValueError("current-source reduced-TG binding is restricted to A=0 cases")
    guide_family = "reduced-tg"
    relative_alpha = None
    return GuideParameterBinding(
        case_id=expected_case.case_id,
        guide_family=guide_family,
        relative_alpha=relative_alpha,
    )


def bind_validated_reduced_tg_guide(
    summary_path: Path,
    *,
    expected_case: TrappedCase,
) -> GuideParameterBinding:
    """Bind a registry-validated finite-diameter reduced-TG width."""
    artifact = load_validated_reduced_tg_guide(summary_path, case=expected_case)
    return GuideParameterBinding(
        case_id=expected_case.case_id,
        guide_family="reduced-tg",
        relative_alpha=artifact.relative_alpha,
    )
