from __future__ import annotations

import json
from pathlib import Path

from hrdmc.system.settings import TrappedCase
from hrdmc.validation.sampler_equivalence.models import VMCSamplerChoice


def load_calibrated_sampler_choices(
    summary_path: Path,
    *,
    expected_case: TrappedCase,
) -> dict[str, VMCSamplerChoice]:
    """Load calibrated sampler choices for one trapped case."""
    resolved = summary_path.resolve()
    try:
        payload = json.loads(resolved.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError("VMC calibration summary is not valid JSON") from exc
    if not isinstance(payload, dict):
        raise ValueError("VMC calibration summary must be a JSON object")
    if payload.get("status") != "calibrated" or payload.get("case_id") != expected_case.case_id:
        raise ValueError("VMC calibration is not calibrated for the requested case")
    selected = payload.get("selected")
    if not isinstance(selected, dict):
        raise ValueError("VMC calibration has no selected sampler choices")
    choices: dict[str, VMCSamplerChoice] = {}
    for sampler in (
        "random_walk_metropolis",
        "branching_free_mala",
    ):
        row = selected.get(sampler)
        if not isinstance(row, dict):
            raise ValueError(f"VMC calibration has no selected {sampler} choice")
        if row.get("method") != sampler:
            raise ValueError(f"VMC calibration selected {sampler} choice has the wrong method")
        choice = VMCSamplerChoice(
            method=sampler,
            proposal_scale=float(row["proposal_scale"]),
            drift_limiter=str(row["drift_limiter"]),
        )
        choice.validate()
        choices[sampler] = choice
    return choices
