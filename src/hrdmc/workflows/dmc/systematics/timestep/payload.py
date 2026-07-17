"""Time-step workflow payload assembly helpers."""

from __future__ import annotations

from typing import Any


def attach_workflow_artifacts(
    payload: dict[str, Any], artifacts: dict[str, str | None]
) -> dict[str, Any]:
    """Attach the stable workflow-artifact projection without changing payload keys."""

    payload["workflow_artifacts"] = artifacts
    return payload
