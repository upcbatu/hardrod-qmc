from __future__ import annotations

import json
from collections.abc import Mapping
from typing import Any

from hrdmc.artifacts.schema import to_jsonable


def print_run_summary(
    *,
    run: str,
    status: str,
    summary: Mapping[str, Any] | None = None,
    artifacts: Mapping[str, str | None] | None = None,
    verbose_payload: Any | None = None,
    verbose_json: bool = False,
) -> None:
    """Print a bounded run summary; keep full scientific payloads in artifacts."""

    if verbose_json:
        if verbose_payload is None:
            raise ValueError("verbose_json requires verbose_payload")
        print(json.dumps(to_jsonable(verbose_payload), indent=2, allow_nan=False))
        return
    payload: dict[str, Any] = {
        "run": run,
        "status": status,
    }
    if summary:
        payload["summary"] = dict(summary)
    if artifacts:
        payload["artifacts"] = {name: path for name, path in artifacts.items() if path is not None}
    print(json.dumps(to_jsonable(payload), indent=2, allow_nan=False))
