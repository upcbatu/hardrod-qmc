from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from hrdmc.io.artifacts import utc_timestamp, write_json_atomic

CHECKPOINT_SCHEMA_VERSION = "hrdmc_run_checkpoint_v1"


def write_checkpoint(path: str | Path, payload: dict[str, Any]) -> Path:
    checkpoint = {
        "schema_version": CHECKPOINT_SCHEMA_VERSION,
        "updated_utc": utc_timestamp(),
        **payload,
    }
    return write_json_atomic(path, checkpoint)


def read_checkpoint(path: str | Path) -> dict[str, Any]:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    schema = payload.get("schema_version")
    if schema != CHECKPOINT_SCHEMA_VERSION:
        raise ValueError(f"unsupported checkpoint schema: {schema}")
    return payload
