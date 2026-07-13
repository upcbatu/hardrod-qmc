from __future__ import annotations

import json
from pathlib import Path
from typing import Any, cast

import numpy as np

from hrdmc.artifacts import ensure_dir
from hrdmc.artifacts.schema import to_jsonable

SCHEMA_VERSION = "dmc_streaming_checkpoint_v4"
LEGACY_V3_SCHEMA_VERSION = "rn_block_streaming_checkpoint_v3"

_V3_METADATA_KEYS = {
    "rn_event_count": "scheduled_move_count",
    "rn_interval_steps": "scheduled_move_interval_steps",
    "collective_rn_enabled": "scheduled_move_enabled",
}
_V3_ARRAY_KEYS = {
    "rn_logk_mean_trace": "scheduled_log_target_mean_trace",
    "rn_logq_mean_trace": "scheduled_log_proposal_mean_trace",
    "rn_logw_increment_mean_trace": "scheduled_log_weight_increment_mean_trace",
    "rn_logw_increment_variance_trace": "scheduled_log_weight_increment_variance_trace",
    "interval_trace_rn_logk_values": "interval_trace_scheduled_log_target_values",
    "interval_trace_rn_logq_values": "interval_trace_scheduled_log_proposal_values",
    "interval_trace_rn_logw_increment_values": (
        "interval_trace_scheduled_log_weight_increment_values"
    ),
    "interval_trace_rn_logw_increment_variance_values": (
        "interval_trace_scheduled_log_weight_increment_variance_values"
    ),
}


def save_streaming_checkpoint(
    path: str | Path,
    *,
    metadata: dict[str, Any],
    arrays: dict[str, np.ndarray],
) -> Path:
    target = Path(path)
    ensure_dir(target.parent)
    payload = {
        **metadata,
        "schema_version": SCHEMA_VERSION,
    }
    tmp = target.with_name(f".{target.name}.tmp")
    savez_compressed = cast(Any, np.savez_compressed)
    archive_arrays: dict[str, Any] = {
        "checkpoint_metadata": np.asarray(json.dumps(to_jsonable(payload), allow_nan=True)),
        **arrays,
    }
    savez_compressed(tmp, **archive_arrays)
    npz_tmp = tmp.with_suffix(tmp.suffix + ".npz")
    npz_tmp.replace(target)
    return target


def load_streaming_checkpoint(path: str | Path) -> tuple[dict[str, Any], dict[str, np.ndarray]]:
    with np.load(Path(path), allow_pickle=False) as archive:
        metadata = json.loads(str(archive["checkpoint_metadata"].item()))
        schema = metadata.get("schema_version")
        arrays = {
            key: np.asarray(archive[key]) for key in archive.files if key != "checkpoint_metadata"
        }
    if schema == SCHEMA_VERSION:
        return metadata, arrays
    if schema == LEGACY_V3_SCHEMA_VERSION:
        return _migrate_v3_checkpoint(metadata, arrays)
    raise ValueError(f"unsupported DMC streaming checkpoint schema: {schema}")


def _migrate_v3_checkpoint(
    metadata: dict[str, Any],
    arrays: dict[str, np.ndarray],
) -> tuple[dict[str, Any], dict[str, np.ndarray]]:
    """Normalize the historical RN-block v3 wire layout for validation.

    Version 3 did not bind checkpoints to a complete run identity. The
    normalized payload therefore remains explicitly unverified; the resume
    validator rejects it instead of silently attaching the current caller's
    guide or algorithm configuration.
    """

    migrated_metadata = dict(metadata)
    for old_key, new_key in _V3_METADATA_KEYS.items():
        if new_key not in migrated_metadata and old_key in migrated_metadata:
            migrated_metadata[new_key] = migrated_metadata[old_key]
        migrated_metadata.pop(old_key, None)
    if migrated_metadata.get("scheduled_move_enabled"):
        migrated_metadata.setdefault("scheduled_move_name", "collective_rn")
    else:
        migrated_metadata.setdefault("scheduled_move_name", None)
    migrated_metadata["source_schema_version"] = LEGACY_V3_SCHEMA_VERSION
    migrated_metadata["schema_version"] = SCHEMA_VERSION
    migrated_metadata["resume_identity"] = None
    migrated_metadata["resume_identity_sha256"] = None

    migrated_arrays = dict(arrays)
    for old_key, new_key in _V3_ARRAY_KEYS.items():
        if new_key not in migrated_arrays and old_key in migrated_arrays:
            migrated_arrays[new_key] = migrated_arrays[old_key]
        migrated_arrays.pop(old_key, None)
    return migrated_metadata, migrated_arrays
