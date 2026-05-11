from __future__ import annotations

import json
from pathlib import Path
from typing import Any, cast

import numpy as np

from hrdmc.io.artifacts import ensure_dir
from hrdmc.io.schema import to_jsonable

SCHEMA_VERSION = "rn_block_streaming_checkpoint_v1"


def save_streaming_checkpoint(
    path: str | Path,
    *,
    metadata: dict[str, Any],
    arrays: dict[str, np.ndarray],
) -> Path:
    target = Path(path)
    ensure_dir(target.parent)
    payload = {
        "schema_version": SCHEMA_VERSION,
        **metadata,
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
        if schema != SCHEMA_VERSION:
            raise ValueError(f"unsupported RN streaming checkpoint schema: {schema}")
        arrays = {
            key: np.asarray(archive[key])
            for key in archive.files
            if key != "checkpoint_metadata"
        }
    return metadata, arrays
