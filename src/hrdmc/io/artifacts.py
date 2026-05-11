from __future__ import annotations

import hashlib
import json
import platform
import sys
from collections.abc import Sequence
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from hrdmc.io.schema import to_jsonable


def ensure_dir(path: str | Path) -> Path:
    p = Path(path)
    p.mkdir(parents=True, exist_ok=True)
    return p


def utc_timestamp() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def canonical_json_bytes(payload: Any) -> bytes:
    return json.dumps(
        to_jsonable(payload),
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=True,
    ).encode("utf-8")


def payload_sha256(payload: Any) -> str:
    return hashlib.sha256(canonical_json_bytes(payload)).hexdigest()


def file_sha256(path: str | Path) -> str:
    h = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def write_json_atomic(path: str | Path, payload: Any) -> Path:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    tmp = p.with_name(f".{p.name}.tmp")
    with tmp.open("w", encoding="utf-8") as f:
        json.dump(to_jsonable(payload), f, indent=2, sort_keys=True)
        f.write("\n")
    tmp.replace(p)
    return p


def write_json(path: str | Path, payload: dict[str, Any]) -> None:
    write_json_atomic(path, payload)


def build_run_provenance(command: list[str] | None = None) -> dict[str, Any]:
    return {
        "created_utc": utc_timestamp(),
        "python_version": sys.version.split()[0],
        "platform": platform.platform(),
        "command": command,
    }


def config_fingerprint(config: Any) -> str:
    return payload_sha256(config)


def _artifact_entry(root: Path, path: Path) -> dict[str, Any]:
    return {
        "path": path.relative_to(root).as_posix(),
        "size_bytes": path.stat().st_size,
        "sha256": file_sha256(path),
    }


def _bundle_payload(manifest: dict[str, Any]) -> dict[str, Any]:
    return {
        "schema_version": manifest["schema_version"],
        "run_id": manifest["run_id"],
        "config_fingerprint": manifest["config_fingerprint"],
        "artifacts": manifest["artifacts"],
    }


def write_run_manifest(
    output_dir: str | Path,
    *,
    run_name: str,
    config: dict[str, Any],
    artifacts: Sequence[str | Path],
    schema_version: str,
    provenance: dict[str, Any] | None = None,
    status: str = "completed",
) -> Path:
    root = ensure_dir(output_dir)
    fingerprint = config_fingerprint(config)
    artifact_paths = [Path(path) for path in artifacts]
    manifest: dict[str, Any] = {
        "schema_version": "hrdmc_run_manifest_v1",
        "result_schema_version": schema_version,
        "run_name": run_name,
        "run_id": f"{run_name}_{utc_timestamp()}_{fingerprint[:12]}",
        "status": status,
        "config_fingerprint": fingerprint,
        "config": to_jsonable(config),
        "provenance": provenance or build_run_provenance(),
        "artifacts": [_artifact_entry(root, path) for path in artifact_paths if path.exists()],
    }
    manifest["bundle_sha256"] = payload_sha256(_bundle_payload(manifest))
    return write_json_atomic(root / "run_manifest.json", manifest)


def verify_run_manifest(manifest_path: str | Path) -> tuple[bool, list[str]]:
    path = Path(manifest_path)
    manifest = json.loads(path.read_text(encoding="utf-8"))
    root = path.parent
    errors: list[str] = []
    for entry in manifest.get("artifacts", []):
        artifact = root / entry["path"]
        if not artifact.exists():
            errors.append(f"missing artifact: {entry['path']}")
            continue
        if artifact.stat().st_size != entry["size_bytes"]:
            errors.append(f"size mismatch: {entry['path']}")
        if file_sha256(artifact) != entry["sha256"]:
            errors.append(f"sha256 mismatch: {entry['path']}")
    expected = payload_sha256(_bundle_payload(manifest))
    if manifest.get("bundle_sha256") != expected:
        errors.append("bundle sha256 mismatch")
    return not errors, errors
