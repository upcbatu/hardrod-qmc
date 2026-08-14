from __future__ import annotations

import csv
import hashlib
import io
import json
import sys
from collections.abc import Iterable, Sequence
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from hrdmc.artifacts.schema import to_jsonable

IMPLEMENTATION_PATHS = ("src/hrdmc", "experiments", "pyproject.toml")


def ensure_dir(path: str | Path) -> Path:
    p = Path(path)
    p.mkdir(parents=True, exist_ok=True)
    return p


def _utc_timestamp() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def _canonical_json_bytes(payload: Any) -> bytes:
    return json.dumps(
        to_jsonable(payload),
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")


def _payload_sha256(payload: Any) -> str:
    return hashlib.sha256(_canonical_json_bytes(payload)).hexdigest()


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
        json.dump(to_jsonable(payload), f, indent=2, sort_keys=True, allow_nan=False)
        f.write("\n")
    tmp.replace(p)
    return p


def write_json(path: str | Path, payload: dict[str, Any]) -> None:
    write_json_atomic(path, payload)


def csv_text(
    rows: Iterable[dict[str, Any]],
    *,
    exclude: Sequence[str] = (),
    fieldnames: Sequence[str] | None = None,
) -> str:
    values = list(rows)
    excluded = set(exclude)
    fields = (
        list(fieldnames)
        if fieldnames is not None
        else sorted({key for row in values for key in row if key not in excluded}) or ["empty"]
    )
    stream = io.StringIO(newline="")
    writer = csv.DictWriter(stream, fieldnames=fields, extrasaction="ignore", lineterminator="\n")
    writer.writeheader()
    writer.writerows(values)
    return stream.getvalue()


def write_csv(
    path: str | Path,
    rows: Iterable[dict[str, Any]],
    *,
    exclude: Sequence[str] = (),
    fieldnames: Sequence[str] | None = None,
) -> Path:
    target = Path(path)
    target.write_text(csv_text(rows, exclude=exclude, fieldnames=fieldnames), encoding="utf-8")
    return target


def config_fingerprint(config: Any) -> str:
    return _payload_sha256(config)


def _implementation_identity(source_root: str | Path | None = None) -> dict[str, Any]:
    """Hash every tracked scientific source byte, including dirty-tree edits."""
    root = _source_root(source_root)
    files: list[Path] = []
    for relative in IMPLEMENTATION_PATHS:
        path = root / relative
        files.extend(path.rglob("*.py") if path.is_dir() else (path,))
    files = sorted((path for path in files if path.is_file()), key=lambda p: str(p))
    digest = hashlib.sha256()
    for path in files:
        relative = path.relative_to(root).as_posix().encode()
        data = path.read_bytes()
        digest.update(len(relative).to_bytes(8, "big"))
        digest.update(relative)
        digest.update(len(data).to_bytes(8, "big"))
        digest.update(data)
    return {
        "status": "identified",
        "source_tree_sha256": digest.hexdigest(),
        "source_file_count": len(files),
    }


def _source_root(source_root: str | Path | None) -> Path:
    if source_root is not None:
        root = Path(source_root).resolve()
        if not (root / "pyproject.toml").is_file():
            raise ValueError(f"source_root is not a project root: {root}")
        return root
    for start in (Path.cwd(), Path(__file__).resolve()):
        for candidate in (start, *start.parents):
            if (candidate / "pyproject.toml").is_file():
                return candidate
    raise ValueError("project source root is unavailable")


def _artifact_entry(root: Path, path: Path) -> dict[str, Any]:
    return {
        "path": path.relative_to(root).as_posix(),
        "size_bytes": path.stat().st_size,
        "sha256": file_sha256(path),
    }


def write_run_manifest(
    output_dir: str | Path,
    *,
    run_name: str,
    config: dict[str, Any],
    artifacts: Sequence[str | Path],
    status: str = "completed",
) -> Path:
    root = ensure_dir(output_dir)
    fingerprint = config_fingerprint(config)
    artifact_paths = [Path(path) for path in artifacts]
    missing = [path for path in artifact_paths if not path.is_file()]
    if missing:
        raise FileNotFoundError(
            "run manifest cannot omit requested artifacts: "
            + ", ".join(str(path) for path in missing)
        )
    if len({path.resolve() for path in artifact_paths}) != len(artifact_paths):
        raise ValueError("run manifest artifacts must be unique")
    manifest: dict[str, Any] = {
        "run_name": run_name,
        "run_id": f"{run_name}_{_utc_timestamp()}_{fingerprint[:12]}",
        "status": status,
        "config_fingerprint": fingerprint,
        "config": to_jsonable(config),
        "provenance": {
            "python_version": sys.version.split()[0],
            "implementation": _implementation_identity(),
        },
        "artifacts": [_artifact_entry(root, path) for path in artifact_paths],
    }
    return write_json_atomic(root / "run_manifest.json", manifest)


def verify_run_manifest(manifest_path: str | Path) -> tuple[bool, list[str]]:
    path = Path(manifest_path)
    manifest = json.loads(path.read_text(encoding="utf-8"))
    root = path.parent
    errors: list[str] = []
    config = manifest.get("config")
    if config is None or manifest.get("config_fingerprint") != config_fingerprint(config):
        errors.append("config fingerprint mismatch")
    seen_paths: set[str] = set()
    for entry in manifest.get("artifacts", []):
        relative_text = str(entry.get("path", ""))
        relative_path = Path(relative_text)
        if (
            not relative_text
            or relative_path.is_absolute()
            or ".." in relative_path.parts
            or relative_text in seen_paths
        ):
            errors.append(f"invalid artifact path: {relative_text}")
            continue
        seen_paths.add(relative_text)
        artifact = root / relative_path
        if not artifact.exists():
            errors.append(f"missing artifact: {relative_text}")
            continue
        if artifact.stat().st_size != entry.get("size_bytes"):
            errors.append(f"size mismatch: {relative_text}")
        if file_sha256(artifact) != entry.get("sha256"):
            errors.append(f"sha256 mismatch: {relative_text}")
    return not errors, errors


def load_manifest_bound_artifact(
    manifest_path: str | Path,
    artifact_path: str | Path,
    *,
    allowed_unrelated_artifact_roots: Sequence[str] = (),
) -> tuple[dict[str, Any], tuple[str, ...]]:
    """Load a manifest after verifying one selected artifact exactly."""
    path = Path(manifest_path).resolve()
    root = path.parent
    selected = Path(artifact_path).resolve()
    try:
        relative_selected = selected.relative_to(root).as_posix()
    except ValueError as exc:
        raise ValueError("selected artifact must be inside its run-manifest directory") from exc
    manifest = json.loads(path.read_text(encoding="utf-8"))
    entries_value = manifest.get("artifacts")
    if not isinstance(entries_value, list):
        raise ValueError(f"run manifest has no artifact records: {path}")
    entries = [entry for entry in entries_value if isinstance(entry, dict)]
    matches = [entry for entry in entries if entry.get("path") == relative_selected]
    if len(matches) != 1:
        raise ValueError(f"selected artifact is not uniquely recorded by its manifest: {selected}")
    entry = matches[0]
    if not selected.is_file():
        raise ValueError(f"selected artifact does not exist: {selected}")
    if entry.get("size_bytes") != selected.stat().st_size:
        raise ValueError(f"selected artifact size does not match its manifest: {selected}")
    if entry.get("sha256") != file_sha256(selected):
        raise ValueError(f"selected artifact digest does not match its manifest: {selected}")
    allowed_roots = _validated_artifact_roots(allowed_unrelated_artifact_roots)
    _, errors = verify_run_manifest(path)
    tolerated: list[str] = []
    for error in errors:
        error_path = _manifest_artifact_error_path(error)
        if error_path == relative_selected:
            raise ValueError(f"selected artifact verification failed: {path}: {error}")
        if (
            error_path is not None
            and Path(error_path).parts
            and Path(error_path).parts[0] in allowed_roots
        ):
            tolerated.append(error)
            continue
        raise ValueError(f"run manifest verification failed: {path}: {error}")
    return manifest, tuple(tolerated)


def _validated_artifact_roots(values: Sequence[str]) -> set[str]:
    roots = set(values)
    invalid = any(
        not value
        or Path(value).is_absolute()
        or len(Path(value).parts) != 1
        or value in {".", ".."}
        for value in roots
    )
    if invalid:
        raise ValueError("allowed unrelated artifact roots must be simple relative names")
    return roots


def _manifest_artifact_error_path(error: str) -> str | None:
    for prefix in ("missing artifact: ", "size mismatch: ", "sha256 mismatch: "):
        if error.startswith(prefix):
            return error[len(prefix) :]
    return None
