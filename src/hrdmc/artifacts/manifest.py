from __future__ import annotations

import hashlib
import importlib.metadata
import json
import platform
import subprocess
import sys
from collections.abc import Sequence
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from hrdmc.artifacts.schema import to_jsonable

IMPLEMENTATION_SOURCE_DIRECTORIES = ("src/hrdmc", "experiments")
IMPLEMENTATION_SOURCE_FILES = ("pyproject.toml",)
RUNTIME_DISTRIBUTIONS = ("numpy", "scipy", "numba", "matplotlib", "tqdm")


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
        allow_nan=False,
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
        json.dump(to_jsonable(payload), f, indent=2, sort_keys=True, allow_nan=False)
        f.write("\n")
    tmp.replace(p)
    return p


def write_json(path: str | Path, payload: dict[str, Any]) -> None:
    write_json_atomic(path, payload)


def build_run_provenance(
    command: list[str] | None = None,
    *,
    source_root: str | Path | None = None,
) -> dict[str, Any]:
    """Describe the runtime and bind the run to the sampled source tree."""

    return {
        "created_utc": utc_timestamp(),
        "python_version": sys.version.split()[0],
        "platform": platform.platform(),
        "dependencies": _runtime_dependencies(),
        "command": command,
        "implementation": implementation_identity(source_root),
    }


def implementation_identity(source_root: str | Path | None = None) -> dict[str, Any]:
    """Return a content identity for scientific source files, including dirty trees."""

    root = _resolve_source_root(source_root)
    if root is None:
        return {"status": "source_root_unavailable"}
    files = _implementation_source_paths(root)
    if not files:
        return {"status": "source_files_unavailable"}

    digest = hashlib.sha256()
    for path in files:
        relative = path.relative_to(root).as_posix().encode("utf-8")
        digest.update(len(relative).to_bytes(8, byteorder="big"))
        digest.update(relative)
        data = path.read_bytes()
        digest.update(len(data).to_bytes(8, byteorder="big"))
        digest.update(data)

    identity: dict[str, Any] = {
        "status": "identified",
        "source_tree_sha256": digest.hexdigest(),
        "source_file_count": len(files),
    }
    identity.update(_git_identity(root))
    return identity


def _runtime_dependencies() -> dict[str, str]:
    versions: dict[str, str] = {}
    for distribution in RUNTIME_DISTRIBUTIONS:
        try:
            versions[distribution] = importlib.metadata.version(distribution)
        except importlib.metadata.PackageNotFoundError:
            versions[distribution] = "not_installed"
    return versions


def _resolve_source_root(source_root: str | Path | None) -> Path | None:
    if source_root is not None:
        candidate = Path(source_root).resolve()
        if not (candidate / "pyproject.toml").is_file():
            raise ValueError(f"source_root is not a project root: {candidate}")
        return candidate
    for start in (Path.cwd(), Path(__file__).resolve()):
        for candidate in (start, *start.parents):
            if (candidate / "pyproject.toml").is_file():
                return candidate
    return None


def _implementation_source_paths(root: Path) -> list[Path]:
    paths = [
        root / relative for relative in IMPLEMENTATION_SOURCE_FILES if (root / relative).is_file()
    ]
    for relative in IMPLEMENTATION_SOURCE_DIRECTORIES:
        directory = root / relative
        if directory.is_dir():
            paths.extend(path for path in directory.rglob("*.py") if path.is_file())
    return sorted(set(paths), key=lambda path: path.relative_to(root).as_posix())


def _git_identity(root: Path) -> dict[str, Any]:
    try:
        revision = subprocess.run(
            ["git", "-C", str(root), "rev-parse", "--verify", "HEAD"],
            check=True,
            capture_output=True,
            text=True,
            timeout=2.0,
        ).stdout.strip()
        status = subprocess.run(
            [
                "git",
                "-C",
                str(root),
                "status",
                "--porcelain",
                "--untracked-files=all",
                "--",
                *IMPLEMENTATION_SOURCE_DIRECTORIES,
                *IMPLEMENTATION_SOURCE_FILES,
            ],
            check=True,
            capture_output=True,
            text=True,
            timeout=2.0,
        ).stdout
    except (OSError, subprocess.SubprocessError):
        return {"git_status": "unavailable"}
    return {
        "git_status": "clean" if not status.strip() else "dirty",
        "git_revision": revision,
    }


def config_fingerprint(config: Any) -> str:
    return payload_sha256(config)


def _artifact_entry(root: Path, path: Path) -> dict[str, Any]:
    return {
        "path": path.relative_to(root).as_posix(),
        "size_bytes": path.stat().st_size,
        "sha256": file_sha256(path),
    }


def _bundle_payload_v1(manifest: dict[str, Any]) -> dict[str, Any]:
    return {
        "schema_version": manifest["schema_version"],
        "run_id": manifest["run_id"],
        "config_fingerprint": manifest["config_fingerprint"],
        "artifacts": manifest["artifacts"],
    }


def _bundle_payload_v2(manifest: dict[str, Any]) -> dict[str, Any]:
    return {
        "schema_version": manifest["schema_version"],
        "result_schema_version": manifest["result_schema_version"],
        "run_name": manifest["run_name"],
        "run_id": manifest["run_id"],
        "status": manifest["status"],
        "config_fingerprint": manifest["config_fingerprint"],
        "config": manifest["config"],
        "provenance": manifest["provenance"],
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
    manifest_provenance = _merge_run_provenance(provenance)
    provenance_errors = _provenance_identity_errors(manifest_provenance)
    if provenance_errors:
        raise ValueError("run provenance is incomplete: " + "; ".join(provenance_errors))
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
        "schema_version": "hrdmc_run_manifest_v2",
        "result_schema_version": schema_version,
        "run_name": run_name,
        "run_id": f"{run_name}_{utc_timestamp()}_{fingerprint[:12]}",
        "status": status,
        "config_fingerprint": fingerprint,
        "config": to_jsonable(config),
        "provenance": manifest_provenance,
        "artifacts": [_artifact_entry(root, path) for path in artifact_paths],
    }
    manifest["bundle_sha256"] = payload_sha256(_bundle_payload_v2(manifest))
    return write_json_atomic(root / "run_manifest.json", manifest)


def _merge_run_provenance(provenance: dict[str, Any] | None) -> dict[str, Any]:
    """Keep mandatory runtime/source identity when callers add provenance fields."""

    merged = build_run_provenance()
    if provenance is None:
        return merged
    supplied = to_jsonable(provenance)
    if not isinstance(supplied, dict):
        raise TypeError("run provenance must be a mapping")
    for key, value in supplied.items():
        if key in {"dependencies", "implementation"}:
            current = merged.get(key)
            if isinstance(current, dict) and isinstance(value, dict):
                merged[key] = {**current, **value}
                continue
        merged[key] = value
    return merged


def _provenance_identity_errors(provenance: Any) -> list[str]:
    if not isinstance(provenance, dict):
        return ["provenance is not a mapping"]
    errors: list[str] = []
    implementation = provenance.get("implementation")
    source_digest = (
        implementation.get("source_tree_sha256") if isinstance(implementation, dict) else None
    )
    if (
        not isinstance(source_digest, str)
        or len(source_digest) != 64
        or any(character not in "0123456789abcdef" for character in source_digest)
    ):
        errors.append("implementation source-tree SHA-256 is missing")
    dependencies = provenance.get("dependencies")
    if (
        not isinstance(dependencies, dict)
        or not dependencies
        or not all(
            isinstance(name, str) and isinstance(version, str)
            for name, version in dependencies.items()
        )
    ):
        errors.append("runtime dependency versions are missing")
    if not isinstance(provenance.get("python_version"), str) or not provenance["python_version"]:
        errors.append("Python version is missing")
    if not isinstance(provenance.get("platform"), str) or not provenance["platform"]:
        errors.append("platform identity is missing")
    return errors


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
    schema = manifest.get("schema_version")
    try:
        if schema == "hrdmc_run_manifest_v1":
            expected = payload_sha256(_bundle_payload_v1(manifest))
        elif schema == "hrdmc_run_manifest_v2":
            errors.extend(_provenance_identity_errors(manifest.get("provenance")))
            expected = payload_sha256(_bundle_payload_v2(manifest))
        else:
            errors.append(f"unsupported manifest schema: {schema}")
            expected = None
    except KeyError as exc:
        errors.append(f"manifest is missing required field: {exc.args[0]}")
        expected = None
    if expected is not None and manifest.get("bundle_sha256") != expected:
        errors.append("bundle sha256 mismatch")
    return not errors, errors
