from __future__ import annotations

from hrdmc.io.artifacts import verify_run_manifest, write_json_atomic, write_run_manifest
from hrdmc.io.checkpoint import read_checkpoint, write_checkpoint


def test_run_manifest_hashes_artifact_bundle(tmp_path) -> None:
    summary = write_json_atomic(tmp_path / "summary.json", {"schema_version": "x", "value": 1})
    manifest = write_run_manifest(
        tmp_path,
        run_name="unit_run",
        config={"dt": 0.001, "seeds": [1, 2]},
        artifacts=[summary],
        schema_version="unit_schema_v1",
        provenance={"created_utc": "2026-05-11T00:00:00Z"},
    )

    ok, errors = verify_run_manifest(manifest)

    assert ok
    assert errors == []


def test_run_manifest_detects_artifact_tampering(tmp_path) -> None:
    summary = write_json_atomic(tmp_path / "summary.json", {"schema_version": "x", "value": 1})
    manifest = write_run_manifest(
        tmp_path,
        run_name="unit_run",
        config={"dt": 0.001},
        artifacts=[summary],
        schema_version="unit_schema_v1",
        provenance={"created_utc": "2026-05-11T00:00:00Z"},
    )
    summary.write_text('{"schema_version":"x","value":2}\n', encoding="utf-8")

    ok, errors = verify_run_manifest(manifest)

    assert not ok
    assert any("sha256 mismatch" in error for error in errors)


def test_checkpoint_roundtrip(tmp_path) -> None:
    checkpoint = write_checkpoint(
        tmp_path / "checkpoint.json",
        {
            "status": "running",
            "completed_cases": [{"case_id": "N4_a0.5_omega0.1"}],
        },
    )

    payload = read_checkpoint(checkpoint)

    assert payload["schema_version"] == "hrdmc_run_checkpoint_v1"
    assert payload["status"] == "running"
    assert payload["completed_cases"][0]["case_id"] == "N4_a0.5_omega0.1"
