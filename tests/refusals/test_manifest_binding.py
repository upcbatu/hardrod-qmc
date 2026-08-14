from __future__ import annotations

import pytest

from hrdmc.artifacts.manifest import (
    load_manifest_bound_artifact,
    verify_run_manifest,
    write_json_atomic,
    write_run_manifest,
)


def test_manifest_verification_detects_artifact_tampering(tmp_path) -> None:
    summary = write_json_atomic(tmp_path / "summary.json", {"schema_version": "x", "value": 1})
    manifest = write_run_manifest(
        tmp_path, run_name="unit_run", config={"dt": 0.001}, artifacts=[summary]
    )
    summary.write_text('{"schema_version":"x","value":2}\n', encoding="utf-8")

    valid, errors = verify_run_manifest(manifest)

    assert not valid
    assert any("sha256 mismatch" in error for error in errors)


def test_selected_artifact_loader_rejects_nonplot_drift(tmp_path) -> None:
    summary = write_json_atomic(tmp_path / "summary.json", {"schema_version": "x"})
    table = tmp_path / "table.csv"
    table.write_text("value\n1\n", encoding="utf-8")
    manifest = write_run_manifest(
        tmp_path,
        run_name="unit_run",
        config={"dt": 0.001},
        artifacts=[summary, table],
    )
    table.write_text("value\n2\n", encoding="utf-8")

    with pytest.raises(ValueError, match="manifest verification failed"):
        load_manifest_bound_artifact(manifest, summary, allowed_unrelated_artifact_roots=("plots",))
