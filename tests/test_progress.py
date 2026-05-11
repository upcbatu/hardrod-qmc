from __future__ import annotations

from hrdmc.io import NullProgress, progress_bar, progress_requested


def test_progress_requested_reads_flag_and_environment(monkeypatch) -> None:
    monkeypatch.delenv("DMC_PROGRESS", raising=False)
    assert not progress_requested(False)
    assert progress_requested(True)

    monkeypatch.setenv("DMC_PROGRESS", "1")
    assert progress_requested(False)


def test_progress_bar_disabled_returns_null_progress() -> None:
    with progress_bar(total=10, label="test", enabled=False) as bar:
        assert isinstance(bar, NullProgress)
        bar.update(3)
