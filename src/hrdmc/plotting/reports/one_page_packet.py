from __future__ import annotations

from pathlib import Path
from typing import Any

from hrdmc.plotting.figures.benchmark_packet import write_benchmark_packet_plots


def write_one_page_packet(
    output_dir: str | Path,
    payload: dict[str, Any],
    *,
    formats: tuple[str, ...] = ("pdf",),
) -> list[str]:
    """Write the benchmark packet report figure.

    The benchmark figure renderer owns the actual composition; this report
    entry point keeps the report lane separate for future multi-case packets.
    """

    paths = write_benchmark_packet_plots(output_dir, payload, formats=formats)
    return [path for path in paths if "benchmark_packet_one_page" in path]
