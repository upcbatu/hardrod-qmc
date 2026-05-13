from __future__ import annotations

from pathlib import Path
from typing import Any

from hrdmc.plotting.figures.exact_validation_density import (
    write_tg_density_limit_plot,
    write_trapped_tg_density_plots,
)
from hrdmc.plotting.figures.exact_validation_scorecards import (
    write_anchor_error_plot,
    write_homogeneous_sanity_plot,
)
from hrdmc.plotting.style import load_pyplot


def write_exact_validation_packet_plots(
    output_dir: str | Path,
    payload: dict[str, Any],
    *,
    formats: tuple[str, ...] = ("png", "pdf"),
) -> list[str]:
    """Render canonical exact-anchor packet figures."""

    output = Path(output_dir)
    plot_dir = output / "plots"
    plt = load_pyplot(plot_dir)
    paths: list[Path] = []
    paths.extend(write_anchor_error_plot(plt, plot_dir, payload, formats=formats))
    paths.extend(write_trapped_tg_density_plots(plt, plot_dir, payload, formats=formats))
    paths.extend(write_tg_density_limit_plot(plt, plot_dir, payload, formats=formats))
    paths.extend(write_homogeneous_sanity_plot(plt, plot_dir, payload, formats=formats))
    return [str(path.relative_to(output)) for path in paths]
