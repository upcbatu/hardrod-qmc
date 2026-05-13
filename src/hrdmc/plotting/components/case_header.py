from __future__ import annotations

from typing import Any

from hrdmc.plotting import tokens


def draw_case_header(fig: Any, payload: dict[str, Any]) -> None:  # noqa: ANN401
    controls = payload.get("controls", {})
    seeds = payload.get("seeds", [])
    metadata = payload.get("stationarity", {})
    seed_text = f"{len(seeds)} seeds" if isinstance(seeds, list) else "seeds unavailable"
    text = (
        f"N={payload.get('n_particles', '?')}  "
        f"a={payload.get('rod_length', '?')}  "
        f"omega={payload.get('omega', '?')}  "
        f"dt={controls.get('dt', '?')}  "
        f"M={controls.get('walkers', '?')}  "
        f"{seed_text}  "
        f"prod_tau={controls.get('production_tau', '?')}  "
        f"engine={metadata.get('engine_corridor', 'rn_block')}"
    )
    fig.suptitle(
        text,
        x=0.01,
        y=0.995,
        ha="left",
        va="top",
        fontsize=8,
        family="monospace",
        color=tokens.INK_SOFT,
    )
