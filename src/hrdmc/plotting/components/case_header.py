from __future__ import annotations

from typing import Any

from hrdmc.plotting import tokens


def draw_case_header(fig: Any, payload: dict[str, Any]) -> None:  # noqa: ANN401
    controls = payload.get("controls", {})
    seeds = payload.get("seeds", [])
    seed_text = f"{len(seeds)} seeds" if isinstance(seeds, list) else "seeds unavailable"
    rod_text = (
        rf"$a/a_{{\mathrm{{ho}}}}={payload.get('rod_length_ho')}$"
        if payload.get("case_parameterization") == "harmonic_oscillator_units"
        else f"a={payload.get('rod_length', '?')}"
    )
    trap_text = (
        "oscillator units"
        if payload.get("case_parameterization") == "harmonic_oscillator_units"
        else f"omega={payload.get('omega', '?')}"
    )
    text = (
        f"N={payload.get('n_particles', '?')}  |  "
        f"{rod_text}  |  "
        f"{trap_text}  |  "
        rf"$\Delta\tau\,\omega={controls.get('dt', '?')}$  |  "
        f"{controls.get('walkers', '?')} walkers  |  "
        f"{seed_text}  |  "
        rf"$\tau_{{\mathrm{{prod}}}}\omega={controls.get('production_tau', '?')}$"
    )
    fig.suptitle(
        text,
        x=0.01,
        y=0.985,
        ha="left",
        va="top",
        fontsize=7.2,
        color=tokens.INK_SOFT,
    )
