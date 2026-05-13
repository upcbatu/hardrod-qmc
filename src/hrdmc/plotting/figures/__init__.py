from __future__ import annotations

from typing import Any


def __getattr__(name: str) -> Any:  # noqa: ANN401
    if name == "write_benchmark_packet_plots":
        from hrdmc.plotting.figures.benchmark_packet import write_benchmark_packet_plots

        return write_benchmark_packet_plots
    if name == "write_claim_matrix_plot":
        from hrdmc.plotting.figures.claim_matrix import write_claim_matrix_plot

        return write_claim_matrix_plot
    if name == "write_exact_tg_trap_plots":
        from hrdmc.plotting.figures.exact_tg_trap import write_exact_tg_trap_plots

        return write_exact_tg_trap_plots
    if name == "write_exact_validation_packet_plots":
        from hrdmc.plotting.figures.exact_validation_packet import (
            write_exact_validation_packet_plots,
        )

        return write_exact_validation_packet_plots
    if name == "write_finite_a_n2_reference_plots":
        from hrdmc.plotting.figures.finite_a_n2_reference import (
            write_finite_a_n2_reference_plots,
        )

        return write_finite_a_n2_reference_plots
    raise AttributeError(name)
