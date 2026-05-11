from __future__ import annotations


def excluded_length(n_particles: int, length: float, rod_length: float) -> float:
    """Reduced coordinate length L' = L - N a for hard-rod geometry."""
    if n_particles <= 0:
        raise ValueError("n_particles must be positive")
    if length <= 0:
        raise ValueError("length must be positive")
    if rod_length < 0:
        raise ValueError("rod_length must be non-negative")
    reduced_length = length - n_particles * rod_length
    if reduced_length <= 0:
        raise ValueError("excluded length N * a must be smaller than L")
    return float(reduced_length)
