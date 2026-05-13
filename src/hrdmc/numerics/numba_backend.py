from __future__ import annotations

from collections.abc import Callable
from typing import Any, TypeVar, overload

F = TypeVar("F", bound=Callable[..., Any])

try:
    from numba import njit as _numba_njit
except ModuleNotFoundError:  # pragma: no cover - optional performance extra
    NUMBA_AVAILABLE = False
else:
    NUMBA_AVAILABLE = True


@overload
def njit(function: F, /, **kwargs: Any) -> F: ...


@overload
def njit(*args: Any, **kwargs: Any) -> Callable[[F], F]: ...


def njit(*args: Any, **kwargs: Any) -> Any:
    """Return numba.njit when available, otherwise an identity decorator."""

    if NUMBA_AVAILABLE:
        return _numba_njit(*args, **kwargs)
    if args and callable(args[0]) and len(args) == 1 and not kwargs:
        return args[0]

    def decorator(function: F) -> F:
        return function

    return decorator


def numba_backend_name() -> str:
    return "numba" if NUMBA_AVAILABLE else "python"


def backend_label(prefix: str) -> str:
    return f"{prefix}_{numba_backend_name()}"
