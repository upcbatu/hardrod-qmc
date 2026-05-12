from __future__ import annotations

import math
from dataclasses import fields, is_dataclass
from typing import Any

import numpy as np


def to_jsonable(obj: Any) -> Any:  # noqa: ANN401
    if is_dataclass(obj) and not isinstance(obj, type):
        return {field.name: to_jsonable(getattr(obj, field.name)) for field in fields(obj)}
    if isinstance(obj, dict):
        return {str(k): to_jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [to_jsonable(v) for v in obj]
    if isinstance(obj, np.ndarray):
        return to_jsonable(obj.tolist())
    if isinstance(obj, np.generic):
        return to_jsonable(obj.item())
    if isinstance(obj, float) and not math.isfinite(obj):
        return None
    return obj
