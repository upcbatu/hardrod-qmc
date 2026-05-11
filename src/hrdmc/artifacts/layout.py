from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class ArtifactRoute:
    """Canonical result route for reproducible experiment outputs."""

    layer: str
    family: str
    name: str

    def path_under(self, root: Path) -> Path:
        return root / "results" / self.layer / self.family / self.name


def repo_root_from(path: Path, *, marker: str = "pyproject.toml") -> Path:
    current = path.resolve()
    if current.is_file():
        current = current.parent
    for candidate in (current, *current.parents):
        if (candidate / marker).exists():
            return candidate
    raise ValueError(f"could not find repository root from {path}")


def artifact_dir(root: Path, route: ArtifactRoute) -> Path:
    return route.path_under(root)
