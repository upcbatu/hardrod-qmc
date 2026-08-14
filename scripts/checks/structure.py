"""Module size, subject naming, and package layering. Limits in [tool.hrdmc.structure]."""

from __future__ import annotations

import argparse
import ast
import sys
from collections import defaultdict
from pathlib import Path

import tomllib

Failure = tuple[str, str]


def load_limits(root: Path) -> dict[str, object]:
    config = tomllib.loads((root / "pyproject.toml").read_text(encoding="utf-8"))
    section = config.get("tool", {}).get("hrdmc", {}).get("structure", {})
    if not section:
        raise SystemExit("pyproject.toml has no [tool.hrdmc.structure] section")
    return section


def module_files(package_root: Path) -> list[Path]:
    return sorted(p for p in package_root.rglob("*.py") if "__pycache__" not in p.parts)


def check_module_size(
    files: list[Path],
    root: Path,
    *,
    max_lines: int,
    max_callable_lines: int,
    max_symbols: int,
) -> list[Failure]:
    failures: list[Failure] = []
    for path in files:
        source = path.read_text(encoding="utf-8")
        lines = len(source.splitlines())
        try:
            tree = ast.parse(source)
        except SyntaxError as exc:
            failures.append((str(path.relative_to(root)), f"does not parse: {exc}"))
            continue
        symbols = sum(
            1
            for node in tree.body
            if isinstance(node, ast.FunctionDef | ast.AsyncFunctionDef | ast.ClassDef)
        )
        rel = str(path.relative_to(root))
        if lines > max_lines:
            failures.append((rel, f"{lines} lines exceeds the {max_lines}-line module ceiling"))
        if symbols > max_symbols:
            failures.append(
                (rel, f"{symbols} top-level symbols exceeds the {max_symbols}-symbol ceiling")
            )
        for node in ast.walk(tree):
            if not isinstance(node, ast.FunctionDef | ast.AsyncFunctionDef):
                continue
            span = getattr(node, "end_lineno", node.lineno) - node.lineno + 1
            if span > max_callable_lines:
                failures.append(
                    (
                        f"{rel}:{node.lineno}",
                        f"{node.name} spans {span} lines, over the "
                        f"{max_callable_lines}-line callable ceiling",
                    )
                )
    return failures


def check_subject_names(
    files: list[Path],
    root: Path,
    *,
    stage_packages: set[str],
    role_module_names: set[str],
    subject_group_min_modules: int,
    subject_group_min_lines: int,
) -> list[Failure]:
    package_root = root / "src" / "hrdmc"
    by_directory_and_prefix: dict[tuple[Path, str], list[Path]] = defaultdict(list)
    failures: list[Failure] = []

    for path in files:
        if path.name == "__init__.py":
            continue
        relative = path.relative_to(package_root)
        if (
            len(relative.parts) == 2
            and relative.parts[0] in stage_packages
            and path.stem in role_module_names
        ):
            failures.append(
                (
                    str(path.relative_to(root)),
                    f"role-only module name is ambiguous at the {relative.parts[0]} stage",
                )
            )
        prefix, separator, _remainder = path.stem.partition("_")
        if separator:
            by_directory_and_prefix[(path.parent, prefix)].append(path)

    for (directory, prefix), paths in sorted(by_directory_and_prefix.items()):
        if len(paths) < 2:
            continue
        total_lines = sum(len(path.read_text(encoding="utf-8").splitlines()) for path in paths)
        if len(paths) < subject_group_min_modules and total_lines < subject_group_min_lines:
            continue
        locations = ", ".join(path.name for path in sorted(paths))
        failures.append(
            (
                str(directory.relative_to(root)),
                f"{len(paths)} modules share prefix {prefix!r} ({total_lines} lines): "
                f"{locations}; create a subject directory",
            )
        )
    return failures


def parse_layering(rules: list[str]) -> list[tuple[str, str]]:
    parsed: list[tuple[str, str]] = []
    for rule in rules:
        importer, _, imported = rule.partition(" must not import ")
        if not importer or not imported:
            raise SystemExit(f"malformed layering rule: {rule!r}")
        parsed.append((importer.strip(), imported.strip()))
    return parsed


def imported_modules(tree: ast.AST) -> set[str]:
    found: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom) and node.module:
            found.add(node.module)
        elif isinstance(node, ast.Import):
            found.update(alias.name for alias in node.names)
    return found


def check_layering(
    files: list[Path],
    root: Path,
    *,
    rules: list[tuple[str, str]],
) -> list[Failure]:
    failures: list[Failure] = []
    for path in files:
        try:
            tree = ast.parse(path.read_text(encoding="utf-8"))
        except SyntaxError:
            continue
        module = ".".join(path.relative_to(root / "src").with_suffix("").parts)
        imports = imported_modules(tree)
        for importer, imported in rules:
            if not (module == importer or module.startswith(f"{importer}.")):
                continue
            offenders = sorted(
                name
                for name in imports
                if name == imported or name.startswith(f"{imported}.")
            )
            if offenders:
                failures.append(
                    (
                        str(path.relative_to(root)),
                        f"{importer} must not import {imported}: {', '.join(offenders)}",
                    )
                )
    return failures


def report(title: str, failures: list[Failure]) -> None:
    if not failures:
        print(f"  ok    {title}")
        return
    print(f"  FAIL  {title}")
    for subject, detail in failures:
        print(f"          {subject}: {detail}")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[2])
    args = parser.parse_args()
    root = args.root.resolve()

    limits = load_limits(root)
    files = module_files(root / "src" / "hrdmc")
    if not files:
        raise SystemExit(f"no modules found under {root / 'src' / 'hrdmc'}")

    print(f"structure: {len(files)} modules under src/hrdmc")
    size = check_module_size(
        files,
        root,
        max_lines=int(limits["max-module-lines"]),  # type: ignore[arg-type]
        max_callable_lines=int(limits["max-callable-lines"]),  # type: ignore[arg-type]
        max_symbols=int(limits["max-module-symbols"]),  # type: ignore[arg-type]
    )
    names = check_subject_names(
        files,
        root,
        stage_packages=set(limits.get("stage-packages", [])),  # type: ignore[arg-type]
        role_module_names=set(limits.get("role-module-names", [])),  # type: ignore[arg-type]
        subject_group_min_modules=int(limits["subject-group-min-modules"]),  # type: ignore[arg-type]
        subject_group_min_lines=int(limits["subject-group-min-lines"]),  # type: ignore[arg-type]
    )
    layers = check_layering(
        files,
        root,
        rules=parse_layering(list(limits.get("layering", []))),  # type: ignore[arg-type]
    )

    report("module size", size)
    report("subject module names", names)
    report("package layering", layers)

    total = len(size) + len(names) + len(layers)
    if total:
        print(f"structure: {total} failures")
        return 1
    print("structure: clean")
    return 0


if __name__ == "__main__":
    sys.exit(main())
