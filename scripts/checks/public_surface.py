#!/usr/bin/env python3
"""Bounded, dependency-free audit of physicist-facing repository text."""

from __future__ import annotations

import argparse
import re
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path

TEXT_SUFFIXES = {
    ".csv",
    ".json",
    ".md",
    ".py",
    ".rst",
    ".sh",
    ".toml",
    ".txt",
    ".yaml",
    ".yml",
}
PUBLIC_TREES = ("src", "experiments", "docs")
PUBLIC_FILES = ("README.md", "Makefile")


@dataclass(frozen=True)
class Rule:
    name: str
    pattern: re.Pattern[str]


@dataclass(frozen=True)
class Finding:
    category: str
    rule: str
    path: Path
    line_number: int
    line: str


HARD_RULES = (
    Rule("old-unit-parameter", re.compile(r"\bomega_code\b")),
    Rule(
        "old-dmc-owner",
        re.compile(
            r"\b(?:workflows|monte_carlo)\.dmc\.rn_block\b"
            r"|(?:src/hrdmc/)?(?:workflows|monte_carlo)/dmc/rn_block"
            r"|experiments/dmc/rn_block"
        ),
    ),
    Rule(
        "old-case-id",
        re.compile(r"\bN\d+_a[A-Za-z0-9.+-]+_omega[A-Za-z0-9.+-]+\b"),
    ),
)
LANGUAGE_RULES = (
    Rule("decision-shorthand", re.compile(r"\b(?:GO|NO[-_]GO|P[0-2]|LLM)\b")),
    Rule(
        "process-language",
        re.compile(
            r"\b(?:gate|hygiene|smoke|prompt|subagent)\b"
            r"|gold standard|hostile audit|claim[ _-]?boundary",
            re.IGNORECASE,
        ),
    ),
)
UNIT_REVIEW_RULES = (
    Rule("kinetic-unit-algebra", re.compile(r"hbar\^2/\(2m")),
    Rule("sqrt-two", re.compile(r"sqrt\(2\)")),
)


def _public_paths(root: Path) -> Iterable[Path]:
    for tree_name in PUBLIC_TREES:
        tree = root / tree_name
        if not tree.exists():
            continue
        for path in sorted(tree.rglob("*")):
            if path.is_file() and path.suffix.lower() in TEXT_SUFFIXES:
                yield path
    for file_name in PUBLIC_FILES:
        path = root / file_name
        if path.is_file():
            yield path


def _scan(root: Path) -> list[Finding]:
    findings: list[Finding] = []
    groups = (
        ("ERROR", HARD_RULES),
        ("REVIEW_LANGUAGE", LANGUAGE_RULES),
        ("REVIEW_UNIT", UNIT_REVIEW_RULES),
    )
    for path in _public_paths(root):
        relative = path.relative_to(root)
        for line_number, line in enumerate(
            path.read_text(encoding="utf-8", errors="replace").splitlines(),
            start=1,
        ):
            for category, rules in groups:
                for rule in rules:
                    if rule.pattern.search(line):
                        findings.append(Finding(category, rule.name, relative, line_number, line))
    return findings


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path.cwd())
    parser.add_argument(
        "--limit",
        type=int,
        default=60,
        help="maximum displayed findings per category",
    )
    args = parser.parse_args()
    root = args.root.resolve()
    findings = _scan(root)

    categories = ("ERROR", "REVIEW_LANGUAGE", "REVIEW_UNIT")
    for category in categories:
        matches = [finding for finding in findings if finding.category == category]
        for finding in matches[: args.limit]:
            excerpt = finding.line.strip()
            print(f"[{category} {finding.rule}] {finding.path}:{finding.line_number}: {excerpt}")
        hidden = len(matches) - args.limit
        if hidden > 0:
            print(f"[{category}] ... {hidden} additional findings suppressed")
        print(f"[{category}] count={len(matches)}")

    return 1 if any(finding.category == "ERROR" for finding in findings) else 0


if __name__ == "__main__":
    raise SystemExit(main())
