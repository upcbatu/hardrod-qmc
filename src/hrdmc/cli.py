from __future__ import annotations

import argparse


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Trapped hard-rod QMC thesis scaffold")
    sub = parser.add_subparsers(dest="command")
    sub.add_parser("version", help="Print package version")
    return parser


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()
    if args.command == "version":
        from hrdmc import __version__

        print(__version__)
        return 0
    parser.print_help()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
