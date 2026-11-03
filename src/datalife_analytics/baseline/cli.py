"""One command: validate, split, fit, evaluate, write report.

Generate in-process from the v0.1.0 generator, or read a dataset directory::

    datalife-baseline --seed 20260101 --subjects 80 --split-seed 7 \\
        --out reports/generated

    datalife-baseline --input tests/fixtures --prefix synthetic_v0_1_0_ \\
        --split-seed 7 --out reports/generated
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import date
from pathlib import Path

from datalife_analytics.contract import ContractError
from datalife_analytics.generator import GeneratorConfig, generate
from datalife_analytics.io import read_dataset

from .pipeline import run_study
from .report import write_reports
from .split import LeakageError


def _date(text: str) -> date:
    try:
        return date.fromisoformat(text)
    except ValueError as exc:
        raise argparse.ArgumentTypeError(f"expected YYYY-MM-DD, got {text!r}") from exc


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="datalife-baseline",
        description="Run the leakage-safe baseline study on the v0.1.0 synthetic contract.",
    )
    parser.add_argument("--input", type=Path, default=None, help="dataset directory with manifest")
    parser.add_argument("--prefix", default="", help="filename prefix used by the dataset")
    parser.add_argument("--seed", type=int, default=20260101, help="generator seed")
    parser.add_argument("--subjects", type=int, default=80)
    parser.add_argument("--start", type=_date, default=date(2024, 1, 1))
    parser.add_argument("--end", type=_date, default=date(2025, 12, 31))
    parser.add_argument("--not-resulted-rate", type=float, default=0.03)
    parser.add_argument("--split-seed", type=int, default=7)
    parser.add_argument("--train-fraction", type=float, default=0.7)
    parser.add_argument(
        "--k-folds",
        type=int,
        default=None,
        help=(
            "Subject-stratified K-fold count for aggregate research evaluation. "
            "Omit to keep the default two-way subject split. "
            "Folds are never for clinical decision-making."
        ),
    )
    parser.add_argument("--out", type=Path, required=True)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        if args.input is not None:
            tables = read_dataset(args.input, prefix=args.prefix)
            manifest = json.loads((args.input / f"{args.prefix}manifest.json").read_text("utf-8"))
            generator_seed = manifest.get("config", {}).get("seed")
            n_subjects = manifest.get("config", {}).get("n_subjects")
            input_path = str(args.input)
        else:
            cfg = GeneratorConfig(
                seed=args.seed,
                n_subjects=args.subjects,
                start=args.start,
                end=args.end,
                not_resulted_rate=args.not_resulted_rate,
            )
            tables = generate(cfg)
            generator_seed = args.seed
            n_subjects = args.subjects
            input_path = None

        result = run_study(
            tables,
            split_seed=args.split_seed,
            train_fraction=args.train_fraction,
            generator_seed=generator_seed,
            n_subjects=n_subjects,
            input_path=input_path,
            n_splits=args.k_folds,
        )
    except (ValueError, ContractError, LeakageError, OSError, KeyError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2

    json_path, md_path = write_reports(result.payload, args.out)
    split = result.payload["split"]
    if split.get("mode") == "subject_stratified_kfold":
        summary = f"k-folds={split['n_splits']} examples={split['n_examples']}"
    else:
        summary = f"train={split['n_train']} test={split['n_test']}"
    print(
        f"wrote {json_path} and {md_path} "
        f"({summary} "
        f"contract={result.payload['lineage']['contract_version']})"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
