"""Record enough environment and random state to reproduce a run."""

from __future__ import annotations

import platform
import sys
from typing import Any

import numpy as np
import pandas as pd

from datalife_analytics import __version__
from datalife_analytics.contract import load_contract

from .study import ELEVATED_SBP_THRESHOLD, NON_USE, STUDY_ID, STUDY_QUESTION


def lineage(
    *,
    split_seed: int,
    train_fraction: float,
    generator_seed: int | None,
    n_subjects: int | None,
    input_path: str | None,
    contract_version: str | None = None,
    n_splits: int | None = None,
) -> dict[str, Any]:
    contract = load_contract(contract_version) if contract_version else load_contract()
    recorded = {
        "study_id": STUDY_ID,
        "study_question": STUDY_QUESTION,
        "non_use": NON_USE,
        "contract": contract.raw["contract"],
        "contract_version": contract.version,
        "elevated_sbp_threshold": ELEVATED_SBP_THRESHOLD,
        "split_seed": split_seed,
        "train_fraction": train_fraction,
        "generator_seed": generator_seed,
        "n_subjects": n_subjects,
        "input_path": input_path,
        "package_version": __version__,
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "numpy": np.__version__,
        "pandas": pd.__version__,
    }
    if n_splits is not None:
        recorded["n_splits"] = n_splits
    return recorded
