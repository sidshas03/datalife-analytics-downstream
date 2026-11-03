"""Validate, split, fit, evaluate, and build the report payload."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd

from datalife_analytics.contract import load_contract, validate

from . import metrics as metrics_mod
from .features import FeatureEncoder
from .lineage import lineage
from .models import NaivePrevalence, RidgeLogistic
from .split import (
    KFOLD_RESEARCH_ONLY,
    assert_no_cross_split_time_leakage,
    assert_no_subject_leakage,
    assert_no_time_leakage,
    assert_no_time_leakage_against_tables,
    subject_split,
    subject_stratified_kfold,
)
from .study import ForwardExample, build_forward_examples, missingness_summary

_METRIC_KEYS = ("accuracy", "balanced_accuracy", "brier", "roc_auc", "ece")


@dataclass
class StudyResult:
    payload: dict[str, Any]


def _evaluate(model, rows, labels, examples_side, *, split_seed: int):
    if labels.size == 0:
        return {"n": 0, "note": "empty partition; metrics withheld"}
    scores = model.predict_proba(rows)
    preds = model.predict(rows)
    summary = metrics_mod.summarize(labels, preds, scores)
    summary["uncertainty"] = metrics_mod.bootstrap_ci(labels, scores, preds, seed=split_seed)
    summary["subgroups"] = metrics_mod.subgroup_metrics(examples_side, labels, preds, scores)
    return summary


def _fit_split(
    train: list[ForwardExample],
    held_out: list[ForwardExample],
    *,
    split_seed: int,
) -> dict[str, Any]:
    encoder = FeatureEncoder().fit(train)
    x_train = encoder.transform(train)
    y_train = encoder.labels(train)
    x_held = encoder.transform(held_out)
    y_held = encoder.labels(held_out)

    naive = NaivePrevalence().fit(y_train)
    logistic = RidgeLogistic().fit(x_train, y_train)
    return {
        "features": list(encoder.feature_names),
        "impute": {"median_bmi": encoder.median_bmi, "median_sbp": encoder.median_sbp},
        "naive": naive,
        "logistic": logistic,
        "models": {
            naive.name: {
                "train": _evaluate(naive, x_train, y_train, train, split_seed=split_seed),
                "test": _evaluate(naive, x_held, y_held, held_out, split_seed=split_seed),
                "prevalence": naive.prevalence,
                "majority_class": naive.majority,
            },
            logistic.name: {
                "train": _evaluate(logistic, x_train, y_train, train, split_seed=split_seed),
                "test": _evaluate(logistic, x_held, y_held, held_out, split_seed=split_seed),
            },
        },
        "y_held": y_held,
        "naive_scores": naive.predict_proba(x_held),
        "naive_preds": naive.predict(x_held),
        "logistic_scores": logistic.predict_proba(x_held),
        "logistic_preds": logistic.predict(x_held),
    }


def _average_bound(items: list[dict[str, Any]], metric: str, side: str) -> float | None:
    values = [
        item[metric][side]
        for item in items
        if item.get(metric) and item[metric].get(side) is not None
    ]
    if not values:
        return None
    return float(sum(values) / len(values))


def _average_summaries(summaries: list[dict[str, Any]]) -> dict[str, Any]:
    """Mean of per-fold training metrics. ``n`` is the mean fold size."""
    n = int(round(sum(int(item["n"]) for item in summaries) / len(summaries)))
    n_positive = int(round(sum(int(item["n_positive"]) for item in summaries) / len(summaries)))
    n_positive = min(n_positive, n)
    averaged: dict[str, Any] = {
        "n": n,
        "n_positive": n_positive,
        "n_negative": n - n_positive,
        "note": "mean of per-fold training metrics; each subject is left out of one fold",
        "subgroups": {},
    }
    for key in _METRIC_KEYS:
        values = [item[key] for item in summaries if item.get(key) is not None]
        averaged[key] = float(sum(values) / len(values)) if values else None
    intervals = [item.get("uncertainty") or {} for item in summaries]
    averaged["uncertainty"] = {
        metric: {
            "lo": _average_bound(intervals, metric, "lo"),
            "hi": _average_bound(intervals, metric, "hi"),
        }
        for metric in ("accuracy", "roc_auc")
    }
    return averaged


def _pooled_summary(
    examples: list[ForwardExample],
    labels: np.ndarray,
    preds: np.ndarray,
    scores: np.ndarray,
    *,
    split_seed: int,
) -> dict[str, Any]:
    if labels.size == 0:
        return {"n": 0, "note": "empty partition; metrics withheld"}
    summary = metrics_mod.summarize(labels, preds, scores)
    summary["uncertainty"] = metrics_mod.bootstrap_ci(labels, scores, preds, seed=split_seed)
    summary["subgroups"] = metrics_mod.subgroup_metrics(examples, labels, preds, scores)
    summary["note"] = "pooled out-of-fold predictions; each subject is scored once"
    return summary


def _run_subject_kfold(
    examples: list[ForwardExample],
    flow: dict[str, int],
    tables: dict[str, pd.DataFrame],
    *,
    n_splits: int,
    split_seed: int,
    train_fraction: float,
    generator_seed: int | None,
    n_subjects: int | None,
    input_path: str | None,
    contract_version: str,
) -> StudyResult:
    folds = subject_stratified_kfold(examples, n_splits=n_splits, seed=split_seed)
    fitted = [_fit_split(train, validation, split_seed=split_seed) for train, validation in folds]

    oof_examples: list[ForwardExample] = []
    labels: list[np.ndarray] = []
    naive_scores: list[np.ndarray] = []
    naive_preds: list[np.ndarray] = []
    logistic_scores: list[np.ndarray] = []
    logistic_preds: list[np.ndarray] = []
    fold_rows: list[dict[str, Any]] = []
    for index, ((train, validation), scored) in enumerate(zip(folds, fitted, strict=True)):
        oof_examples.extend(validation)
        labels.append(scored["y_held"])
        naive_scores.append(scored["naive_scores"])
        naive_preds.append(scored["naive_preds"])
        logistic_scores.append(scored["logistic_scores"])
        logistic_preds.append(scored["logistic_preds"])
        positive = sum(int(ex.outcome) for ex in validation)
        fold_rows.append(
            {
                "fold": index,
                "n_train": len(train),
                "n_validation": len(validation),
                "train_subjects": len({ex.subject_key for ex in train}),
                "validation_subjects": len({ex.subject_key for ex in validation}),
                "validation_positive_rate": positive / len(validation),
                "median_bmi": scored["impute"]["median_bmi"],
                "median_sbp": scored["impute"]["median_sbp"],
            }
        )

    y_oof = np.concatenate(labels)
    naive_score = np.concatenate(naive_scores)
    naive_pred = np.concatenate(naive_preds)
    logistic_score = np.concatenate(logistic_scores)
    logistic_pred = np.concatenate(logistic_preds)
    prevalence = float(sum(item["naive"].prevalence for item in fitted) / len(fitted))
    payload: dict[str, Any] = {
        "lineage": lineage(
            split_seed=split_seed,
            train_fraction=train_fraction,
            generator_seed=generator_seed,
            n_subjects=n_subjects,
            input_path=input_path,
            contract_version=contract_version,
            n_splits=n_splits,
        ),
        "cohort_flow": flow,
        "split": {
            "mode": "subject_stratified_kfold",
            "n_splits": n_splits,
            "n_examples": len(oof_examples),
            "folds": fold_rows,
        },
        "missingness": missingness_summary(tables),
        "features": fitted[0]["features"],
        "impute": {
            "median_bmi": float(sum(item["impute"]["median_bmi"] for item in fitted) / len(fitted)),
            "median_sbp": float(sum(item["impute"]["median_sbp"] for item in fitted) / len(fitted)),
        },
        "models": {
            "naive_prevalence": {
                "train": _average_summaries(
                    [item["models"]["naive_prevalence"]["train"] for item in fitted]
                ),
                "test": _pooled_summary(
                    oof_examples, y_oof, naive_pred, naive_score, split_seed=split_seed
                ),
                "prevalence": prevalence,
                "majority_class": 1 if prevalence >= 0.5 else 0,
            },
            "ridge_logistic": {
                "train": _average_summaries(
                    [item["models"]["ridge_logistic"]["train"] for item in fitted]
                ),
                "test": _pooled_summary(
                    oof_examples, y_oof, logistic_pred, logistic_score, split_seed=split_seed
                ),
            },
        },
        "limitations": [
            "Rows are synthetic; distributions are invented and not derived from patients.",
            "The 140 mmHg cut is a labelling rule for this study, not a diagnostic claim.",
            "On this generator, prior SBP is highly informative by construction; a strong "
            "test AUC is not evidence of a real clinical signal.",
            "Ridge logistic is a transparent baseline, not a candidate for deployment.",
            "Subject-level folds block subject leakage; features are earlier than the outcome.",
            "Imputation medians are fit on each training fold only. "
            "Tiny subgroup cells report n only.",
            "Calibration (ECE) on small folds is noisy; treat it as a diagnostic.",
            "No causal, fairness, or clinical-effectiveness claim is supported by this run.",
            KFOLD_RESEARCH_ONLY,
        ],
    }
    return StudyResult(payload=payload)


def run_study(
    tables: dict[str, pd.DataFrame],
    *,
    split_seed: int,
    train_fraction: float = 0.7,
    generator_seed: int | None = None,
    n_subjects: int | None = None,
    input_path: str | None = None,
    contract_version: str | None = None,
    n_splits: int | None = None,
) -> StudyResult:
    contract = load_contract(contract_version) if contract_version else load_contract()
    validate(tables, contract)
    examples, flow = build_forward_examples(tables, contract)
    # Halt on leaked dates before the split or any model sees the examples.
    assert_no_time_leakage(examples)
    assert_no_time_leakage_against_tables(examples, tables)
    if n_splits is not None:
        return _run_subject_kfold(
            examples,
            flow,
            tables,
            n_splits=n_splits,
            split_seed=split_seed,
            train_fraction=train_fraction,
            generator_seed=generator_seed,
            n_subjects=n_subjects,
            input_path=input_path,
            contract_version=contract.version,
        )
    train, test = subject_split(examples, seed=split_seed, train_fraction=train_fraction)
    assert_no_subject_leakage(train, test)
    assert_no_time_leakage(train)
    assert_no_time_leakage(test)
    assert_no_cross_split_time_leakage(train, test)

    encoder = FeatureEncoder().fit(train)
    x_train = encoder.transform(train)
    y_train = encoder.labels(train)
    x_test = encoder.transform(test)
    y_test = encoder.labels(test)

    naive = NaivePrevalence().fit(y_train)
    logistic = RidgeLogistic().fit(x_train, y_train)

    def evaluate(model, rows, labels, examples_side):
        if labels.size == 0:
            return {"n": 0, "note": "empty partition; metrics withheld"}
        scores = model.predict_proba(rows)
        preds = model.predict(rows)
        summary = metrics_mod.summarize(labels, preds, scores)
        summary["uncertainty"] = metrics_mod.bootstrap_ci(labels, scores, preds, seed=split_seed)
        summary["subgroups"] = metrics_mod.subgroup_metrics(examples_side, labels, preds, scores)
        return summary

    payload: dict[str, Any] = {
        "lineage": lineage(
            split_seed=split_seed,
            train_fraction=train_fraction,
            generator_seed=generator_seed,
            n_subjects=n_subjects,
            input_path=input_path,
            contract_version=contract.version,
        ),
        "cohort_flow": flow,
        "split": {
            "n_train": len(train),
            "n_test": len(test),
            "train_subjects": len({ex.subject_key for ex in train}),
            "test_subjects": len({ex.subject_key for ex in test}),
        },
        "missingness": missingness_summary(tables),
        "features": list(encoder.feature_names),
        "impute": {"median_bmi": encoder.median_bmi, "median_sbp": encoder.median_sbp},
        "models": {
            naive.name: {
                "train": evaluate(naive, x_train, y_train, train),
                "test": evaluate(naive, x_test, y_test, test),
                "prevalence": naive.prevalence,
                "majority_class": naive.majority,
            },
            logistic.name: {
                "train": evaluate(logistic, x_train, y_train, train),
                "test": evaluate(logistic, x_test, y_test, test),
            },
        },
        "limitations": [
            "Rows are synthetic; distributions are invented and not derived from patients.",
            "The 140 mmHg cut is a labelling rule for this study, not a diagnostic claim.",
            "On this generator, prior SBP is highly informative by construction; a strong "
            "test AUC is not evidence of a real clinical signal.",
            "Ridge logistic is a transparent baseline, not a candidate for deployment.",
            "Subject-level split blocks subject leakage; features are earlier than the outcome.",
            "Imputation medians are fit on train only. Tiny subgroup cells report n only.",
            "Calibration (ECE) on small test sets is noisy; treat it as a diagnostic.",
            "No causal, fairness, or clinical-effectiveness claim is supported by this run.",
        ],
    }
    return StudyResult(payload=payload)
