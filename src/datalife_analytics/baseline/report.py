"""Machine-readable JSON metrics plus a short Markdown report."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from .split import KFOLD_RESEARCH_ONLY


def _fmt(value: object) -> str:
    if value is None:
        return "n/a"
    if isinstance(value, float):
        return f"{value:.3f}"
    return str(value)


def _metrics_table(side: dict[str, Any]) -> str:
    if side.get("n") == 0:
        return f"_empty partition: {side.get('note', 'metrics withheld')}_"
    auc_ci = (side.get("uncertainty") or {}).get("roc_auc") or {}
    acc_ci = (side.get("uncertainty") or {}).get("accuracy") or {}
    return "\n".join(
        [
            "| metric | value |",
            "|---|---|",
            f"| n | {side.get('n')} (pos {side.get('n_positive')}, neg {side.get('n_negative')}) |",
            f"| accuracy | {_fmt(side.get('accuracy'))} "
            f"(95% boot {_fmt(acc_ci.get('lo'))}–{_fmt(acc_ci.get('hi'))}) |",
            f"| balanced accuracy | {_fmt(side.get('balanced_accuracy'))} |",
            f"| Brier | {_fmt(side.get('brier'))} |",
            f"| ROC AUC | {_fmt(side.get('roc_auc'))} "
            f"(95% boot {_fmt(auc_ci.get('lo'))}–{_fmt(auc_ci.get('hi'))}) |",
            f"| ECE (5-bin) | {_fmt(side.get('ece'))} |",
        ]
    )


def to_markdown(payload: dict[str, Any]) -> str:
    lineage = payload["lineage"]
    flow = payload["cohort_flow"]
    split = payload["split"]
    naive = payload["models"]["naive_prevalence"]
    logistic = payload["models"]["ridge_logistic"]

    subgroup_lines = [
        "| group | level | n | accuracy | ROC AUC |",
        "|---|---|---|---|---|",
    ]
    for group, rows in (logistic.get("test", {}).get("subgroups") or {}).items():
        for row in rows:
            subgroup_lines.append(
                f"| {group} | {row['level']} | {row['n']} | "
                f"{_fmt(row.get('accuracy'))} | {_fmt(row.get('roc_auc'))} |"
            )

    missing_lines = ["| table.column | missing fraction |", "|---|---|"]
    for table, cols in payload["missingness"].items():
        for name, fraction in cols.items():
            missing_lines.append(f"| `{table}.{name}` | {fraction:.3f} |")

    limitations = "\n".join(f"- {item}" for item in payload["limitations"])
    kfold = split.get("mode") == "subject_stratified_kfold"
    if kfold:
        research_note = f"\n> {KFOLD_RESEARCH_ONLY}\n"
        n_splits_row = f"| n_splits | {lineage['n_splits']} |\n"
        cohort_split = (
            f"| folds | {split['n_splits']} |\n| out-of-fold examples | {split['n_examples']} |"
        )
        fold_lines = [
            "| fold | train examples (subjects) | validation examples (subjects) "
            "| validation positive rate |",
            "|---|---|---|---|",
        ]
        for fold in split["folds"]:
            fold_lines.append(
                f"| {fold['fold']} | {fold['n_train']} ({fold['train_subjects']}) | "
                f"{fold['n_validation']} ({fold['validation_subjects']}) | "
                f"{fold['validation_positive_rate']:.3f} |"
            )
        fold_section = (
            "\n## Folds\n\n"
            "Train metrics below are means across folds. Test metrics are pooled "
            "out-of-fold predictions, so each subject is scored once.\n\n"
            + "\n".join(fold_lines)
            + "\n"
        )
    else:
        research_note = "\n"
        n_splits_row = ""
        cohort_split = (
            f"| train examples (subjects) | {split['n_train']} ({split['train_subjects']}) |\n"
            f"| test examples (subjects) | {split['n_test']} ({split['test_subjects']}) |"
        )
        fold_section = ""

    return f"""# Baseline study report

**Study:** `{lineage["study_id"]}`
**Contract:** `{lineage["contract"]}` `{lineage["contract_version"]}`

{lineage["study_question"]}

> {lineage["non_use"]}
{research_note}
## Lineage

| key | value |
|---|---|
| package | {lineage["package_version"]} |
| python | {lineage["python"]} |
| numpy / pandas | {lineage["numpy"]} / {lineage["pandas"]} |
| generator_seed | {lineage["generator_seed"]} |
| input_path | {lineage["input_path"]} |
| n_subjects | {lineage["n_subjects"]} |
| split_seed | {lineage["split_seed"]} |
| train_fraction | {lineage["train_fraction"]} |
{n_splits_row}| SBP threshold | {lineage["elevated_sbp_threshold"]} |

## Cohort flow

| step | n |
|---|---|
| subjects | {flow["n_subjects"]} |
| encounters | {flow["n_encounters"]} |
| observations | {flow["n_observations"]} |
| dropped (single encounter) | {flow["dropped_single_encounter"]} |
| dropped (no outcome SBP) | {flow["dropped_no_outcome_sbp"]} |
| dropped (no prior encounter) | {flow["dropped_no_prior"]} |
| forward examples | {flow["n_examples"]} |
{cohort_split}

{fold_section}## Missingness

{chr(10).join(missing_lines)}

## Naive comparator (prevalence / majority)

Train prevalence = {_fmt(naive["prevalence"])}; majority class = {naive["majority_class"]}.

### Naive comparator, train

{_metrics_table(naive["train"])}

### Naive comparator, test

{_metrics_table(naive["test"])}

## Ridge logistic baseline

Impute medians (train only): BMI `{payload["impute"]["median_bmi"]}`,
last SBP `{payload["impute"]["median_sbp"]}`.

### Logistic baseline, train

{_metrics_table(logistic["train"])}

### Logistic baseline, test

{_metrics_table(logistic["test"])}

## Subgroup behaviour (logistic, test)

Cells with n < 8 report n only.

{chr(10).join(subgroup_lines)}

## Limitations

{limitations}
"""


def write_reports(payload: dict[str, Any], directory: str | Path) -> tuple[Path, Path]:
    target = Path(directory)
    target.mkdir(parents=True, exist_ok=True)
    json_path = target / "metrics.json"
    md_path = target / "report.md"
    json_path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    md_path.write_text(to_markdown(payload), encoding="utf-8")
    return json_path, md_path
