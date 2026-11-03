"""End-to-end baseline study against the v0.1.0 generator and committed fixture."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from datalife_analytics.baseline.cli import main
from datalife_analytics.baseline.pipeline import run_study
from datalife_analytics.baseline.split import LeakageError, subject_stratified_kfold
from datalife_analytics.baseline.study import ForwardExample
from datalife_analytics.io import read_dataset


def test_run_study_includes_naive_comparator_and_limitations(small_tables, small_config):
    result = run_study(
        small_tables,
        split_seed=7,
        generator_seed=small_config.seed,
        n_subjects=small_config.n_subjects,
    )
    models = result.payload["models"]
    assert "naive_prevalence" in models
    assert "ridge_logistic" in models
    assert models["naive_prevalence"]["test"]["n"] > 0
    assert models["ridge_logistic"]["test"]["n"] > 0
    assert result.payload["limitations"]
    assert result.payload["lineage"]["contract_version"] == "0.1.0"
    assert result.payload["lineage"]["split_seed"] == 7
    assert result.payload["cohort_flow"]["n_subjects"] == small_config.n_subjects


def test_run_study_is_deterministic(small_tables):
    first = run_study(small_tables, split_seed=4, generator_seed=7, n_subjects=60)
    second = run_study(small_tables, split_seed=4, generator_seed=7, n_subjects=60)
    assert (
        first.payload["models"]["ridge_logistic"]["test"]["accuracy"]
        == second.payload["models"]["ridge_logistic"]["test"]["accuracy"]
    )
    assert first.payload["split"] == second.payload["split"]


def test_impute_medians_are_finite(small_tables):
    result = run_study(small_tables, split_seed=7)
    impute = result.payload["impute"]
    assert isinstance(impute["median_bmi"], float)
    assert isinstance(impute["median_sbp"], float)


def test_cli_one_command_writes_json_and_markdown(tmp_path: Path):
    out = tmp_path / "reports"
    code = main(["--seed", "20260101", "--subjects", "40", "--split-seed", "7", "--out", str(out)])
    assert code == 0
    assert (out / "metrics.json").is_file()
    report = (out / "report.md").read_text("utf-8")
    assert "Naive comparator, test" in report
    assert "Limitations" in report
    assert "0.1.0" in report


def test_cli_from_committed_fixture(tmp_path: Path):
    out = tmp_path / "from-fixture"
    code = main(
        [
            "--input",
            str(Path("tests/fixtures")),
            "--prefix",
            "synthetic_v0_1_0_",
            "--split-seed",
            "1",
            "--out",
            str(out),
        ]
    )
    assert code == 0
    text = (out / "metrics.json").read_text("utf-8")
    assert "tests/fixtures" in text
    assert "0.1.0" in text


def test_cli_rejects_bad_input(tmp_path: Path, capsys):
    code = main(["--input", str(tmp_path / "missing"), "--out", str(tmp_path / "out")])
    assert code == 2
    assert "error:" in capsys.readouterr().err


def test_read_dataset_feeds_the_pipeline():
    tables = read_dataset("tests/fixtures", prefix="synthetic_v0_1_0_")
    result = run_study(tables, split_seed=2, input_path="tests/fixtures")
    assert result.payload["cohort_flow"]["n_subjects"] == 25
    assert result.payload["lineage"]["contract_version"] == "0.1.0"


def _row(
    subject: str,
    outcome: int,
    *,
    outcome_date: str = "2024-06-01T09:00:00Z",
    feature_dates: tuple[str, ...] = ("2024-01-01T09:00:00Z",),
) -> ForwardExample:
    return ForwardExample(
        subject_key=subject,
        outcome=outcome,
        outcome_date=outcome_date,
        feature_end_date=feature_dates[-1],
        feature_dates=feature_dates,
        sex="F",
        age_band="45-59",
        region="R01",
        n_prior=len(feature_dates),
        last_encounter_type="routine",
        mean_bmi=24.0,
        last_sbp=120.0,
        missing_bmi_rate=0.0,
        missing_sbp_rate=0.0,
    )


def _cohort(n_neg: int, n_pos: int) -> list[ForwardExample]:
    rows = [_row(f"neg-{index:03d}", 0) for index in range(n_neg)]
    rows += [_row(f"pos-{index:03d}", 1) for index in range(n_pos)]
    return rows


def _fold_keys(folds):
    return [
        ([ex.subject_key for ex in train], [ex.subject_key for ex in validation])
        for train, validation in folds
    ]


def _subject_class_counts(validation: list[ForwardExample]) -> tuple[int, int]:
    subjects: dict[str, int] = {}
    for example in validation:
        subjects[example.subject_key] = int(example.outcome)
    n_pos = sum(label == 1 for label in subjects.values())
    return len(subjects) - n_pos, n_pos


def test_kfold_is_deterministic_for_a_repeated_seed():
    examples = _cohort(12, 12)
    first = _fold_keys(subject_stratified_kfold(examples, n_splits=4, seed=11))
    second = _fold_keys(subject_stratified_kfold(examples, n_splits=4, seed=11))
    assert first == second
    assert first != _fold_keys(subject_stratified_kfold(examples, n_splits=4, seed=12))


def test_kfold_balances_classes_and_keeps_each_subject_in_one_fold():
    examples = _cohort(20, 20)
    examples.append(_row("neg-000", 0, outcome_date="2024-06-02T09:00:00Z"))
    folds = subject_stratified_kfold(examples, n_splits=4, seed=7)
    assert len(folds) == 4

    seen: list[set[str]] = []
    for train, validation in folds:
        train_subjects = {ex.subject_key for ex in train}
        val_subjects = {ex.subject_key for ex in validation}
        assert train_subjects.isdisjoint(val_subjects)
        assert train_subjects | val_subjects == {ex.subject_key for ex in examples}
        neg, pos = _subject_class_counts(validation)
        assert neg == 5
        assert pos == 5
        assert sum(ex.subject_key == "neg-000" for ex in train + validation) == 2
        assert ("neg-000" in train_subjects) ^ ("neg-000" in val_subjects)
        val_keys = [ex.subject_key for ex in validation]
        assert val_keys == [ex.subject_key for ex in examples if ex.subject_key in val_subjects]
        seen.append(val_subjects)

    assert set.union(*seen) == {ex.subject_key for ex in examples}
    for index, left in enumerate(seen):
        for right in seen[index + 1 :]:
            assert left.isdisjoint(right)


def test_kfold_class_counts_differ_by_at_most_one():
    folds = subject_stratified_kfold(_cohort(10, 11), n_splits=4, seed=3)
    neg = []
    pos = []
    for _, validation in folds:
        n_neg, n_pos = _subject_class_counts(validation)
        neg.append(n_neg)
        pos.append(n_pos)
    assert max(neg) - min(neg) <= 1
    assert max(pos) - min(pos) <= 1


@pytest.mark.parametrize("n_splits", [0, 1, -4, True, 2.5])
def test_kfold_rejects_invalid_n_splits(n_splits):
    with pytest.raises(ValueError, match="n_splits"):
        subject_stratified_kfold(_cohort(6, 6), n_splits=n_splits, seed=1)


def test_kfold_rejects_empty_and_undersized_cohorts():
    with pytest.raises(ValueError, match="empty"):
        subject_stratified_kfold([], n_splits=2, seed=1)
    with pytest.raises(ValueError, match="undersized"):
        subject_stratified_kfold(_cohort(2, 1), n_splits=4, seed=1)
    with pytest.raises(ValueError, match="undersized"):
        subject_stratified_kfold(_cohort(6, 1), n_splits=3, seed=1)


def test_kfold_rejects_time_leakage():
    examples = _cohort(4, 3)
    examples.append(
        _row(
            "pos-003",
            1,
            outcome_date="2024-03-01T09:00:00Z",
            feature_dates=("2024-03-01T09:00:00Z",),
        )
    )
    with pytest.raises(LeakageError, match="time leakage"):
        subject_stratified_kfold(examples, n_splits=2, seed=1)


def test_default_study_stays_a_two_way_split(small_tables):
    result = run_study(small_tables, split_seed=7)
    assert "mode" not in result.payload["split"]
    assert "n_splits" not in result.payload["lineage"]
    assert result.payload["split"]["n_train"] > 0
    assert result.payload["split"]["n_test"] > 0


def test_run_study_kfold_is_an_aggregate_and_is_deterministic(small_tables):
    first = run_study(small_tables, split_seed=4, n_splits=3)
    second = run_study(small_tables, split_seed=4, n_splits=3)
    split = first.payload["split"]
    assert split == second.payload["split"]
    assert split["mode"] == "subject_stratified_kfold"
    assert split["n_splits"] == 3
    assert len(split["folds"]) == 3
    assert sum(fold["n_validation"] for fold in split["folds"]) == split["n_examples"]
    assert first.payload["models"]["ridge_logistic"]["test"]["n"] == split["n_examples"]
    assert first.payload["lineage"]["n_splits"] == 3
    assert any("clinical decision-making" in item for item in first.payload["limitations"])
    for fold in split["folds"]:
        assert fold["n_train"] > 0
        assert fold["n_validation"] > 0
        assert fold["validation_subjects"] > 0


def test_cli_help_keeps_kfold_optional(capsys):
    with pytest.raises(SystemExit) as exc:
        main(["--help"])
    assert exc.value.code == 0
    help_text = capsys.readouterr().out
    assert "--k-folds" in help_text
    assert "clinical decision-making" in help_text


def test_cli_kfold_writes_research_only_report(tmp_path: Path):
    out = tmp_path / "kfold"
    code = main(
        [
            "--seed",
            "20260101",
            "--subjects",
            "40",
            "--split-seed",
            "7",
            "--k-folds",
            "2",
            "--out",
            str(out),
        ]
    )
    assert code == 0
    report = (out / "report.md").read_text("utf-8")
    assert "never for clinical decision-making" in report
    metrics = json.loads((out / "metrics.json").read_text("utf-8"))
    assert metrics["split"]["mode"] == "subject_stratified_kfold"
    assert metrics["split"]["n_splits"] == 2
    assert metrics["lineage"]["n_splits"] == 2


def test_cli_rejects_invalid_kfold(tmp_path: Path, capsys):
    code = main(
        [
            "--seed",
            "1",
            "--subjects",
            "20",
            "--k-folds",
            "1",
            "--out",
            str(tmp_path / "out"),
        ]
    )
    assert code == 2
    assert "n_splits" in capsys.readouterr().err
