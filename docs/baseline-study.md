# Leakage-safe baseline study

**Study id:** `elevated-sbp-forward-v1`
**Contract:** `datalife-synthetic-longitudinal` `0.1.0`

Depends on the generator and tables from
[#2](https://github.com/datalife-ehealth/datalife-analytics-downstream/issues/2) /
[PR #5](https://github.com/datalife-ehealth/datalife-analytics-downstream/pull/5).
See [generator.md](generator.md) for how those tables are produced.

## Question

Predict whether a synthetic subject's *later* encounter has elevated systolic blood
pressure (SBP >= 140 mmHg) using only *earlier* encounters of that same subject.
Covariates: sex, age band, region, last encounter type, prior BMI, prior SBP, and
structural missingness rates for those markers.

This is a labelling rule on generated data. It is not a diagnostic definition, a
fairness evaluation, or an operational signal.

## Why this split

- **Subject-aware:** each `subject_key` is assigned wholly to train or test from a
  seeded shuffle. Tests fail if any subject appears on both sides.
- **Time-aware:** features are built from encounters strictly before the outcome
  timestamp. Tests fail if a feature timestamp is on or after the outcome, or if a
  buggy split puts a subject's later rows in train and earlier rows in test.
- **Train-only impute:** numeric medians are fit on train and applied to test.
- **Contract first:** `validate()` from the v0.1.0 contract runs before any split
  or fit. Schema mismatch and missing columns fail closed.

## Subject-stratified K-fold

`--k-folds` replaces the two-way split with a seeded subject-stratified K-fold.
Each `subject_key` sits in exactly one validation fold, and every train/validation
pair is checked for subject and time leakage. The fold count must be an integer
of at least 2. An empty cohort, or one with fewer subjects (or fewer subjects in
a class) than the fold count, fails closed. Omit `--k-folds` to keep the default
two-way split. `--train-fraction` applies only to that two-way split.

Folds are for aggregate research evaluation and are never for clinical
decision-making. Reported test metrics are pooled out-of-fold predictions: each
subject is scored once. They are not a diagnosis, a treatment recommendation, or
an operational signal.

```bash
datalife-baseline --seed 20260101 --subjects 80 --split-seed 7 \
    --k-folds 5 --out reports/generated
```

## Naive comparator

A constant predictor equal to the training positive rate (hard label = majority
class). The logistic baseline is reported next to it, not instead of it.

## One command

```bash
pip install -e ".[dev]" -c constraints.txt
datalife-baseline --seed 20260101 --subjects 80 --split-seed 7 \
    --out reports/generated
```

That run generates a v0.1.0 cohort (or you can pass `--input` to a directory written
by `datalife-fixtures generate`), builds forward examples, splits by subject, fits
both models, and writes `metrics.json` plus `report.md` under `reports/generated/`
(git-ignored). Against the committed fixture:

```bash
datalife-baseline --input tests/fixtures --prefix synthetic_v0_1_0_ \
    --split-seed 7 --out reports/generated
```

The fixture is small on purpose; use a generated cohort for less noisy metrics.

## Adding a model without breaking the contract

1. Keep contract `0.1.0` unless the field set actually changes.
2. Add a fit/predict object next to `NaivePrevalence` / `RidgeLogistic` that consumes
   the encoded rows from `FeatureEncoder`.
3. Register it in `run_study` so it shows up under `payload["models"]` beside the
   naive comparator.
4. Do not move imputation onto the test set. Do not split by row if a subject would
   land on both sides.

Generated reports, fitted weights, and notebook outputs stay out of Git.
