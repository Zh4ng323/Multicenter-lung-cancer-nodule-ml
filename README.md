# Lung cancer versus benign pulmonary nodules — multimodal tongue/face/pulse study

This repository provides the analysis code, locked model, synthetic validation
data, and research prototype associated with the manuscript:

> Multicenter development and validation of machine learning to distinguish
> lung cancer from benign pulmonary nodules using objective tongue, facial,
> and pulse features.

The study screened 6,018 participant records across multiple centers and
retained 1,580 participants with complete multimodal data as the development
cohort and 843 as the external validation cohort. From 121 LASSO-selected
features (28 tongue, 83 face, 10 pulse), nine algorithms — XGBoost, LightGBM,
random forest, logistic regression, ANN, SVM, KNN, GBDT, and MLP — were
compared by stratified 10-fold cross-validation, and an RBF support vector
machine (`C=1.0`, `gamma=scale`, `class_weight=balanced`) was selected and
locked before held-out internal and external evaluation.

| AUC | Pooled out-of-fold | Internal test | External |
|---|---|---|---|
| Paper (full cohort) | 0.951 (0.939–0.962) | 0.926 (0.898–0.954) | 0.954 (0.942–0.967) |

Risk strata use fixed cut points 0.40 / 0.80 (low / medium / high).
Kernel SHAP ranked three pulse-derived variables (`h1/t1`, `h4/h1`, `t1/t`) as
the strongest individual features; modality-level analyses identified the face
as the principal source of discrimination.

## Quick start

```bash
conda env create -f environment.yml && conda activate lungcancer-ml
make verify
```

The `verify` target loads the locked model and scores the released synthetic
subsets, providing an executable verification of model loading, inference, and
ballpark discrimination on synthetic data (internal AUC ≈ 0.93, external
AUC ≈ 0.96).

## Web app

`web_app/` contains the source code of the research prototype described in the
paper (multimodal data entry, inference, risk stratification, and SHAP
explanation). The prototype is deployed at **http://106.15.62.250:7860/** — you
can read the source here or try the running site directly with the demo
account:

- Username: `doctor`
- Password: `diagnosis123`

`test_samples/` provides three single-row examples (low / medium / high risk)
for upload on the site.

**The web app is a research prototype, not a cleared clinical device.** The
score strata are exploratory operating points derived from the enriched study
cohorts, not validated clinical risk thresholds, and should not be used for
clinical decision-making. Prospective validation is required before any
clinical use.

To run locally:

```bash
cd web_app && docker compose up --build
```

## Reproducibility notes

- Environment: Python 3.10, scikit-learn 1.3.2, pandas 1.4.4 (`environment.yml`).
- Random seed 42; 80:20 stratified split; 10-fold cross-validation.
- Pulse ratios use slash notation (`h4/h1`, `h1/t1`, `h3/h1`, `w2/t`, `t1/t`,
  `w1/t`); renaming these to hyphens breaks column matching and degrades
  predictions.
- The released data in `data/` are synthetic (see `data/README.md`). Full-cohort
  reproduction requires the complete source datasets, which are available from
  the corresponding author on reasonable request, subject to institutional and
  ethics approval.

## Full reproduction

The complete pipeline — nine-model grid search, cross-validation, internal and
external evaluation, SHAP interpretation, and risk-stratification confidence
intervals — can be reproduced from the source datasets:

```bash
make train
make interp
make subset
make samples
```

## Layout

```
analysis/      training, CI, interpretation scripts, config
data/          synthetic subsets (CC BY 4.0)
models/        locked_svm/ 4-file artifact
web_app/       Gradio research prototype
test_samples/  3 risk-stratum samples
docs/          provenance and reproducibility notes
tests/         pytest suite
```

## License

- Code and models: MIT (`LICENSE`)
- Data (`data/`): CC BY 4.0 (`data/LICENSE-data.txt`)

The data in `data/` are synthetic — 300 internal-test and 150 external rows
generated from per-class statistics of the real cohorts; no real participant
rows are included (see `data/README.md`).
