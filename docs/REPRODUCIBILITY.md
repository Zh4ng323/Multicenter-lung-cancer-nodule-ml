# Reproducibility

## Environment

The locked artifact was created and the paper analyses were run under:

- **Python 3.10**
- scikit-learn **1.3.2**, pandas **1.4.4**, numpy **1.24.4**, scipy **1.10.1**
- xgboost 2.1.2, lightgbm 4.5.0, shap 0.44.1, joblib 1.4.2

Recreate exactly:

```bash
conda env create -f environment.yml && conda activate lungcancer-ml
```

(or `pip install -r requirements-analysis.txt` for the analysis subset).

**Do not bump scikit-learn / pandas** without re-validating: the locked
`best_model.joblib` was pickled under these versions, and `pandas.read_csv`
dtype inference on the slash-notation pulse columns can shift across versions,
perturbing AUCs.

## Random seeds

All stochastic steps use `random_state=42`:

- `train_test_split(test_size=0.2, random_state=42, stratify=y)` — the 80:20
  internal split.
- `StratifiedKFold(n_splits=10, shuffle=True, random_state=42)` — CV.
- Every stochastic estimator (XGBoost, LightGBM, RandomForest, GBDT, MLP, ANN,
  LR) passes `random_state=42`.
- The subset/test-sample generators (`make_synthetic_data.py`,
  `make_test_samples.py`) use `numpy.random.RandomState(42)`.

Kernel SHAP background sampling is the only non-seeded step, but the exact
background prototypes are shipped in `models/locked_svm/shap_background.npy`,
so SHAP re-explanation is reproducible.

## What reproduces, and how

| Paper result | How to reproduce | Reproducibility |
|---|---|---|
| Table 2 (9-model 10-fold CV) | `python analysis/01_train_multimodal.py` → `results/run_*/cv_results_summary.csv` + `CI汇总/Table2_*.csv` | exact (needs full dev matrix) |
| Table 3 (locked SVM internal + external) | same script, `CI汇总/Table3_SVM_internal_external_CI.csv`; check via `make verify` on the released subsets | exact / approximate |
| Supp risk-strata CI | same script, `CI汇总/Supp_RiskStrata_*.csv` | exact |
| Figs. 9–10, Table 7 (SHAP / drop-one / LASSO surrogate) | `python analysis/02_interp_three_layer.py` | exact |

## Quick check

```bash
make verify
```

Loads the shipped locked SVM and the released synthetic subsets and checks
internal AUC ≥ 0.88, external AUC ≥ 0.90, and a monotone lung-cancer proportion
across the low / medium / high score strata. Runs in a few seconds.

## Full retraining

Requires the complete source datasets (development n=1580, external n=843),
which are available from the corresponding author on reasonable request.

```bash
make train      # ~15 min: 9-model grid search, lock SVM, eval, SHAP, CIs
make interp     # 3-layer evidence chain
```

Outputs land in `results/` (gitignored).

## Test samples

```bash
make samples    # regenerates test_samples/*.csv from data/external_subset.csv
```

Each sample is verified against the locked SVM to confirm it lands in the
intended risk stratum before being written.

## Determinism caveats

- The locked SVM's reported CV AUC (0.9499 in `model_metadata.json`) matches
  the paper's Table 2 SVM row (0.950 mean fold AUC) to rounding.
- Subset AUCs differ from full-cohort AUCs because of sampling; the
  `verify_performance.py` thresholds account for this and check approximate
  consistency only.
