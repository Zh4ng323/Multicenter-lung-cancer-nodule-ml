# Paper-table provenance

Which script reproduces which paper element.

## Reproducible from this release

| Paper element | Reproduced by | Released data | Notes |
|---|---|---|---|
| Table 1 (cohort characteristics) | n/a (descriptive) | — | Demographic detail is not in the released data. |
| Table 2 (9-model stratified 10-fold CV) | `analysis/01_train_multimodal.py` | full dev matrix (on request) | Writes `cv_results_summary.csv` + `CI汇总/Table2_CV_dual_CI.csv`. |
| Table 3 (locked SVM internal + external) | `analysis/01_train_multimodal.py`; check via `analysis/verify_performance.py` | synthetic subsets (`make verify`) | Full-cohort internal 0.926 / external 0.954; synthetic subset 0.925 / 0.957. |
| Supp risk-strata CI | `analysis/01_train_multimodal.py` | full dev + external (on request) | `CI汇总/Supp_RiskStrata_*.csv`. |
| Fig. 9 (SHAP of locked SVM) | `analysis/01_train_multimodal.py` + `02_interp_three_layer.py` (L1) | full dev matrix | Top features: `h1/t1`, `h4/h1`, `t1/t` (all pulse-derived). |
| Fig. 10, Table 7 (drop-one + LASSO surrogate) | `analysis/02_interp_three_layer.py` (L2 + L3) | full dev matrix | Face largest paired ΔAUC (0.087); pulse ΔAUC CI crosses zero. |
| Figs. 11–12 (score strata + case montage) | `analysis/01_train_multimodal.py` (risk-strata outputs) | synthetic subsets | Monotone LC proportion low≪mid≪high on both subsets. |

## Locked-SVM hyperparameters

Grid search selected and the pipeline then froze:

```
SVC(C=1.0, kernel='rbf', gamma='scale', class_weight='balanced',
    probability=True, random_state=42)
```

These are encoded in `analysis/config.py::LOCKED_SVM_PARAMS` and were used
unchanged for internal and external evaluation.
