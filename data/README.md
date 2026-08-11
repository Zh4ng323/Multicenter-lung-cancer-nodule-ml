# Synthetic data subsets

The two CSVs here are fully **synthetic**, generated from per-class
multivariate Gaussian statistics (means and Ledoit-Wolf shrunk covariances in
the standardized feature space) of the real development and external cohorts.
No real participant rows are included.

| File | Rows | Class (LC / BPN) |
|---|---|---|
| `internal_test_subset.csv` | 300 | 146 / 154 |
| `external_subset.csv` | 150 | 65 / 85 |

Columns match the locked model exactly: 121 features (slash-notation pulse
ratios) + `target` (1 = lung cancer, 0 = benign pulmonary nodule).

Scored with the locked SVM, the synthetic subsets show ballpark discrimination
consistent with the paper (internal AUC ≈ 0.93, external AUC ≈ 0.96) and a
monotone increase in lung-cancer proportion across the low / medium /
high risk strata. Run `python analysis/verify_performance.py` (or `make verify`)
to confirm.

Regenerate with `analysis/make_synthetic_data.py` (requires access to the
complete source datasets).

## Slash-notation warning

Pulse ratio features use the forward slash: `h4/h1`, `h1/t1`, `h3/h1`, `w2/t`,
`t1/t`, `w1/t`. Renaming these to hyphens breaks column matching and silently
degrades predictions.

## Feature definitions

121 features (canonical source: `web_app/feature_config.py`):

- Tongue (28): `TB-*`, `TC-*`, `Per-all`, `Per-part`
- Face (83): `lipcolor-*`, `wholecolor-*`, `color-*`, `GLCM-*`
- Pulse (10): `h4/h1, h4, h5, h1/t1, h3/h1, t4, w2/t, t1/t, t5, w1/t`

## License

CC BY 4.0 — see `LICENSE-data.txt`.
