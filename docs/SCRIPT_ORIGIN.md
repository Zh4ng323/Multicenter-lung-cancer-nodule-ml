# Script origin map

The analysis scripts in this release were renamed from the author's working
files for clarity. This table preserves the audit trail back to the original
Chinese-dated filenames used during paper development.

| Release path | Original working filename | Role |
|---|---|---|
| `analysis/01_train_multimodal.py` | `0804-8精简版-最终-最佳模型SHAP.py` | Main pipeline: load data, 80:20 split, 9-model grid search, lock RBF-SVM (C=1.0, gamma=scale, class_weight=balanced), 10-fold CV, held-out internal + external evaluation, Kernel SHAP on the best model, risk-stratification CIs (Table 2, Table 3, Supp risk-strata CI), and deploy the 4-file artifact to `web_app/saved_models/`. |
| `analysis/02_interp_three_layer.py` | `0803-解释链-三层合一.py` | Three-layer evidence chain: L1 Kernel-SHAP attribution, L2 drop-one-modality paired ΔAUC, L3 LASSO sparse surrogate with stability selection (Figs. 9–10, Table 7). |
| `analysis/config.py` | (new) | Canonical configuration: repo-relative paths, seeds, locked-SVM hyperparameters, risk thresholds; imports the 121-feature list from `web_app/feature_config.py`. |
| `analysis/verify_performance.py` | (new) | Score the locked SVM on the released subsets and check performance against the paper. |
| `analysis/make_synthetic_data.py` | (new) | Generate the released synthetic subsets (~300 internal / ~150 external) from per-class statistics of the real cohorts. |
| `analysis/make_test_samples.py` | (new) | Pick one synthetic row per risk stratum as web-app demonstration samples. |

## What changed during the rename

Only the `__main__` entry points were patched — hardcoded absolute Windows
paths were replaced with `argparse` flags whose defaults come from
`analysis/config.py` (repo-relative via `pathlib`). The risk-stratum label in
`01_train_multimodal.py` was harmonized from "Intermediate" to "Medium" to
match the deployed web application's terminology. Every model-training,
evaluation, CI, and SHAP function body is otherwise the author's original code,
so numerical outputs are identical to the working files.

If you need to diff this release against the original working scripts, compare
`analysis/01_train_multimodal.py` to `0804-8精简版-最终-最佳模型SHAP.py` and
`analysis/02_interp_three_layer.py` to `0803-解释链-三层合一.py`; the only
differences are in the final `if __name__ == "__main__":` block of each.

## Scripts intentionally NOT included

These working files were excluded from the canonical release (see
`docs/TABLE_PROVENANCE.md` for the reasoning):

- `0803-CI-主-Table2_3.py`, `0803-CI-补充-Table4_7.py`,
  `0803-CI-补充-风险分层.py` — Table 2/3 CI is now integrated into
  `01_train_multimodal.py`; Tables 4–7 depend on single-modality/PSM datasets
  that are not part of this release.
- `0803-解释链-Lasso代理.py`, `0803-解释链-SHAP与Drop-One.py` — superseded by
  the combined `02_interp_three_layer.py`.
- `0504-8精简版-最终 copy.py`, `0504-8精简版-最终-original.py`,
  `0728-9-LightGBM-NaN-Robust.py`, `0729-单模态训练-舌面脉.py` — older or
  single-modality variants superseded by the canonical pipeline.
