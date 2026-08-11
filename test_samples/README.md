# Web-app demonstration test samples

Three single-row CSVs (121 features, no `target` column) ready to upload
into the research web application. Each lands in a different risk stratum and
showcases the distribution trends the paper describes for that group.

Generated reproducibly by `analysis/make_test_samples.py` from
`data/external_subset.csv` using the locked SVM.

Risk strata use fixed exploratory cut points: low <= 0.40,
medium 0.40-0.80, high > 0.80 (calibrated probability).

| Sample | Calibrated proba | Stratum | True label |
|---|---|---|---|
| `sample_low_risk.csv` | 0.0010 | **low-risk** | Benign pulmonary nodule |
| `sample_medium_risk.csv` | 0.6250 | **medium-risk** | Benign pulmonary nodule |
| `sample_high_risk.csv` | 0.9968 | **high-risk** | Lung cancer |

## Showcased trend features

Paper direction (lung cancer vs benign nodules):
- Higher in lung cancer: `h1/t1`, `h4`, `h5`, `color-G-3`, `wholecolor-Cr`
- Lower in lung cancer: `Per-all`, `color-Cb-1`, `GLCM-asm_3`, `h4/h1`, `w1/t`, `t1/t`

Per-sample values (high-risk should look lung-cancer-like, low-risk the
reverse, medium in between):

### `sample_low_risk.csv` (low-risk)
- `h1/t1` = 1.2235
- `color-G-3` = 105.3668
- `wholecolor-Cr` = 150.0295
- `h4` = 1.3934
- `h5` = 0.0182
- `Per-all` = 0.3931
- `color-Cb-1` = 118.4005
- `GLCM-asm_3` = 0.2766
- `h4/h1` = 0.3354
- `w1/t` = 0.2073
- `t1/t` = 0.1594

### `sample_medium_risk.csv` (medium-risk)
- `h1/t1` = 3.6969
- `color-G-3` = 109.8518
- `wholecolor-Cr` = 154.1531
- `h4` = 5.5888
- `h5` = 0.7256
- `Per-all` = 0.2700
- `color-Cb-1` = 112.6247
- `GLCM-asm_3` = 0.1766
- `h4/h1` = 0.3388
- `w1/t` = 0.1704
- `t1/t` = 0.1456

### `sample_high_risk.csv` (high-risk)
- `h1/t1` = 5.8547
- `color-G-3` = 93.9083
- `wholecolor-Cr` = 154.8331
- `h4` = 3.8326
- `h5` = 1.6755
- `Per-all` = 0.1430
- `color-Cb-1` = 107.0787
- `GLCM-asm_3` = 0.1302
- `h4/h1` = 0.1529
- `w1/t` = 0.1796
- `t1/t` = 0.1528

## Usage

Upload any of these CSVs in the web app's CSV-import panel (deployed
site and demo account are in the top-level README).

These samples are synthetic and are for software demonstration, not
clinical use.