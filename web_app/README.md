# Web app — multimodal lung-cancer risk prototype

The deployed prototype (`http://106.15.62.250:7860/`, demo account `doctor` /
`diagnosis123`) and the source here are described in the top-level README.
This page covers what is needed to run or modify the app.

## Run locally

```bash
pip install -r requirements.txt
python app.py              # open http://localhost:7860
# or: docker compose up --build
```

The app loads the locked SVM from `saved_models/20260804_2201_SVM/` and a
LightGBM model for the algorithm-level sensitivity view; select between them
in the UI.

## Test samples

Upload any file from `../test_samples/` via the CSV-import panel:

| File | Expected risk level |
|---|---|
| `sample_low_risk.csv` | Low (calibrated proba ≤ 0.40) |
| `sample_medium_risk.csv` | Medium (0.40–0.80) |
| `sample_high_risk.csv` | High (> 0.80) |

CSV format: exactly the 121 feature columns in `feature_config.py`
(`FEATURE_NAMES`) order; a `target` column, if present, is ignored. Pulse
ratios use slash notation (`h4/h1`, `h1/t1`, `h3/h1`, `w2/t`, `t1/t`, `w1/t`);
renaming to hyphens breaks column matching.

## Authentication

Auth uses Gradio's built-in authentication (`AUTH_ENABLED=true`) with accounts
stored in `users.json` as sha256 hashes. `users.template.json` includes the
`doctor` account (copy it to `users.json` on first run). Change the password
before any public deployment via `auth_manager.py`. Locally, auth is off by
default (all logins accepted).

| Variable | Default | Meaning |
|---|---|---|
| `AUTH_ENABLED` | `false` | Enable Gradio basic auth. |
| `LUNGAI_USERS_FILE` | `users.json` | Path to the user→hash JSON file. |
| `LUNGAI_USERS` | _(empty)_ | Inline users `user:pass,user:pass`. |
| `SERVER_NAME` | `0.0.0.0` | Bind host. |
| `SERVER_PORT` | `7860` | Bind port. |

## Files

| File | Role |
|---|---|
| `app.py` | Gradio UI: data entry, CSV import, inference, SHAP, risk display. |
| `predictor.py` | Loads the artifact, applies the calibrator, assigns risk level. |
| `risk_stratifier.py` | Low/medium/high mapping at fixed 0.40 / 0.80 cut points. |
| `feature_config.py` | **Canonical** 121 feature names + thresholds + seed. |
| `model_manager.py` | Saves/loads the 4-file model artifact. |
| `shap_explainer.py` | Kernel SHAP waterfall / importance displays. |
| `auth_manager.py` | Optional Gradio basic auth. |
| `saved_models/` | Shipped locked SVM + LightGBM artifacts. |

## Retraining

The web app ships pre-trained artifacts and does not retrain. Retrain from the
full cohorts with `python ../analysis/01_train_multimodal.py`; the fresh
artifact is written to `web_app/saved_models/<timestamp>_<model>/` and
auto-discovered on next launch.
