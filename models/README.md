# Locked model artifact

`locked_svm/` contains the frozen multimodal SVM used for all reported internal
and external evaluations in the paper.

| File | Contents |
|---|---|
| `best_model.joblib` | A scikit-learn `Pipeline` (median imputer + standard scaler + `SVC(C=1.0, kernel='rbf', gamma='scale', class_weight='balanced', probability=True, random_state=42)`). |
| `calibrator.joblib` | `IsotonicRegression` fit on out-of-fold training predictions; applied to `predict_proba` raw probabilities. |
| `shap_background.npy` | 50 k-means prototypes from the training set, used as the Kernel SHAP background. |
| `model_metadata.json` | `model_name`, `cv_auc` (0.9499), `feature_names` (121), `risk_thresholds {0.40, 0.80}`, `calibration_method`, `training_samples` (1264), `random_seed` (42). |

## Loading

```python
import joblib, json
pipeline = joblib.load("locked_svm/best_model.joblib")
calibrator = joblib.load("locked_svm/calibrator.joblib")
raw = pipeline.predict_proba(X)[:, 1]
cal = calibrator.transform(raw.reshape(-1, 1)).ravel()   # calibrated probability
```

## Compatibility

The artifact was created under **Python 3.10 / scikit-learn 1.3.2 / pandas
1.4.4 / joblib 1.4.2** (see `environment.yml`). Loading under a different
scikit-learn may emit a warning or fail; install the pinned versions to
reproduce exactly.

## Retraining

To regenerate this artifact from the full cohorts, run
`python analysis/01_train_multimodal.py` (requires the full source matrices).
The fresh artifact is written to `web_app/saved_models/<ts>_<model>/` and can
be copied here to replace the locked model.
