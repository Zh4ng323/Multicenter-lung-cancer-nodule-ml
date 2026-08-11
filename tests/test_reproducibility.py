# -*- coding: utf-8 -*-
"""
Smoke tests for the release.

Covers the reviewer-fast-path guarantees:
  * the locked artifact loads and its feature list matches feature_config;
  * `verify_performance` would PASS (AUCs above the subset thresholds);
  * the three test samples each land in the intended risk stratum via the
    web app's own DiagnosisPredictor.

Run with `make test` (or `python -m pytest -q tests/`).
"""

import sys
import json
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "analysis"))
sys.path.insert(0, str(REPO / "web_app"))

from config import (  # noqa: E402
    FEATURE_NAMES,
    LOCKED_SVM_DIR,
    SUBSET_INTERNAL,
    SUBSET_EXTERNAL,
    LOW_THRESHOLD,
    HIGH_THRESHOLD,
)
from feature_config import FEATURE_NAMES as WEB_FEATURE_NAMES  # noqa: E402


def test_feature_lists_match():
    """analysis/config and web_app/feature_config must agree on the 121 features."""
    assert FEATURE_NAMES == WEB_FEATURE_NAMES
    assert len(FEATURE_NAMES) == 121


def test_metadata_feature_names_match():
    with open(LOCKED_SVM_DIR / "model_metadata.json", encoding="utf-8") as f:
        meta = json.load(f)
    assert meta["feature_names"] == FEATURE_NAMES
    assert meta["n_features"] == 121
    assert meta["risk_thresholds"] == {"low": 0.40, "high": 0.80}


@pytest.mark.parametrize("path,n_min", [
    (SUBSET_INTERNAL, 290),
    (SUBSET_EXTERNAL, 140),
])
def test_subsets_present_and_shaped(path, n_min):
    import pandas as pd
    assert path.exists(), f"missing {path}"
    df = pd.read_csv(path)
    assert len(df) >= n_min
    assert list(df.columns[:-1]) == FEATURE_NAMES
    assert df.columns[-1] == "target"
    assert set(df["target"].unique()) <= {0, 1}


def test_verify_performance_passes():
    """Re-run the scoring checks inline and assert the ballpark thresholds."""
    import joblib
    import numpy as np
    import pandas as pd
    from sklearn.metrics import roc_auc_score

    pipeline = joblib.load(LOCKED_SVM_DIR / "best_model.joblib")
    calibrator = joblib.load(LOCKED_SVM_DIR / "calibrator.joblib")

    def cal(raw):
        c = calibrator.transform(raw.reshape(-1, 1)).ravel()
        return np.clip(c, 0.001, 0.999)

    df = pd.read_csv(SUBSET_INTERNAL)
    prob = cal(pipeline.predict_proba(df[FEATURE_NAMES].astype(float))[:, 1])
    assert roc_auc_score(df["target"], prob) >= 0.88

    df = pd.read_csv(SUBSET_EXTERNAL)
    prob = cal(pipeline.predict_proba(df[FEATURE_NAMES].astype(float))[:, 1])
    assert roc_auc_score(df["target"], prob) >= 0.90


@pytest.mark.parametrize("fname,level", [
    ("sample_low_risk.csv", "low-risk"),
    ("sample_medium_risk.csv", "medium-risk"),
    ("sample_high_risk.csv", "high-risk"),
])
def test_test_samples_stratify_correctly(fname, level):
    import pandas as pd
    from predictor import DiagnosisPredictor

    df = pd.read_csv(REPO / "test_samples" / fname)
    assert list(df.columns) == FEATURE_NAMES  # no target column, exact order
    assert len(df) == 1

    model_dir = str(REPO / "web_app" / "saved_models" / "20260804_2201_SVM")
    pred = DiagnosisPredictor(model_dir).load()
    preds, _ = pred.predict_batch(df)
    assert preds["Risk_Level"].iloc[0] == level
