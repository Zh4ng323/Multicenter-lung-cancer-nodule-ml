# -*- coding: utf-8 -*-
"""
Central configuration: repo-relative paths, seeds, locked-SVM hyperparameters,
risk thresholds. Feature names are imported from web_app/feature_config.py.
"""

from pathlib import Path
import sys

# ---------------------------------------------------------------------------
# Repo layout
# ---------------------------------------------------------------------------
# analysis/ is one level below the repo root.
REPO_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = REPO_ROOT / "data"
MODELS_DIR = REPO_ROOT / "models"
WEBAPP_DIR = REPO_ROOT / "web_app"
RESULTS_DIR = REPO_ROOT / "results"  # gitignored — training outputs land here

# Locked model artifact (shipped with the repo).
LOCKED_SVM_DIR = MODELS_DIR / "locked_svm"

# Released synthetic subsets (shipped with the repo).
SUBSET_INTERNAL = DATA_DIR / "internal_test_subset.csv"
SUBSET_EXTERNAL = DATA_DIR / "external_subset.csv"

# Full source matrices (NOT shipped — they live one level above the release
# folder in the author's working directory). Used only by make_synthetic_data.py
# and 01_train_multimodal.py for full-cohort reproduction; external users will
# not have these and should rely on the released subsets + locked model.
SOURCE_ROOT = REPO_ROOT.parent
FULL_DEV_CSV = SOURCE_ROOT / "去除重复人员异常值处理后-Lasso后舌脉面.csv"
FULL_EXTERNAL_CSV = SOURCE_ROOT / "两组舌面脉 - 1异常值处理后 - lasso后指标-外部验证.csv"

# ---------------------------------------------------------------------------
# Single source of truth: import from the web app's feature_config
# ---------------------------------------------------------------------------
# Make web_app importable when running analysis scripts from anywhere.
if str(WEBAPP_DIR) not in sys.path:
    sys.path.insert(0, str(WEBAPP_DIR))

from feature_config import (  # noqa: E402  (import after sys.path tweak)
    FEATURE_NAMES,
    TARGET_COLUMN,
    RANDOM_SEED,
    RISK_THRESHOLDS,
    N_FEATURES,
)

# ---------------------------------------------------------------------------
# Reproducibility constants
# ---------------------------------------------------------------------------
N_FOLDS = 10
TEST_SIZE = 0.2  # 80:20 train / held-out internal test
N_BOOTSTRAP = 1000  # stratified percentile bootstrap for CIs

# Locked multimodal SVM (paper Table 3). Grid search selected these and they
# were then frozen for internal + external evaluation.
LOCKED_SVM_PARAMS = dict(
    C=1.0,
    kernel="rbf",
    gamma="scale",
    class_weight="balanced",
    probability=True,
    random_state=RANDOM_SEED,
)

# Risk-stratum operating points (exploratory, fixed).
LOW_THRESHOLD = float(RISK_THRESHOLDS["low"])    # 0.40
HIGH_THRESHOLD = float(RISK_THRESHOLDS["high"])  # 0.80

# Published full-cohort AUCs (paper Table 3) for reference / verification.
PAPER_AUC = {
    "cv_oof": 0.951,      # pooled out-of-fold, DeLong CI 0.939-0.962
    "cv_oof_ci": (0.939, 0.962),
    "internal": 0.926,    # held-out internal test, DeLong CI 0.898-0.954
    "internal_ci": (0.898, 0.954),
    "external": 0.954,    # external cohort, DeLong CI 0.942-0.967
    "external_ci": (0.942, 0.967),
}


def locked_model_paths():
    """Return the 4 artifact paths for the locked SVM as a dict."""
    return {
        "pipeline": LOCKED_SVM_DIR / "best_model.joblib",
        "calibrator": LOCKED_SVM_DIR / "calibrator.joblib",
        "shap_background": LOCKED_SVM_DIR / "shap_background.npy",
        "metadata": LOCKED_SVM_DIR / "model_metadata.json",
    }
