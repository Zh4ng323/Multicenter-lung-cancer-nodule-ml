# -*- coding: utf-8 -*-
"""
Verify the locked SVM on the released synthetic subsets against the paper's
reported performance (ballpark check).
"""

import json
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import (
    roc_auc_score,
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    average_precision_score,
)

sys.path.insert(0, str(Path(__file__).resolve().parent))
from config import (  # noqa: E402
    FEATURE_NAMES,
    TARGET_COLUMN,
    LOCKED_SVM_DIR,
    SUBSET_INTERNAL,
    SUBSET_EXTERNAL,
    LOW_THRESHOLD,
    HIGH_THRESHOLD,
    PAPER_AUC,
)

# Ballpark bounds (subset-tolerant). Published full-cohort CIs are tighter.
INTERNAL_AUC_MIN = 0.88
EXTERNAL_AUC_MIN = 0.90


def load_locked_model():
    paths = {
        "pipeline": LOCKED_SVM_DIR / "best_model.joblib",
        "calibrator": LOCKED_SVM_DIR / "calibrator.joblib",
        "metadata": LOCKED_SVM_DIR / "model_metadata.json",
    }
    missing = [p for p in paths.values() if not p.exists()]
    if missing:
        sys.exit("Missing locked-model artifact files:\n  " +
                 "\n  ".join(str(p) for p in missing))
    pipeline = joblib.load(paths["pipeline"])
    calibrator = joblib.load(paths["calibrator"])
    with open(paths["metadata"], encoding="utf-8") as f:
        metadata = json.load(f)
    # Sanity: the model's feature order must match the canonical list.
    if metadata.get("feature_names") != FEATURE_NAMES:
        sys.exit("Feature-name mismatch between model_metadata.json and "
                 "feature_config.FEATURE_NAMES.")
    return pipeline, calibrator, metadata


def calibrate(calibrator, raw):
    """Mirror web_app/predictor.py: isotonic transform + clip to [0.001, 0.999]."""
    cal = calibrator.transform(raw.reshape(-1, 1)).ravel()
    return np.clip(cal, 0.001, 0.999)


def score(pipeline, calibrator, df):
    X = df[FEATURE_NAMES].astype(float)
    raw = pipeline.predict_proba(X)[:, 1]
    return calibrate(calibrator, raw)


def metric_block(y_true, prob):
    pred = (prob >= 0.5).astype(int)
    return {
        "AUC": roc_auc_score(y_true, prob),
        "Accuracy": accuracy_score(y_true, pred),
        "Precision": precision_score(y_true, pred, zero_division=0),
        "Sensitivity": recall_score(y_true, pred, zero_division=0),
        "Specificity": recall_score(1 - y_true, 1 - pred, zero_division=0),
        "F1": f1_score(y_true, pred, zero_division=0),
        "AP": average_precision_score(y_true, prob),
    }


def strata_lc_proportion(y_true, prob):
    """Observed LC proportion in each exploratory score stratum."""
    groups = {
        "low (<= %.2f)" % LOW_THRESHOLD: prob <= LOW_THRESHOLD,
        "medium": (prob > LOW_THRESHOLD) & (prob <= HIGH_THRESHOLD),
        "high (> %.2f)" % HIGH_THRESHOLD: prob > HIGH_THRESHOLD,
    }
    out = {}
    for name, mask in groups.items():
        n = int(mask.sum())
        lc = int(((y_true == 1) & mask).sum())
        out[name] = (n, lc, (lc / n) if n else float("nan"))
    return out


def main():
    print("=" * 64)
    print("Locked-SVM performance verification (released subsets)")
    print("=" * 64)

    pipeline, calibrator, meta = load_locked_model()
    print(f"Model: {meta.get('model_name')}  cv_auc={meta.get('cv_auc'):.4f}  "
          f"calibration={meta.get('calibration_method')}  "
          f"n_features={meta.get('n_features')}")
    print(f"Risk thresholds: low <= {LOW_THRESHOLD}, high > {HIGH_THRESHOLD}")

    results = {}
    for label, path, auc_min, paper in [
        ("Internal-test subset", SUBSET_INTERNAL, INTERNAL_AUC_MIN, "internal"),
        ("External subset", SUBSET_EXTERNAL, EXTERNAL_AUC_MIN, "external"),
    ]:
        df = pd.read_csv(path)
        y = df[TARGET_COLUMN].astype(int).values
        prob = score(pipeline, calibrator, df)
        m = metric_block(y, prob)
        results[label] = (m, y, prob, auc_min, paper)

        print(f"\n--- {label} (n={len(df)}, LC={int(y.sum())}, "
              f"BPN={int((y == 0).sum())}) ---")
        paper_auc = PAPER_AUC[paper]
        paper_ci = PAPER_AUC[paper + "_ci"]
        ok = m["AUC"] >= auc_min
        flag = "PASS" if ok else "FAIL"
        print(f"  AUC          {m['AUC']:.4f}   "
              f"(paper full-cohort: {paper_auc:.3f}, "
              f"95% CI {paper_ci[0]:.3f}-{paper_ci[1]:.3f}; subset >= {auc_min})  {flag}")
        for k in ("Accuracy", "Precision", "Sensitivity", "Specificity",
                  "F1", "AP"):
            print(f"  {k:<12} {m[k]:.4f}")

        strata = strata_lc_proportion(y, prob)
        print("  Exploratory score strata (observed LC proportion):")
        for name, (n, lc, p) in strata.items():
            print(f"    {name:<22} n={n:<4} LC={lc:<4} prop={p:.3f}")

    # --- Assertions ----------------------------------------------------------
    print("\n" + "=" * 64)
    print("Checks")
    print("=" * 64)
    all_ok = True

    for label, (m, y, prob, auc_min, paper) in results.items():
        ok = m["AUC"] >= auc_min
        all_ok &= ok
        print(f"  [{'OK' if ok else 'XX'}] {label} AUC {m['AUC']:.4f} >= {auc_min}")

    # Monotone LC proportion low < medium < high on both subsets.
    for label, (m, y, prob, auc_min, paper) in results.items():
        strata = strata_lc_proportion(y, prob)
        props = [strata[k][2] for k in strata]
        # allow the medium stratum to be empty (n=0) on small subsets;
        # if present, require monotonic increase.
        valid = [p for p in props if not np.isnan(p)]
        mono = all(valid[i] <= valid[i + 1] + 1e-9 for i in range(len(valid) - 1))
        # If a stratum is empty we cannot assert direction strictly; treat as OK.
        ok = mono or len(valid) < 2
        all_ok &= ok
        print(f"  [{'OK' if ok else 'XX'}] {label} monotone LC proportion across strata")

    print("=" * 64)
    if all_ok:
        print("RESULT: PASS — locked model shows ballpark discrimination "
              "consistent with the paper on the released subsets.")
    else:
        print("RESULT: FAIL — one or more checks did not pass.")
        sys.exit(1)


if __name__ == "__main__":
    main()
