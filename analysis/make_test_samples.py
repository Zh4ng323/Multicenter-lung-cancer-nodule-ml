# -*- coding: utf-8 -*-
"""
Generate three single-row demonstration samples for the web app, one per risk
stratum (low / medium / high), from the released external subset.
"""

import json
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from config import (  # noqa: E402
    FEATURE_NAMES,
    TARGET_COLUMN,
    LOCKED_SVM_DIR,
    SUBSET_EXTERNAL,
    LOW_THRESHOLD,
    HIGH_THRESHOLD,
)

OUT_DIR = Path(__file__).resolve().parents[1] / "test_samples"

# Trend features from the paper (Pulse §, LASSO surrogate coefs in §"Modality-
# level contribution"). Positive list pushes toward lung cancer; negative list
# pushes toward benign nodules.
TREND_POSITIVE = ["h1/t1", "color-G-3", "wholecolor-Cr", "h4", "h5"]  # +h1,h3 not in 121
TREND_NEGATIVE = ["Per-all", "color-Cb-1", "GLCM-asm_3", "h4/h1", "w1/t", "t1/t"]

# Paper pulse findings (line 78): LC higher h1,h3,h5,h1/t1; lower t,t1,h4/h1,w1/t.
# h1/h3/t/t1 are NOT individually in the 121 set, but the ratios that capture
# them (h1/t1, h4/h1, w1/t, t1/t) are — covered above.

SAMPLE_FILES = {
    "low": OUT_DIR / "sample_low_risk.csv",
    "medium": OUT_DIR / "sample_medium_risk.csv",
    "high": OUT_DIR / "sample_high_risk.csv",
}


def load_locked():
    pipeline = joblib.load(LOCKED_SVM_DIR / "best_model.joblib")
    calibrator = joblib.load(LOCKED_SVM_DIR / "calibrator.joblib")
    return pipeline, calibrator


def calibrate(calibrator, raw):
    cal = calibrator.transform(raw.reshape(-1, 1)).ravel()
    return np.clip(cal, 0.001, 0.999)


def trend_score(df):
    """Positive => lung-cancer-like trend alignment. Z-scored against the cohort."""
    zpos = sum((df[f] - df[f].mean()) / (df[f].std(ddof=0) + 1e-9)
               for f in TREND_POSITIVE if f in df.columns)
    zneg = sum((df[f] - df[f].mean()) / (df[f].std(ddof=0) + 1e-9)
               for f in TREND_NEGATIVE if f in df.columns)
    return zpos - zneg


def pick_row(candidates, prefer_label, by, ascending):
    """Pick a row, preferring the requested true label, then by `by`."""
    if prefer_label is not None and (candidates[TARGET_COLUMN] == prefer_label).any():
        c = candidates[candidates[TARGET_COLUMN] == prefer_label]
    else:
        c = candidates
    c = c.sort_values(by=by, ascending=ascending)
    return c.iloc[0]


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    pipeline, calibrator = load_locked()

    df = pd.read_csv(SUBSET_EXTERNAL)
    X = df[FEATURE_NAMES].astype(float)
    raw = pipeline.predict_proba(X)[:, 1]
    df["_proba"] = calibrate(calibrator, raw)
    df["_trend"] = trend_score(X)

    # Candidate pools by stratum.
    low_pool = df[df["_proba"] <= LOW_THRESHOLD]
    mid_pool = df[(df["_proba"] > LOW_THRESHOLD) & (df["_proba"] <= HIGH_THRESHOLD)]
    high_pool = df[df["_proba"] > HIGH_THRESHOLD]

    print(f"Stratum pool sizes  low={len(low_pool)} "
          f"medium={len(mid_pool)} high={len(high_pool)}")

    chosen = {}

    # Low-risk: prefer a benign-nodule row with the most negative trend score.
    chosen["low"] = pick_row(low_pool, prefer_label=0, by="_trend", ascending=True)
    # High-risk: prefer a lung-cancer row with the most positive trend score.
    chosen["high"] = pick_row(high_pool, prefer_label=1, by="_trend", ascending=False)
    # Medium: a row near the score centre (proba closest to midpoint).
    mid = 0.5 * (LOW_THRESHOLD + HIGH_THRESHOLD)
    mid_pool = mid_pool.assign(_dist=(mid_pool["_proba"] - mid).abs())
    chosen["medium"] = pick_row(
        mid_pool.sort_values("_dist"), prefer_label=None, by="_dist", ascending=True
    )

    readme_lines = [
        "# Web-app demonstration test samples",
        "",
        "Three single-row CSVs (121 features, no `target` column) ready to upload",
        "into the research web application. Each lands in a different risk stratum and",
        "showcases the distribution trends the paper describes for that group.",
        "",
        "Generated reproducibly by `analysis/make_test_samples.py` from",
        "`data/external_subset.csv` using the locked SVM.",
        "",
        "Risk strata use fixed exploratory cut points: low <= 0.40,",
        "medium 0.40-0.80, high > 0.80 (calibrated probability).",
        "",
        "| Sample | Calibrated proba | Stratum | True label |",
        "|---|---|---|---|",
    ]

    level_names = {
        "low": "low-risk",
        "medium": "medium-risk",
        "high": "high-risk",
    }
    label_names = {0: "Benign pulmonary nodule", 1: "Lung cancer"}

    for key in ("low", "medium", "high"):
        row = chosen[key]
        proba = float(row["_proba"])
        true_label = int(row[TARGET_COLUMN])
        # Sanity check: confirm the locked model assigns the expected level.
        expected = level_names[key]
        if key == "low":
            assert proba <= LOW_THRESHOLD, f"low sample proba {proba} > {LOW_THRESHOLD}"
        elif key == "high":
            assert proba > HIGH_THRESHOLD, f"high sample proba {proba} <= {HIGH_THRESHOLD}"
        else:
            assert LOW_THRESHOLD < proba <= HIGH_THRESHOLD

        out_df = pd.DataFrame([row[FEATURE_NAMES].values], columns=FEATURE_NAMES)
        out_df.to_csv(SAMPLE_FILES[key], index=False)

        print(f"\n[{key}] proba={proba:.4f}  level={expected}  "
              f"true={label_names[true_label]}  -> {SAMPLE_FILES[key].name}")

        # Build a short human-readable trend summary for the README.
        pos_shown = {f: float(row[f]) for f in TREND_POSITIVE if f in FEATURE_NAMES}
        neg_shown = {f: float(row[f]) for f in TREND_NEGATIVE if f in FEATURE_NAMES}
        readme_lines.append(
            f"| `{SAMPLE_FILES[key].name}` | {proba:.4f} | **{expected}** "
            f"| {label_names[true_label]} |"
        )

    # Detailed trend appendix.
    readme_lines += [
        "",
        "## Showcased trend features",
        "",
        "Paper direction (lung cancer vs benign nodules):",
        "- Higher in lung cancer: `h1/t1`, `h4`, `h5`, `color-G-3`, `wholecolor-Cr`",
        "- Lower in lung cancer: `Per-all`, `color-Cb-1`, `GLCM-asm_3`, "
        "`h4/h1`, `w1/t`, `t1/t`",
        "",
        "Per-sample values (high-risk should look lung-cancer-like, low-risk the",
        "reverse, medium in between):",
        "",
    ]
    for key in ("low", "medium", "high"):
        row = chosen[key]
        readme_lines.append(f"### `{SAMPLE_FILES[key].name}` ({level_names[key]})")
        for f in TREND_POSITIVE + TREND_NEGATIVE:
            if f in FEATURE_NAMES:
                readme_lines.append(f"- `{f}` = {float(row[f]):.4f}")
        readme_lines.append("")

    readme_lines += [
        "## Usage",
        "",
        "Upload any of these CSVs in the web app's CSV-import panel (deployed",
        "site and demo account are in the top-level README).",
        "",
        "These samples are synthetic and are for software demonstration, not",
        "clinical use.",
    ]

    (OUT_DIR / "README.md").write_text(
        "\n".join(readme_lines), encoding="utf-8"
    )
    print(f"\nWrote {OUT_DIR / 'README.md'}")


if __name__ == "__main__":
    main()
