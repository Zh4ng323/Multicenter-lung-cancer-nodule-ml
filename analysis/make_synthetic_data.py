# -*- coding: utf-8 -*-
"""
Generate the released synthetic subsets (~300 internal-test, ~150 external).

Rows are drawn from per-class multivariate Gaussian distributions whose means
and Ledoit-Wolf shrunk covariances are estimated from the real cohorts in the
standardized feature space, then transformed back to the raw feature scale.
No real participant rows are included. Scored with the locked SVM, the
synthetic subsets reproduce the paper's ballpark performance (internal AUC
~0.93, external ~0.96).

Reads the full source matrices, which are not shipped; external users receive
the synthetic CSVs directly.
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.covariance import LedoitWolf
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score

sys.path.insert(0, str(Path(__file__).resolve().parent))
from config import (  # noqa: E402
    FEATURE_NAMES,
    TARGET_COLUMN,
    RANDOM_SEED,
    LOW_THRESHOLD,
    HIGH_THRESHOLD,
    FULL_DEV_CSV,
    FULL_EXTERNAL_CSV,
    LOCKED_SVM_DIR,
    SUBSET_INTERNAL,
    SUBSET_EXTERNAL,
)

N_INTERNAL = (154, 146)  # (benign, cancer) — matches paper class proportions
N_EXTERNAL = (85, 65)
ALPHA = 1.0  # 1.0 = pure per-class Gaussian statistics; no separation boost


def _sample(df, n0, n1, seed):
    """Fit per-class Gaussians in standardized space and sample n0+n1 rows."""
    sc = StandardScaler().fit(df[FEATURE_NAMES])
    Z = sc.transform(df[FEATURE_NAMES])
    y = df[TARGET_COLUMN].values
    params = {}
    for c in (0, 1):
        Zc = Z[y == c]
        params[c] = (Zc.mean(axis=0), LedoitWolf().fit(Zc).covariance_)
    pooled = Z.mean(axis=0)

    rng = np.random.RandomState(seed)
    rows, ys = [], []
    for c, n in ((0, n0), (1, n1)):
        mu, cov = params[c]
        shift = ALPHA * (mu - pooled)
        Zs = rng.multivariate_normal(pooled + shift, cov, size=n)
        rows.append(sc.inverse_transform(Zs))
        ys.extend([c] * n)
    out = pd.DataFrame(np.vstack(rows), columns=FEATURE_NAMES)
    out[TARGET_COLUMN] = ys
    return out


def _report(label, df):
    import joblib
    pipeline = joblib.load(LOCKED_SVM_DIR / "best_model.joblib")
    p = pipeline.predict_proba(df[FEATURE_NAMES].astype(float))[:, 1]
    auc = roc_auc_score(df[TARGET_COLUMN], p)
    masks = [p <= LOW_THRESHOLD,
             (p > LOW_THRESHOLD) & (p <= HIGH_THRESHOLD),
             p > HIGH_THRESHOLD]
    props = [df.loc[m, TARGET_COLUMN].mean() if m.sum() else float("nan") for m in masks]
    print(f"{label}: AUC {auc:.4f}; stratum LC proportions "
          f"{props[0]:.3f} / {props[1]:.3f} / {props[2]:.3f}")


def main():
    if not FULL_DEV_CSV.exists():
        sys.exit(f"Full dev matrix not found: {FULL_DEV_CSV}\n"
                 "Author-side only; the synthetic subsets are already in data/.")
    dev = pd.read_csv(FULL_DEV_CSV)
    ext = pd.read_csv(FULL_EXTERNAL_CSV)

    # Isolate the held-out internal-test partition (seed-42 80:20 stratified split).
    _, X_test, _, y_test = train_test_split(
        dev[FEATURE_NAMES], dev[TARGET_COLUMN],
        test_size=0.2, random_state=RANDOM_SEED, stratify=dev[TARGET_COLUMN])
    internal_partition = pd.concat([X_test, y_test], axis=1)

    internal = _sample(internal_partition, *N_INTERNAL, seed=RANDOM_SEED)
    external = _sample(ext, *N_EXTERNAL, seed=RANDOM_SEED)

    internal = internal[FEATURE_NAMES + [TARGET_COLUMN]]
    external = external[FEATURE_NAMES + [TARGET_COLUMN]]
    internal.to_csv(SUBSET_INTERNAL, index=False)
    external.to_csv(SUBSET_EXTERNAL, index=False)

    print(f"Wrote {SUBSET_INTERNAL}  ({len(internal)} synthetic rows, "
          f"target {internal[TARGET_COLUMN].value_counts().to_dict()})")
    print(f"Wrote {SUBSET_EXTERNAL}  ({len(external)} synthetic rows, "
          f"target {external[TARGET_COLUMN].value_counts().to_dict()})")
    _report("Internal (synthetic)", internal)
    _report("External (synthetic)", external)


if __name__ == "__main__":
    main()
