# -*- coding: utf-8 -*-
"""
Model persistence: save/load trained model artifacts.
"""

import os
import json
import numpy as np
import joblib
from feature_config import FEATURE_NAMES, N_FEATURES, RISK_THRESHOLDS


def save_model_artifact(pipeline, calibrator, shap_background, model_name,
                        cv_auc, output_dir='saved_model', training_samples=None,
                        calibration_method='auto', feature_importance=None):
    """Save the complete model artifact to disk."""
    os.makedirs(output_dir, exist_ok=True)

    # Save pipeline (preprocessor + model)
    joblib.dump(pipeline, os.path.join(output_dir, 'best_model.joblib'))

    # Save calibrator
    joblib.dump(calibrator, os.path.join(output_dir, 'calibrator.joblib'))

    # Save SHAP background data
    np.save(os.path.join(output_dir, 'shap_background.npy'), shap_background)

    # Save metadata
    metadata = {
        'model_name': model_name,
        'cv_auc': float(cv_auc),
        'feature_names': FEATURE_NAMES,
        'n_features': N_FEATURES,
        'risk_thresholds': RISK_THRESHOLDS,
        'calibration_method': calibration_method,
        'training_samples': training_samples,
        'random_seed': 42,
        'feature_importance': feature_importance,
    }
    with open(os.path.join(output_dir, 'model_metadata.json'), 'w', encoding='utf-8') as f:
        json.dump(metadata, f, ensure_ascii=False, indent=2)

    print(f"Model artifact saved to: {os.path.abspath(output_dir)}")
    return output_dir


def load_model_artifact(model_dir='saved_model'):
    """Load model artifact from disk. Returns dict with pipeline, calibrator, metadata."""
    if not os.path.isdir(model_dir):
        return None, f"Model directory not found: {model_dir}"

    model_path = os.path.join(model_dir, 'best_model.joblib')
    calib_path = os.path.join(model_dir, 'calibrator.joblib')
    meta_path = os.path.join(model_dir, 'model_metadata.json')

    for p in (model_path, calib_path, meta_path):
        if not os.path.exists(p):
            return None, f"Missing file: {p}"

    pipeline = joblib.load(model_path)
    calibrator = joblib.load(calib_path)

    with open(meta_path, 'r', encoding='utf-8') as f:
        metadata = json.load(f)

    # Validate feature consistency
    saved_features = metadata.get('feature_names', [])
    if len(saved_features) != N_FEATURES:
        return None, f"Feature count mismatch: saved={len(saved_features)}, expected={N_FEATURES}"
    if saved_features != FEATURE_NAMES:
        return None, "Feature names mismatch between saved model and current config"

    # Load SHAP background if exists
    shap_bg_path = os.path.join(model_dir, 'shap_background.npy')
    shap_background = np.load(shap_bg_path) if os.path.exists(shap_bg_path) else None

    return {
        'pipeline': pipeline,
        'calibrator': calibrator,
        'metadata': metadata,
        'shap_background': shap_background,
    }, None
