# -*- coding: utf-8 -*-
"""
Inference engine: load model and perform predictions with calibration and risk stratification.
"""

import numpy as np
import pandas as pd
from feature_config import FEATURE_NAMES, TARGET_COLUMN
from model_manager import load_model_artifact
from risk_stratifier import assign_risk_level, assign_risk_levels_batch


class DiagnosisPredictor:
    """Loads a saved model artifact and provides single/batch prediction."""

    def __init__(self, model_dir='saved_model'):
        self.model_dir = model_dir
        self.pipeline = None
        self.calibrator = None
        self.metadata = None
        self.shap_background = None
        self._loaded = False

    def load(self):
        """Load model artifact from disk."""
        artifact, err = load_model_artifact(self.model_dir)
        if err:
            raise RuntimeError(err)
        self.pipeline = artifact['pipeline']
        self.calibrator = artifact['calibrator']
        self.metadata = artifact['metadata']
        self.shap_background = artifact['shap_background']
        self._loaded = True
        return self

    @property
    def is_loaded(self):
        return self._loaded

    def _calibrate(self, raw_probs):
        """Apply calibrator to raw probabilities."""
        calibrated = self.calibrator.transform(raw_probs.reshape(-1, 1)).flatten()
        return np.clip(calibrated, 0.001, 0.999)

    def _prepare_features(self, df):
        """Reorder columns to match training order, fill missing with NaN."""
        missing = set(FEATURE_NAMES) - set(df.columns)
        if missing:
            print(f"Warning: {len(missing)} missing features filled with NaN: {missing}")
            for f in missing:
                df[f] = np.nan
        df = df[FEATURE_NAMES]
        # Force float dtype — manual entry produces object-dtype columns when
        # some cells are None and others are float, which LightGBM rejects.
        # astype(float) converts None → NaN and is a no-op for already-numeric
        # columns from CSV.
        try:
            df = df.astype(float)
        except (TypeError, ValueError) as e:
            # Should not happen in normal flow; let the model raise a clearer
            # error if a non-numeric value slipped through.
            print(f"Warning: could not coerce all features to float: {e}")
        return df

    def predict_single(self, feature_dict):
        """Predict for a single patient. feature_dict: {feature_name: value}."""
        if not self._loaded:
            raise RuntimeError("Model not loaded. Call .load() first.")
        df = pd.DataFrame([feature_dict])
        df = self._prepare_features(df)
        raw_prob = self.pipeline.predict_proba(df)[0, 1]
        cal_prob = self._calibrate(np.array([raw_prob]))[0]
        risk = assign_risk_level(cal_prob)
        return {
            'raw_probability': float(raw_prob),
            'calibrated_probability': float(cal_prob),
            'risk_level': risk['level'],
            'risk_level_label': risk['label'],
            'risk_score': risk['score'],
            'risk_color': risk['color'],
        }

    def predict_batch(self, df):
        """Predict for a batch of patients. df: DataFrame with feature columns."""
        if not self._loaded:
            raise RuntimeError("Model not loaded. Call .load() first.")
        df = self._prepare_features(df.copy())
        raw_probs = self.pipeline.predict_proba(df)[:, 1]
        cal_probs = self._calibrate(raw_probs)
        risks = assign_risk_levels_batch(cal_probs)
        predictions = pd.DataFrame({
            'Raw_Probability': raw_probs,
            'Calibrated_Probability': cal_probs,
            'Risk_Level': [r['level'] for r in risks],
            'Risk_Level_Label': [r['label'] for r in risks],
            'Risk_Score': [r['score'] for r in risks],
        })
        return predictions, risks

    def predict_batch_no_impute(self, df):
        """
        Predict for a batch, bypassing the preprocessor (no imputation).
        Calls the underlying model directly so NaN cells are passed through
        to algorithms that handle them natively (e.g. LightGBM, XGBoost).
        Falls back to the full pipeline if the model cannot accept NaN.
        """
        if not self._loaded:
            raise RuntimeError("Model not loaded. Call .load() first.")
        df = self._prepare_features(df.copy())
        model = self.pipeline.named_steps.get('model')
        if model is None:
            # No 'model' step — fall back to full pipeline
            raw_probs = self.pipeline.predict_proba(df)[:, 1]
        else:
            try:
                raw_probs = model.predict_proba(df)[:, 1]
            except Exception as e:
                print(f"[no_impute] direct model call failed ({e}); "
                      f"falling back to full pipeline (with imputation).")
                raw_probs = self.pipeline.predict_proba(df)[:, 1]
        cal_probs = self._calibrate(raw_probs)
        risks = assign_risk_levels_batch(cal_probs)
        predictions = pd.DataFrame({
            'Raw_Probability': raw_probs,
            'Calibrated_Probability': cal_probs,
            'Risk_Level': [r['level'] for r in risks],
            'Risk_Level_Label': [r['label'] for r in risks],
            'Risk_Score': [r['score'] for r in risks],
        })
        return predictions, risks

    def get_model_info(self):
        """Return model metadata for display."""
        if not self._loaded:
            return {}
        return {
            'model_name': self.metadata.get('model_name', 'Unknown'),
            'cv_auc': self.metadata.get('cv_auc', 0),
            'n_features': self.metadata.get('n_features', 0),
            'training_samples': self.metadata.get('training_samples', 0),
            'calibration_method': self.metadata.get('calibration_method', 'auto'),
        }
