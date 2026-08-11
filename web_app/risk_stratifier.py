# -*- coding: utf-8 -*-
"""
Risk stratification: assign risk levels based on calibrated probability.
Thresholds: Low-risk <= 0.40, Medium-risk 0.40-0.80, High-risk > 0.80
"""

from feature_config import RISK_THRESHOLDS

RISK_COLORS = {
    'low-risk': '#2ecc71',
    'medium-risk': '#f39c12',
    'high-risk': '#e74c3c',
}

RISK_LABELS = {
    'low-risk': 'Low Risk',
    'medium-risk': 'Medium Risk',
    'high-risk': 'High Risk',
}


def assign_risk_level(calibrated_prob, thresholds=None):
    """Assign risk level for a single calibrated probability."""
    if thresholds is None:
        thresholds = RISK_THRESHOLDS
    low_t = thresholds['low']
    high_t = thresholds['high']

    if calibrated_prob <= low_t:
        level = 'low-risk'
        score = 1
    elif calibrated_prob <= high_t:
        level = 'medium-risk'
        score = 2
    else:
        level = 'high-risk'
        score = 3

    return {
        'level': level,
        'label': RISK_LABELS[level],
        'score': score,
        'color': RISK_COLORS[level],
    }


def assign_risk_levels_batch(probs, thresholds=None):
    """Vectorized risk assignment for an array of probabilities."""
    return [assign_risk_level(p, thresholds) for p in probs]


def get_risk_html(result):
    """Generate styled HTML for risk display — larger font for screenshots."""
    color = result['color']
    level_label = result['label']
    return f"""
    <div style="
        background-color: {color}; color: white; padding: 10px 20px;
        border-radius: 6px; text-align: center; font-size: 26px;
        font-weight: 700; margin: 2px 0; letter-spacing: 0.5px;
        box-shadow: 0 1px 4px rgba(0,0,0,0.12);
    ">
        Risk Level: {level_label}
    </div>
    """
