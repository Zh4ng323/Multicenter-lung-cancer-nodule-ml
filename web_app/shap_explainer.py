# -*- coding: utf-8 -*-
"""
Single-patient SHAP explainability: compact native-style waterfall,
feature importance bar chart, and decision explanation.
"""

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import shap
from feature_config import FEATURE_NAMES, FEATURE_GROUPS, GROUP_DISPLAY_NAMES

import xgboost as xgb
import lightgbm as lgb
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier

TREE_MODELS = (xgb.XGBClassifier, lgb.LGBMClassifier,
               RandomForestClassifier, GradientBoostingClassifier)

# SHAP palette (same as shap.plots.colors)
_RED = '#ff0051'
_BLUE = '#008bfb'
_GRAY = '#bbbbbb'
_BASE_LINE = '#666666'


def _get_feature_group(feature_name):
    """Return the group name for a feature."""
    for group_key, features in FEATURE_GROUPS.items():
        if feature_name in features:
            return GROUP_DISPLAY_NAMES.get(group_key, group_key)
    return "Other"


def explain_single_patient(pipeline, feature_values, shap_background=None,
                           skip_preprocess=False):
    """Compute SHAP values for a single patient.

    If skip_preprocess=True, the preprocessor (imputer/scaler) is bypassed
    and the raw NaN-containing DataFrame is fed directly to the explainer.
    Use this for tree models (LightGBM/XGBoost) that handle NaN natively.
    """
    model = pipeline.named_steps.get('model')
    preprocessor = pipeline.named_steps.get('preprocessor')

    if isinstance(feature_values, dict):
        df = pd.DataFrame([feature_values])
        for f in FEATURE_NAMES:
            if f not in df.columns:
                df[f] = np.nan
        df = df[FEATURE_NAMES]
    else:
        df = pd.DataFrame([feature_values], columns=FEATURE_NAMES)

    if skip_preprocess:
        X_processed = df.values
    elif preprocessor is not None:
        X_processed = preprocessor.transform(df)
    else:
        X_processed = df.values
    X_raw = df.values[0]

    bg = shap_background if shap_background is not None else X_processed

    is_tree = isinstance(model, TREE_MODELS)
    if is_tree:
        explainer = shap.TreeExplainer(model)
    else:
        explainer = shap.KernelExplainer(model.predict_proba, bg)

    if is_tree:
        shap_vals_raw = explainer.shap_values(X_processed)
    else:
        shap_vals_raw = explainer.shap_values(X_processed, silent=True)

    if isinstance(shap_vals_raw, list) and len(shap_vals_raw) == 2:
        shap_vals = shap_vals_raw[1][0]
    elif isinstance(shap_vals_raw, list):
        shap_vals = shap_vals_raw[0][0]
    else:
        shap_vals = shap_vals_raw[0]

    exp_val = explainer.expected_value
    if isinstance(exp_val, (list, np.ndarray)):
        exp_val = exp_val[1] if len(exp_val) > 1 else exp_val[0]
    exp_val = float(exp_val)

    return {
        'shap_values': shap_vals,
        'expected_value': exp_val,
        'feature_names': FEATURE_NAMES,
        'processed_data': X_processed[0],
        'raw_data': X_raw,
    }


def _fmt_num(v, prec=3):
    try:
        fv = float(v)
        if abs(fv) >= 100 or (abs(fv) > 0 and abs(fv) < 0.01):
            return f"{fv:.3g}"
        return f"{fv:.{prec}g}"
    except Exception:
        return str(v)


def generate_waterfall_plot(explanation, patient_label="Patient", top_n=8,
                            row_height=0.34, entered_features=None):
    """
    Compact native-style SHAP waterfall with controllable row height.

    Y-axis shows feature names only (no raw values) to save width;
    no figure title (page Image label already provides it).
    Native label placement: value text INSIDE the bar when it fits,
    OUTSIDE (on the bar's tip side) when the bar is too narrow.

    If entered_features (set/list of feature names actually entered by the
    user) is provided, manually-entered features are listed individually at
    the top and all imputed features are folded into a single gray row.
    """
    shap_vals = np.asarray(explanation['shap_values'], dtype=float).ravel()
    names = list(explanation['feature_names'])
    base = float(explanation['expected_value'])
    final = base + float(shap_vals.sum())

    if entered_features is not None:
        entered_set = set(entered_features)
        entered_mask = np.array([n in entered_set for n in names])
        entered_idx = np.where(entered_mask)[0]
        imputed_idx = np.where(~entered_mask)[0]

        # All entered features shown individually (sorted by |SHAP|)
        ent_order = entered_idx[np.argsort(-np.abs(shap_vals[entered_idx]))]
        rows = []
        for idx in ent_order:
            rows.append({
                'label': names[idx],
                'value': float(shap_vals[idx]),
                'name': names[idx],
                'imputed': False,
            })
        if len(imputed_idx) > 0:
            rows.append({
                'label': f"{len(imputed_idx)} imputed (training mean)",
                'value': float(shap_vals[imputed_idx].sum()),
                'name': 'imputed',
                'imputed': True,
            })
    else:
        order = np.argsort(-np.abs(shap_vals))
        n_show = min(top_n, len(shap_vals))
        has_other = len(shap_vals) > n_show
        n_indiv = n_show - 1 if has_other else n_show
        rows = []
        for i in range(n_indiv):
            idx = order[i]
            rows.append({
                'label': names[idx],
                'value': float(shap_vals[idx]),
                'name': names[idx],
                'imputed': False,
            })
        if has_other:
            rest = order[n_indiv:]
            rest_sum = float(shap_vals[rest].sum())
            rows.append({
                'label': f"{len(rest)} other features",
                'value': rest_sum,
                'name': 'other',
                'imputed': False,
            })

    n_rows = len(rows)

    # width further reduced so the panel fits its column
    fig_w = 4.0
    fig_h = max(2.0, n_rows * row_height + 0.78)
    fig, ax = plt.subplots(figsize=(fig_w, fig_h), dpi=120)

    ys = list(range(n_rows - 1, -1, -1))

    loc = final
    starts = []
    ends = []
    for r in rows:
        sval = r['value']
        end = loc
        start = loc - sval
        starts.append(start)
        ends.append(end)
        loc = start

    all_x = starts + ends + [base, final]
    xmin, xmax = min(all_x), max(all_x)
    span = max(xmax - xmin, 1e-6)
    pad = 0.14 * span
    ax.set_xlim(xmin - pad, xmax + pad)

    bar_h = 0.62
    fs_bar = 9.5     # SHAP value text (smaller)
    fs_y = 11.5      # y-axis feature names (was 12.5; -1)
    fs_x = 10.5      # x tick / xlabel
    fs_annot = 10.0  # E[f(X)] / f(x)

    for i, r in enumerate(rows):
        y = ys[i]
        sval = r['value']
        x0, x1 = starts[i], ends[i]
        left = min(x0, x1)
        width = abs(sval)
        if r.get('imputed'):
            color = _GRAY
        else:
            color = _RED if sval >= 0 else _BLUE

        ax.barh(y, width, left=left, height=bar_h, color=color,
                edgecolor='white', linewidth=0.4, zorder=3, alpha=0.95)

        # Native SHAP behavior: put value INSIDE if the bar is wide enough,
        # otherwise OUTSIDE on the side the bar points to.
        txt = f"{sval:+.3f}"
        if width > 0.22 * span:
            ax.text(left + width / 2, y, txt, ha='center', va='center',
                    color='white', fontsize=fs_bar, fontweight='bold', zorder=4)
        else:
            if sval >= 0:
                ax.text(x1 + 0.02 * span, y, txt, ha='left', va='center',
                        color=color, fontsize=fs_bar, fontweight='bold', zorder=4)
            else:
                ax.text(x1 - 0.02 * span, y, txt, ha='right', va='center',
                        color=color, fontsize=fs_bar, fontweight='bold', zorder=4)

        if i < n_rows - 1:
            ax.plot([x1, x1], [y - bar_h / 2 - 0.02, ys[i + 1] + bar_h / 2 + 0.02],
                    color=_GRAY, ls='--', lw=0.7, zorder=2)

    ax.axvline(base, color=_BASE_LINE, lw=0.9, ls='--', zorder=1, alpha=0.85)
    ax.axvline(final, color='#222222', lw=1.0, zorder=1, alpha=0.9)

    ax.set_yticks(ys)
    ax.set_yticklabels([r['label'] for r in rows], fontsize=fs_y)
    ax.invert_yaxis()
    ax.set_xlabel('SHAP value (impact on model output)', fontsize=fs_x)
    ax.tick_params(axis='x', labelsize=fs_x - 0.5)
    ax.grid(True, axis='x', alpha=0.22, zorder=0)
    for spine in ('top', 'right'):
        ax.spines[spine].set_visible(False)

    ymin, ymax = ax.get_ylim()
    y_annot = ymax - 0.36
    ax.text(base, y_annot, f"E[f(X)] = {base:.3f}", ha='center', va='bottom',
            fontsize=fs_annot, color='#555555', clip_on=False)
    ax.text(final, y_annot, f"f(x) = {final:.3f}", ha='center', va='bottom',
            fontsize=fs_annot, color='#111111', fontweight='bold', clip_on=False)

    ax.set_ylim(ymin + 0.14, ymax - 0.46)
    fig.subplots_adjust(left=0.30, right=0.97, top=0.90, bottom=0.18)
    return fig


def generate_feature_importance_bar(explanation, top_n=10, row_height=0.28,
                                    entered_features=None):
    """Compact feature importance bar — no title; larger fonts for screenshots.

    If entered_features is provided, only user-entered features are shown
    (sorted by |SHAP|). Imputed features are not displayed here.
    """
    vals = np.abs(np.asarray(explanation['shap_values'], dtype=float).ravel())
    names = list(explanation['feature_names'])
    shap_signed = np.asarray(explanation['shap_values'], dtype=float).ravel()

    if entered_features is not None:
        entered_set = set(entered_features)
        keep = [i for i, n in enumerate(names) if n in entered_set]
        if not keep:
            keep = list(range(len(names)))  # fallback if nothing entered
        vals_show = vals[keep]
        shap_show = shap_signed[keep]
        names_show = [names[i] for i in keep]
    else:
        idx_all = np.argsort(vals)[::-1][:top_n]
        vals_show = vals[idx_all]
        shap_show = shap_signed[idx_all]
        names_show = [names[i] for i in idx_all]

    # Sort the kept set by |SHAP| descending
    order = np.argsort(vals_show)[::-1]
    top_vals = vals_show[order]
    top_names = [names_show[i] for i in order]
    shap_sorted = shap_show[order]
    colors = [_RED if shap_sorted[i] > 0 else _BLUE for i in range(len(top_vals))]

    fs_y = 11.0
    fs_x = 10.0
    fs_val = 9.5

    fig_h = max(2.0, len(top_vals) * row_height + 0.55)
    fig, ax = plt.subplots(figsize=(5.2, fig_h), dpi=120)
    ax.barh(range(len(top_vals)), top_vals, color=colors, height=0.68,
            edgecolor='white', linewidth=0.3)
    ax.set_yticks(range(len(top_vals)))
    ax.set_yticklabels(top_names, fontsize=fs_y)
    ax.invert_yaxis()
    ax.set_xlabel('|SHAP value|', fontsize=fs_x)
    # no set_title — page Image label already shows "Feature Importance"
    ax.tick_params(axis='x', labelsize=fs_x - 0.5)
    ax.grid(True, alpha=0.28, axis='x', zorder=0)
    for spine in ('top', 'right'):
        ax.spines[spine].set_visible(False)
    xmax = float(top_vals.max()) if len(top_vals) else 1.0
    for i, v in enumerate(top_vals):
        ax.text(v + 0.015 * xmax, i, f"{v:.3f}", va='center', ha='left',
                fontsize=fs_val, color='#444444')
    ax.set_xlim(0, xmax * 1.18)
    fig.subplots_adjust(left=0.30, right=0.96, top=0.96, bottom=0.14)
    return fig


def generate_decision_explanation(explanation, patient_label="Patient", top_n=8,
                                  entered_features=None):
    """Generate decision explanation in HTML — height +10%.

    If entered_features is provided, the table lists each user-entered feature
    (sorted by |SHAP|) and folds all imputed features into a single gray row.
    """
    shap_vals = explanation['shap_values']
    feature_names = explanation['feature_names']
    raw_data = explanation['raw_data']
    base_val = explanation['expected_value']
    prediction = base_val + np.sum(shap_vals)

    pos_contribution = sum(v for v in shap_vals if v > 0)
    neg_contribution = sum(v for v in shap_vals if v < 0)

    if prediction > 0.5:
        direction = "High Risk"
        dir_color = "#e74c3c"
    else:
        direction = "Low Risk"
        dir_color = "#2ecc71"

    if entered_features is not None:
        entered_set = set(entered_features)
        entered_idx = [i for i, n in enumerate(feature_names) if n in entered_set]
        imputed_idx = [i for i, n in enumerate(feature_names) if n not in entered_set]
        # Sort entered features by |SHAP| descending
        entered_idx_sorted = sorted(entered_idx,
                                     key=lambda i: -abs(float(shap_vals[i])))
        show_idx = entered_idx_sorted
    else:
        abs_vals = np.abs(shap_vals)
        show_idx = list(np.argsort(abs_vals)[::-1][:top_n])
        imputed_idx = []

    rows = []
    for rank, i in enumerate(show_idx, 1):
        sv = float(shap_vals[i])
        fn = feature_names[i]
        fv = float(raw_data[i])
        group = _get_feature_group(fn)
        if sv > 0:
            color = "#e74c3c"
            arrow = "&#x2191;"  # risk up
        else:
            color = "#3498db"
            arrow = "&#x2193;"  # risk down
        pct = abs(sv) / (abs(pos_contribution) + abs(neg_contribution) + 1e-10) * 100
        rows.append(
            f'<tr>'
            f'<td style="text-align:center; padding:2px 5px;">{rank}</td>'
            f'<td style="padding:2px 5px;"><b>{fn}</b></td>'
            f'<td style="padding:2px 5px; color:#666;">{group}</td>'
            f'<td style="text-align:center; padding:2px 5px;">{fv:.3f}</td>'
            f'<td style="text-align:center; padding:2px 5px; color:{color}; font-weight:bold;">'
            f'{arrow} {sv:+.3f}</td>'
            f'<td style="text-align:right; padding:2px 5px;">{pct:.1f}%</td>'
            f'</tr>'
        )

    if imputed_idx:
        imp_sum = float(sum(float(shap_vals[i]) for i in imputed_idx))
        imp_color = "#e74c3c" if imp_sum > 0 else "#3498db"
        imp_arrow = "&#x2191;" if imp_sum > 0 else "&#x2193;"
        imp_pct = abs(imp_sum) / (abs(pos_contribution) + abs(neg_contribution) + 1e-10) * 100
        rows.append(
            f'<tr style="background:#f5f5f5; color:#888;">'
            f'<td style="text-align:center; padding:2px 5px;">—</td>'
            f'<td style="padding:2px 5px;" colspan="2"><i>{len(imputed_idx)} '
            f'imputed features (training mean)</i></td>'
            f'<td style="text-align:center; padding:2px 5px;"><i>auto</i></td>'
            f'<td style="text-align:center; padding:2px 5px; color:{imp_color}; font-weight:bold;">'
            f'{imp_arrow} {imp_sum:+.3f}</td>'
            f'<td style="text-align:right; padding:2px 5px;">{imp_pct:.1f}%</td>'
            f'</tr>'
        )

    n_entered = len(show_idx)
    n_imputed = len(imputed_idx) if entered_features is not None else 0
    summary_line = ''  # annotation removed per request — no uncertainty display

    html = f"""
    <div style="font-family: system-ui, sans-serif; border: 1px solid #d0d7de; border-radius: 6px;
                padding: 8px 10px; margin: 0; min-height:231px; max-height:254px; overflow:auto;
                background:#fafbfc;">
      <div style="margin-bottom:4px;">
        <span style="font-weight:700; font-size:15px;">Decision — {patient_label}:</span>
        <span style="font-size:14px;">
          Base <b>{base_val:.3f}</b> &rarr; Pred <b style="color:{dir_color};">{prediction:.3f}</b> ({direction})
          &nbsp;|&nbsp; + <span style="color:#e74c3c;">{pos_contribution:.3f}</span>
          &nbsp;|&nbsp; &minus; <span style="color:#3498db;">{abs(neg_contribution):.3f}</span>
          {summary_line}
        </span>
      </div>
      <table style="width:100%; border-collapse:collapse; font-size:13px;">
        <tr style="border-bottom:2px solid #d0d7de; font-weight:700; background:#f0f3f6;">
          <td style="padding:3px 5px;">#</td>
          <td style="padding:3px 5px;">Feature</td>
          <td style="padding:3px 5px;">Dim</td>
          <td style="padding:3px 5px;">Value</td>
          <td style="padding:3px 5px;">SHAP</td>
          <td style="padding:3px 5px;">%</td>
        </tr>
        {''.join(rows)}
      </table>
    </div>
    """
    return html
