# -*- coding: utf-8 -*-
"""
Three-layer evidence chain for the locked SVM: Kernel SHAP (L1), drop-one
modality (L2), and LASSO sparse surrogate (L3).
"""

import os
import warnings
import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from sklearn.base import clone
from sklearn.svm import SVC
from sklearn.linear_model import Lasso
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.metrics import roc_auc_score
from scipy.stats import norm
import shap

warnings.filterwarnings('ignore')

# ==================== Configuration ====================
RANDOM_SEED = 42
RNG = np.random.RandomState(RANDOM_SEED)
N_CV_FOLDS = 5
DPI = 300
SHAP_SAMPLE = 200
SHAP_BACKGROUND = 50
N_BOOTSTRAP_CI = 1000         # Drop-One ΔAUC bootstrap
N_BOOTSTRAP_STABILITY = 50    # Lasso 稳定性选择
FIDELITY_THRESHOLD = 0.95     # Lasso 拐点：Surrogate AUC ≥ 95% × SVM AUC
STABILITY_THRESHOLD = 0.80    # 稳定性阈值

# ==================== Plot style (Nature soft, fonts bumped up) ====================
mpl.rcParams['font.family'] = 'Times New Roman'
mpl.rcParams['mathtext.fontset'] = 'stix'
mpl.rcParams['pdf.fonttype'] = 42
mpl.rcParams['ps.fonttype'] = 42
mpl.rcParams['font.size'] = 13                  # was 11
mpl.rcParams['axes.titlesize'] = 15             # was 13
mpl.rcParams['axes.titleweight'] = 'bold'
mpl.rcParams['axes.labelsize'] = 14             # was 12
mpl.rcParams['xtick.labelsize'] = 13            # was 10
mpl.rcParams['ytick.labelsize'] = 13            # was 10
mpl.rcParams['legend.fontsize'] = 12            # was 9
mpl.rcParams['legend.frameon'] = True
mpl.rcParams['axes.linewidth'] = 1.0
mpl.rcParams['grid.linewidth'] = 0.6
mpl.rcParams['axes.spines.top'] = False
mpl.rcParams['axes.spines.right'] = False
mpl.rcParams['axes.edgecolor'] = '#333333'
mpl.rcParams['axes.labelcolor'] = '#222222'
mpl.rcParams['xtick.color'] = '#333333'
mpl.rcParams['ytick.color'] = '#333333'
mpl.rcParams['figure.facecolor'] = 'white'
mpl.rcParams['savefig.facecolor'] = 'white'
mpl.rcParams['axes.grid'] = True
mpl.rcParams['grid.color'] = '#E5E5E5'
mpl.rcParams['grid.linestyle'] = '--'
mpl.rcParams['axes.axisbelow'] = True

# ==================== Modality mapping ====================
MODALITY_PREFIX_MAP = {
    'Face':   ['color', 'lipcolor', 'wholecolor', 'GLCM'],
    'Tongue': ['TB', 'TC', 'Per'],
    'Pulse':  ['h', 'w', 't'],
}
MODALITY_COLORS = {'Face': '#C26F5E', 'Tongue': '#5B8BCF', 'Pulse': '#6FB58C'}
MODALITY_COLORS_ZH = {'面': '#C26F5E', '舌': '#5B8BCF', '脉': '#6FB58C'}
NEUTRAL_GRAY = '#4A4A4A'
KNEE_GREEN = '#5BA85B'


def add_panel_label(ax, label, x=-0.10, y=1.05):
    ax.text(x, y, label, transform=ax.transAxes,
            fontsize=16, fontweight='bold', va='top', ha='left')


def assign_modality(feature_name):
    name = str(feature_name)
    for modality, prefixes in MODALITY_PREFIX_MAP.items():
        for p in prefixes:
            if name == p or name.startswith(p):
                return modality
    return 'Unmatched'


def features_in_combination(feature_names, modalities):
    return [f for f in feature_names if assign_modality(f) in modalities]


# ==================== Shared utilities ====================
def make_preproc():
    return Pipeline([('imputer', SimpleImputer(strategy='median')),
                     ('scaler', StandardScaler())])


def make_svm():
    return SVC(C=1.0, kernel='rbf', gamma='scale',
               class_weight='balanced', probability=True,
               random_state=RANDOM_SEED)


def compute_auc_ci_delong(y_true, y_proba, alpha=0.05):
    y_true = np.asarray(y_true).ravel()
    y_proba = np.asarray(y_proba).ravel()
    pos_idx = y_true == 1
    neg_idx = y_true == 0
    pos_scores = y_proba[pos_idx]
    neg_scores = y_proba[neg_idx]
    n_pos, n_neg = len(pos_scores), len(neg_scores)
    if n_pos == 0 or n_neg == 0:
        return np.nan, np.nan, np.nan
    pos_sorted = np.sort(pos_scores)
    neg_sorted = np.sort(neg_scores)
    V10 = np.array([np.searchsorted(neg_sorted, s, side='right') / n_neg for s in pos_scores])
    V01 = np.array([np.searchsorted(pos_sorted, s, side='left') / n_pos for s in neg_scores])
    auc_val = roc_auc_score(y_true, y_proba)
    var = max(np.var(V10, ddof=1) / n_pos + np.var(V01, ddof=1) / n_neg, 1e-10)
    se = np.sqrt(var)
    z = norm.ppf(1 - alpha / 2)
    return auc_val, max(0.0, auc_val - z * se), min(1.0, auc_val + z * se)


def bootstrap_delta_auc_ci(y_true, proba_full, proba_reduced, n_bootstrap=N_BOOTSTRAP_CI):
    y_true = np.asarray(y_true)
    proba_full = np.asarray(proba_full)
    proba_reduced = np.asarray(proba_reduced)
    pos_idx_all = np.where(y_true == 1)[0]
    neg_idx_all = np.where(y_true == 0)[0]
    n_pos = len(pos_idx_all)
    n_neg = len(neg_idx_all)
    deltas = np.empty(n_bootstrap)
    rng_local = np.random.RandomState(RANDOM_SEED + 1)
    for b in range(n_bootstrap):
        idx_pos = rng_local.choice(pos_idx_all, n_pos, replace=True)
        idx_neg = rng_local.choice(neg_idx_all, n_neg, replace=True)
        idx = np.concatenate([idx_pos, idx_neg])
        try:
            auc_full = roc_auc_score(y_true[idx], proba_full[idx])
            auc_red = roc_auc_score(y_true[idx], proba_reduced[idx])
            deltas[b] = auc_full - auc_red
        except Exception:
            deltas[b] = np.nan
    deltas = deltas[~np.isnan(deltas)]
    return float(np.percentile(deltas, 2.5)), float(np.percentile(deltas, 97.5))


def train_svm(X_train, y_train, X_test, y_test):
    """训练锁定 SVM + 5 折 CV；报告 CV AUC ± std + Test AUC + DeLong 95% CI。"""
    preproc = make_preproc()
    kfold = StratifiedKFold(n_splits=N_CV_FOLDS, shuffle=True, random_state=RANDOM_SEED)
    pipe = Pipeline([('preprocessor', clone(preproc)), ('model', make_svm())])
    cv_scores = cross_val_score(pipe, X_train, y_train, cv=kfold, scoring='roc_auc', n_jobs=-1)
    cv_auc, cv_std = cv_scores.mean(), cv_scores.std()
    pipe.fit(X_train, y_train)
    y_proba = pipe.predict_proba(X_test)[:, 1]
    test_auc = roc_auc_score(y_test, y_proba)
    _, ci_lo, ci_hi = compute_auc_ci_delong(y_test, y_proba)
    print(f'  SVM (RBF, C=1.0, gamma=scale) | '
          f'CV AUC = {cv_auc:.4f} ± {cv_std:.4f} | '
          f'Test AUC = {test_auc:.4f} (95% CI: {ci_lo:.3f}-{ci_hi:.3f})')
    return {'model': pipe, 'cv_auc': cv_auc, 'cv_std': cv_std,
            'test_auc': test_auc, 'ci_lo': ci_lo, 'ci_hi': ci_hi,
            'y_proba': y_proba}


# =====================================================================
# L1: SHAP aggregation by modality (KernelExplainer)
# =====================================================================
def compute_shap(model, X_train, sample_size=SHAP_SAMPLE):
    preproc = model.named_steps['preprocessor']
    X_proc = preproc.transform(X_train)
    try:
        background = shap.kmeans(X_proc, SHAP_BACKGROUND)
    except Exception:
        bg_idx = RNG.choice(X_proc.shape[0], min(SHAP_BACKGROUND, X_proc.shape[0]), replace=False)
        background = X_proc[bg_idx]
    if X_proc.shape[0] > sample_size:
        idx = RNG.choice(X_proc.shape[0], sample_size, replace=False)
        X_sample = X_proc[idx]
    else:
        X_sample = X_proc
    est = model.named_steps['model']
    print(f'  KernelExplainer: background={SHAP_BACKGROUND}, samples={X_sample.shape[0]}, '
          f'features={X_sample.shape[1]} (可能耗时 10-20 分钟)...')
    explainer = shap.KernelExplainer(lambda arr: est.predict_proba(arr)[:, 1], background)
    shap_vals = explainer.shap_values(X_sample, silent=False)
    if hasattr(shap_vals, 'values'):
        sv = np.asarray(shap_vals.values)
    else:
        sv = np.asarray(shap_vals)
    if sv.ndim == 3:
        sv = sv[:, :, 1]
    elif isinstance(shap_vals, list) and len(shap_vals) == 2:
        sv = np.asarray(shap_vals[1])
    return sv, X_sample


def aggregate_shap_by_modality(shap_vals, feature_names):
    abs_arr = np.abs(shap_vals)
    rows = []
    for modality in ['Face', 'Tongue', 'Pulse', 'Unmatched']:
        cols = [i for i, f in enumerate(feature_names) if assign_modality(f) == modality]
        if not cols:
            continue
        sub = abs_arr[:, cols]
        rows.append({
            'Modality': modality,
            'N_Features': len(cols),
            'Sum_SHAP': float(sub.sum()),
            'Mean_SHAP': float(sub.mean()),
            'Max_SHAP': float(sub.max()),
        })
    df = pd.DataFrame(rows)
    total = df['Sum_SHAP'].sum()
    df['Sum_Percent'] = df['Sum_SHAP'] / total * 100
    return df


def plot_shap_aggregation(shap_df, save_path):
    """双面板棒棒糖：A = Sum|SHAP| 占比，B = 每特征 Mean|SHAP| 密度（点大小 = N_features）。"""
    fig, axes = plt.subplots(1, 2, figsize=(13, 5.6))

    n_max = shap_df['N_Features'].max()

    def size_for(n):
        return 120 + (n / n_max) * 380

    # Panel A: Sum |SHAP| share (%)
    df_s = shap_df.sort_values('Sum_Percent', ascending=True).reset_index(drop=True)
    colors_s = [MODALITY_COLORS.get(m, '#888') for m in df_s['Modality']]
    axes[0].hlines(df_s['Modality'], 0, df_s['Sum_Percent'],
                   color=colors_s, lw=2.6, alpha=0.75, zorder=2)
    axes[0].scatter(df_s['Sum_Percent'], df_s['Modality'],
                    s=360, color=colors_s, edgecolor='black',
                    linewidth=1.3, zorder=3)
    for m, v in zip(df_s['Modality'], df_s['Sum_Percent']):
        axes[0].annotate(f'{v:.1f}%', (v, m),
                         xytext=(20, 0), textcoords='offset points',
                         fontsize=13, fontweight='bold', va='center')
    axes[0].set_xlabel('Share of Total |SHAP| (%)')
    axes[0].set_xlim(0, df_s['Sum_Percent'].max() * 1.32)
    axes[0].set_title('Total Modality Contribution')
    add_panel_label(axes[0], 'A')

    # Panel B: Mean |SHAP| density
    df_m = shap_df.sort_values('Mean_SHAP', ascending=True).reset_index(drop=True)
    colors_m = [MODALITY_COLORS.get(m, '#888') for m in df_m['Modality']]
    sizes_m = [size_for(n) for n in df_m['N_Features']]
    axes[1].hlines(df_m['Modality'], 0, df_m['Mean_SHAP'],
                   color=colors_m, lw=2.6, alpha=0.75, zorder=2)
    axes[1].scatter(df_m['Mean_SHAP'], df_m['Modality'],
                    s=sizes_m, color=colors_m, edgecolor='black',
                    linewidth=1.3, zorder=3)
    for m, v in zip(df_m['Modality'], df_m['Mean_SHAP']):
        axes[1].annotate(f'{v:.3f}', (v, m),
                         xytext=(24, 0), textcoords='offset points',
                         fontsize=12, fontweight='bold', va='center')
    axes[1].set_xlabel('Mean |SHAP| per feature (density)')
    axes[1].set_xlim(0, df_m['Mean_SHAP'].max() * 1.50)
    axes[1].set_title('Per-feature Importance Density')
    add_panel_label(axes[1], 'B')

    n_feat_lookup = dict(zip(shap_df['Modality'], shap_df['N_Features']))
    legend_handles = []
    for modality in ['Face', 'Tongue', 'Pulse']:
        n_feat = n_feat_lookup.get(modality, 0)
        s = size_for(n_feat)
        legend_handles.append(
            Line2D([0], [0], marker='o', color='w',
                   markersize=np.sqrt(s) * 1.0,
                   markerfacecolor=MODALITY_COLORS[modality],
                   markeredgecolor='black', markeredgewidth=1.0,
                   label=f'{modality}  (n = {n_feat})'))
    legend = axes[1].legend(
        handles=legend_handles, loc='lower right',
        frameon=True, fontsize=11, framealpha=0.95,
        borderpad=1.1, labelspacing=1.4,
        handletextpad=1.2, borderaxespad=1.0,
        title='Modality  (marker size = n features)',
        title_fontsize=11)
    legend.get_frame().set_edgecolor('#333333')
    legend.get_frame().set_linewidth(0.8)

    plt.tight_layout(pad=1.2, w_pad=3)
    plt.savefig(save_path, dpi=DPI, bbox_inches='tight')
    plt.close()


def plot_shap_beeswarm(shap_vals, X_sample, feature_names, save_path, top_k=20):
    plt.figure(figsize=(10, 8))
    shap.summary_plot(shap_vals, X_sample, feature_names=feature_names,
                      show=False, max_display=top_k)
    plt.title('SVM Fusion Model - SHAP Beeswarm (Top 20 Features)',
              fontsize=14, fontweight='bold', pad=12)
    plt.tight_layout()
    plt.savefig(save_path, dpi=DPI, bbox_inches='tight')
    plt.close()


# =====================================================================
# L2: Drop-One-Modality (SVM + Bootstrap CI)
# =====================================================================
def drop_one_modality(X_train, y_train, X_test, y_test, feature_names, baseline_auc,
                      baseline_proba):
    kfold = StratifiedKFold(n_splits=N_CV_FOLDS, shuffle=True, random_state=RANDOM_SEED)
    preproc = make_preproc()
    results = []
    for modality in ['Face', 'Tongue', 'Pulse']:
        keep = [m for m in ['Face', 'Tongue', 'Pulse'] if m != modality]
        feats = features_in_combination(feature_names, keep)
        pipe = Pipeline([('preprocessor', clone(preproc)), ('model', make_svm())])
        cv_auc = cross_val_score(pipe, X_train[feats], y_train, cv=kfold,
                                 scoring='roc_auc', n_jobs=-1).mean()
        pipe.fit(X_train[feats], y_train)
        y_proba_red = pipe.predict_proba(X_test[feats])[:, 1]
        test_auc = roc_auc_score(y_test, y_proba_red)
        delta_cv = baseline_auc['cv'] - cv_auc
        delta_test = baseline_auc['test'] - test_auc
        ci_lo, ci_hi = bootstrap_delta_auc_ci(y_test, baseline_proba, y_proba_red)
        _, auc_lo, auc_hi = compute_auc_ci_delong(y_test, y_proba_red)
        print(f'  w/o {modality:<8s} ({len(feats):3d} feats): '
              f'CV AUC = {cv_auc:.4f} | Test AUC = {test_auc:.4f} '
              f'(DeLong 95% CI: {auc_lo:.3f}-{auc_hi:.3f}) | '
              f'Δ = {delta_test:+.4f} (Bootstrap 95% CI: {ci_lo:+.4f} to {ci_hi:+.4f})')
        results.append({
            'Dropped_Modality': modality,
            'Remaining_Features': len(feats),
            'CV_AUC': cv_auc,
            'Test_AUC': test_auc,
            'AUC_Lower': auc_lo,
            'AUC_Upper': auc_hi,
            'Delta_CV_AUC': delta_cv,
            'Delta_Test_AUC': delta_test,
            'Delta_CI_Lower': ci_lo,
            'Delta_CI_Upper': ci_hi,
        })
    df = pd.DataFrame(results).sort_values('Delta_Test_AUC', ascending=False).reset_index(drop=True)
    total = df['Delta_Test_AUC'].sum()
    df['Incremental_Value_Percent'] = df['Delta_Test_AUC'] / total * 100
    return df


def plot_drop_one(drop_df, baseline_auc, baseline_ci, save_path):
    fig, axes = plt.subplots(1, 2, figsize=(14.5, 5.4),
                             gridspec_kw={'width_ratios': [1, 1.25]})

    # ---------- Panel A: Absolute AUC with DeLong CI ----------
    scenarios = ['Baseline\n(SVM, 121 feat.)',
                 'w/o Face\n(38 feat.)',
                 'w/o Tongue\n(93 feat.)',
                 'w/o Pulse\n(111 feat.)']
    aucs = [baseline_auc] + drop_df['Test_AUC'].tolist()
    ci_lo = [baseline_ci[0]] + drop_df['AUC_Lower'].tolist()
    ci_hi = [baseline_ci[1]] + drop_df['AUC_Upper'].tolist()
    colors = [NEUTRAL_GRAY] + [MODALITY_COLORS.get(m, '#888')
                               for m in drop_df['Dropped_Modality']]

    for i, (s, a, lo, hi, c) in enumerate(zip(scenarios, aucs, ci_lo, ci_hi, colors)):
        axes[0].errorbar(i, a, yerr=[[a - lo], [hi - a]], fmt='o', ms=12,
                         color=c, mfc=c, mec='black', mew=1.3,
                         ecolor='#555555', capsize=5, capthick=1.3,
                         elinewidth=1.5, zorder=3)
        axes[0].annotate(f'{a:.3f}', (i, a),
                         xytext=(16, 0), textcoords='offset points',
                         fontsize=12, fontweight='bold',
                         va='center', ha='left')
    axes[0].axhline(baseline_auc, color=NEUTRAL_GRAY,
                    linestyle=':', alpha=0.5, lw=1)
    axes[0].set_xticks(range(len(scenarios)))
    axes[0].set_xticklabels(scenarios, fontsize=11)
    axes[0].set_ylabel('Test AUC (95% DeLong CI)')
    axes[0].set_ylim(min(ci_lo) - 0.04, max(ci_hi) + 0.04)
    axes[0].set_xlim(-0.5, len(scenarios) - 0.5 + 0.8)
    axes[0].set_title('Absolute AUC across scenarios')
    add_panel_label(axes[0], 'A')

    # ---------- Panel B: ΔAUC forest plot ----------
    df_f = drop_df.sort_values('Delta_Test_AUC', ascending=True).reset_index(drop=True)
    colors_f = [MODALITY_COLORS.get(m, '#888') for m in df_f['Dropped_Modality']]
    y_pos = np.arange(len(df_f))
    cap_h = 0.18

    for i, (_, row) in enumerate(df_f.iterrows()):
        lo, hi = row['Delta_CI_Lower'], row['Delta_CI_Upper']
        axes[1].plot([lo, hi], [i, i], color=colors_f[i], lw=2.8, zorder=2)
        axes[1].plot([lo, lo], [i - cap_h, i + cap_h], color=colors_f[i], lw=1.8)
        axes[1].plot([hi, hi], [i - cap_h, i + cap_h], color=colors_f[i], lw=1.8)
        axes[1].scatter(row['Delta_Test_AUC'], i, s=260, marker='o',
                        color=colors_f[i], edgecolor='black',
                        linewidth=1.3, zorder=3)
        axes[1].text(hi + 0.003, i,
                     f'{row["Delta_Test_AUC"]:+.3f} '
                     f'[{lo:+.3f}, {hi:+.3f}]  '
                     f'({row["Incremental_Value_Percent"]:.1f}%)',
                     va='center', fontsize=12)

    axes[1].axvline(0, color='black', lw=1.2, linestyle='-', alpha=0.7, zorder=1)
    axes[1].set_yticks(y_pos)
    axes[1].set_yticklabels([f'w/o {m}' for m in df_f['Dropped_Modality']])
    axes[1].set_xlabel('ΔAUC (Baseline − Drop-One), 95% Bootstrap CI')
    axes[1].set_title('Functional incremental value')
    x_min = df_f['Delta_CI_Lower'].min()
    x_max = df_f['Delta_CI_Upper'].max()
    span = x_max - x_min
    axes[1].set_xlim(x_min - span * 0.08, x_max + span * 0.95)
    add_panel_label(axes[1], 'B')

    plt.tight_layout(pad=1.2, w_pad=3)
    plt.savefig(save_path, dpi=DPI, bbox_inches='tight')
    plt.close()


# =====================================================================
# L3: Lasso L1 Surrogate (路径 + 拐点 + 稳定性 + 核心特征可视化)
# =====================================================================
def fit_lasso_surrogate(X_train_distill, y_distill_continuous, alpha):
    lasso = Lasso(alpha=alpha, max_iter=10000, random_state=RANDOM_SEED, selection='cyclic')
    lasso.fit(X_train_distill, y_distill_continuous)
    return lasso


def count_nonzero(coef):
    return int(np.sum(np.abs(coef.ravel()) > 1e-10))


def evaluate_lasso_path(X_train_proc, y_train_pred_proba, X_test_proc, y_test, y_test_pred_proba,
                        feature_names, alpha_list):
    rows = []
    svm_auc = roc_auc_score(y_test, y_test_pred_proba)
    for alpha in alpha_list:
        lasso = fit_lasso_surrogate(X_train_proc, y_train_pred_proba, alpha)
        coef = lasso.coef_.ravel()
        n_nonzero = count_nonzero(coef)
        raw_test = lasso.predict(X_test_proc)
        auc_real = roc_auc_score(y_test, raw_test)
        corr = float(np.corrcoef(raw_test, y_test_pred_proba)[0, 1]) if raw_test.std() > 0 else 0
        auc_drop = svm_auc - auc_real
        selected_mask = np.abs(coef) > 1e-10
        selected_features = [feature_names[i] for i in range(len(feature_names)) if selected_mask[i]]
        rows.append({
            'Alpha': alpha,
            'N_Features': n_nonzero,
            'Surrogate_AUC': auc_real,
            'SVM_AUC': svm_auc,
            'AUC_Drop_vs_SVM': auc_drop,
            'Fidelity_Corr': corr,
            'Selected_Features': selected_features,
        })
        print(f'  alpha={alpha:<8.5f} | 特征数={n_nonzero:3d} | '
              f'Surrogate AUC={auc_real:.4f} | ΔAUC vs SVM={auc_drop:+.4f} | '
              f'保真度(corr)={corr:.4f}')
    return pd.DataFrame(rows), lasso


def find_knee(path_df, svm_auc, threshold=FIDELITY_THRESHOLD):
    target = threshold * svm_auc
    eligible = path_df[path_df['Surrogate_AUC'] >= target]
    if eligible.empty:
        return path_df.sort_values('Surrogate_AUC', ascending=False).iloc[0]
    return eligible.sort_values('N_Features').iloc[0]


def stability_selection(X_train_proc, y_train_pred_proba, alpha_target,
                        n_bootstrap=N_BOOTSTRAP_STABILITY):
    n, p = X_train_proc.shape
    freq = np.zeros(p)
    pos_idx_all = np.where(y_train_pred_proba >= 0.5)[0]
    neg_idx_all = np.where(y_train_pred_proba < 0.5)[0]
    for _ in range(n_bootstrap):
        n_pos = len(pos_idx_all)
        n_neg = len(neg_idx_all)
        if n_pos == 0 or n_neg == 0:
            idx = RNG.randint(0, n, n)
        else:
            idx = np.concatenate([
                RNG.choice(pos_idx_all, n_pos, replace=True),
                RNG.choice(neg_idx_all, n_neg, replace=True),
            ])
        X_b = X_train_proc[idx]
        y_b = y_train_pred_proba[idx]
        try:
            lasso = Lasso(alpha=alpha_target, max_iter=10000,
                          random_state=RANDOM_SEED, selection='cyclic')
            lasso.fit(X_b, y_b)
            coef = lasso.coef_.ravel()
            freq += (np.abs(coef) > 1e-10).astype(float)
        except Exception:
            continue
    return freq / n_bootstrap


def plot_lasso_results(path_df, knee_row, save_path):
    fig, axes = plt.subplots(1, 2, figsize=(14, 5.4))

    # Panel A: AUC vs Sparsity
    axes[0].plot(path_df['N_Features'], path_df['Surrogate_AUC'],
                 '-o', color='#5B8BCF', lw=2, ms=9,
                 markeredgecolor='black', markeredgewidth=0.8,
                 label='Lasso surrogate', zorder=3)
    axes[0].fill_between(path_df['N_Features'], path_df['Surrogate_AUC'],
                         alpha=0.12, color='#5B8BCF')
    axes[0].axhline(knee_row['SVM_AUC'], color='#C26F5E',
                    linestyle='--', lw=1.5,
                    label=f"SVM AUC = {knee_row['SVM_AUC']:.4f}")
    axes[0].axhline(FIDELITY_THRESHOLD * knee_row['SVM_AUC'], color='gray',
                    linestyle=':', lw=1.2, label='95% threshold')
    axes[0].scatter([knee_row['N_Features']], [knee_row['Surrogate_AUC']],
                    s=240, marker='*', color=KNEE_GREEN,
                    edgecolor='black', linewidth=1, zorder=5,
                    label=f"Knee ({int(knee_row['N_Features'])} feat.)")
    axes[0].annotate(f"Knee point\nN={int(knee_row['N_Features'])}, "
                     f"AUC={knee_row['Surrogate_AUC']:.4f}",
                     xy=(knee_row['N_Features'], knee_row['Surrogate_AUC']),
                     xytext=(0.55, 0.25), textcoords='axes fraction',
                     fontsize=11, ha='left',
                     bbox=dict(boxstyle='round,pad=0.4',
                               facecolor='white', edgecolor=KNEE_GREEN, lw=1))
    axes[0].set_xlabel('Number of Non-zero Features')
    axes[0].set_ylabel('Surrogate AUC')
    axes[0].set_title('AUC vs Sparsity (L1 path)')
    axes[0].legend(frameon=True, fontsize=11, loc='lower right')
    add_panel_label(axes[0], 'A')

    # Panel B: Fidelity correlation
    axes[1].plot(path_df['N_Features'], path_df['Fidelity_Corr'],
                 '-s', color='#6FB58C', lw=2, ms=8,
                 markeredgecolor='black', markeredgewidth=0.8,
                 label='Pearson r', zorder=3)
    axes[1].fill_between(path_df['N_Features'], path_df['Fidelity_Corr'],
                         alpha=0.12, color='#6FB58C')
    axes[1].axhline(0.9, color='gray', linestyle=':', lw=1.2,
                    label='r = 0.9 reference')
    axes[1].set_xlabel('Number of Non-zero Features')
    axes[1].set_ylabel('Correlation with SVM predictions')
    axes[1].set_title('Surrogate Fidelity to SVM')
    axes[1].set_ylim(0, 1.05)
    axes[1].legend(frameon=True, fontsize=11, loc='lower right')
    add_panel_label(axes[1], 'B')

    plt.tight_layout(pad=1.2, w_pad=3)
    plt.savefig(save_path, dpi=DPI, bbox_inches='tight')
    plt.close()


def plot_core_features(final_df, save_path, svm_auc, final_auc):
    """核心特征棒棒糖图。

    修复点（vs 旧版）：
      1) 字体普遍上调（fontsize 8 → 12）
      2) 负系数标签避让 y 轴：通过显式 xlim 在左侧留出 ~18% 余量；
         同时把 offset 调大一点，确保标签与棒棒糖端点之间有视觉间距
      3) 棒棒糖线更粗、点边框更明显
    """
    sorted_df = final_df.sort_values(
        ['Modality', 'Abs_Coefficient'], ascending=[True, True]).reset_index(drop=True)
    colors = [MODALITY_COLORS.get(m, '#888888') for m in sorted_df['Modality']]
    sizes = 80 + (sorted_df['Selection_Frequency'] - 0.80) / 0.20 * 280

    fig, ax = plt.subplots(figsize=(11, max(6.5, len(sorted_df) * 0.45)))
    ax.hlines(sorted_df['Feature'], 0, sorted_df['Coefficient'],
              color=colors, lw=2.0, alpha=0.78, zorder=2)
    ax.scatter(sorted_df['Coefficient'], sorted_df['Feature'],
               s=sizes, color=colors, edgecolor='black',
               linewidth=1.2, zorder=3)

    # ---- 标签放置（关键修复）----
    x_min = float(sorted_df['Coefficient'].min())
    x_max = float(sorted_df['Coefficient'].max())
    x_range = max(x_max - x_min, 0.01)
    # offset 用数据坐标的 3.5%（旧版 2%），给标签和端点之间留间距
    offset = x_range * 0.035

    for f, c in zip(sorted_df['Feature'], sorted_df['Coefficient']):
        if c >= 0:
            ax.text(c + offset, f, f'{c:+.3f}',
                    va='center', ha='left', fontsize=12, fontweight='bold')
        else:
            ax.text(c - offset, f, f'{c:+.3f}',
                    va='center', ha='right', fontsize=12, fontweight='bold')

    # 显式留出双侧边距：左侧 18% 给负系数标签，右侧 12% 给正系数标签
    ax.set_xlim(x_min - x_range * 0.18, x_max + x_range * 0.12)

    ax.axvline(0, color='black', lw=1.0, alpha=0.7)
    ax.set_xlabel('Lasso Coefficient (positive = risk ↑, negative = risk ↓)')
    ax.set_ylabel('')
    ax.set_title(f'Interpretable Core Features '
                 f'(N={len(sorted_df)}, AUC={final_auc:.4f} vs SVM {svm_auc:.4f})',
                 pad=12)
    ax.tick_params(axis='both', labelsize=12)
    ax.grid(axis='x', linestyle='--', alpha=0.4)

    # 给 y 轴特征名留出更多左侧空间
    plt.subplots_adjust(left=0.32)

    legend_elems = [
        Line2D([0], [0], marker='o', color='w', markersize=13,
               markerfacecolor=MODALITY_COLORS['Face'],
               markeredgecolor='black', label='Face'),
        Line2D([0], [0], marker='o', color='w', markersize=13,
               markerfacecolor=MODALITY_COLORS['Tongue'],
               markeredgecolor='black', label='Tongue'),
        Line2D([0], [0], marker='o', color='w', markersize=13,
               markerfacecolor=MODALITY_COLORS['Pulse'],
               markeredgecolor='black', label='Pulse'),
        Line2D([0], [0], marker='o', color='w', markersize=13,
               markerfacecolor='gray', alpha=0.5,
               markeredgecolor='black', label='Marker size = stability'),
    ]
    ax.legend(handles=legend_elems, loc='lower right', frameon=True, fontsize=11)

    plt.savefig(save_path, dpi=DPI, bbox_inches='tight')
    plt.close()


# =====================================================================
# Main: 三层串行执行
# =====================================================================
def main(data_path, output_dir='results_3layer_evidence', target_column='target'):
    l1_dir = os.path.join(output_dir, 'L1_SHAP')
    l2_dir = os.path.join(output_dir, 'L2_DropOne')
    l3_dir = os.path.join(output_dir, 'L3_Lasso')
    for d in [output_dir, l1_dir, l2_dir, l3_dir]:
        os.makedirs(d, exist_ok=True)

    print('=' * 70)
    print('三层证据链合一分析')
    print(f'  L1: SHAP 边际 (KernelExplainer, n={SHAP_SAMPLE})')
    print(f'  L2: Drop-One 功能消融 (Bootstrap CI = {N_BOOTSTRAP_CI})')
    print(f'  L3: Lasso 稀疏代理 (拐点阈值 = {FIDELITY_THRESHOLD}, 稳定性 = {N_BOOTSTRAP_STABILITY}×bts)')
    print('=' * 70)

    data = pd.read_csv(data_path)
    X = data.drop(target_column, axis=1)
    y = data[target_column]
    feature_names = X.columns.tolist()
    pos = int((y == 1).sum()); neg = int((y == 0).sum())
    print(f'N = {len(X)} | Features = {len(feature_names)} | '
          f'Positive = {pos} ({pos / len(y) * 100:.1f}%) | Negative = {neg}')
    for m in ['Face', 'Tongue', 'Pulse']:
        n = sum(1 for f in feature_names if assign_modality(f) == m)
        print(f'  {m:<8s}: {n} features')

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=RANDOM_SEED, stratify=y)

    # ---------- Step 0: 训练锁定 SVM（L1/L2/L3 共享）----------
    print('\n[Step 0/4] 训练锁定 SVM（三层共用）...')
    svm_out = train_svm(X_train, y_train, X_test, y_test)
    fusion_model = svm_out['model']
    fusion_cv = svm_out['cv_auc']
    fusion_test = svm_out['test_auc']
    fusion_proba = svm_out['y_proba']

    pd.DataFrame([{
        'Model': 'SVM (RBF, locked)',
        'CV_AUC': fusion_cv, 'CV_Std': svm_out['cv_std'],
        'Test_AUC': fusion_test,
        'AUC_Lower': svm_out['ci_lo'], 'AUC_Upper': svm_out['ci_hi'],
    }]).to_csv(os.path.join(l1_dir, 'fusion_performance.csv'),
               index=False, encoding='utf-8-sig')

    # ---------- L1: SHAP 边际 ----------
    print('\n[L1/4] 计算 SHAP (KernelExplainer) 并按模态聚合...')
    shap_vals, X_sample = compute_shap(fusion_model, X_train)
    shap_df = aggregate_shap_by_modality(shap_vals, feature_names)
    print(shap_df[['Modality', 'N_Features', 'Sum_SHAP', 'Sum_Percent', 'Mean_SHAP']].to_string(
        index=False, float_format=lambda x: f'{x:.4f}'))
    shap_df.to_csv(os.path.join(l1_dir, 'modality_shap_summary.csv'),
                   index=False, encoding='utf-8-sig')

    mean_abs = np.abs(shap_vals).mean(axis=0)
    top_feat = pd.DataFrame({
        'Feature': feature_names,
        'Modality': [assign_modality(f) for f in feature_names],
        'Mean_SHAP': mean_abs,
    }).sort_values('Mean_SHAP', ascending=False).head(20).reset_index(drop=True)
    top_feat.to_csv(os.path.join(l1_dir, 'top20_features_shap.csv'),
                    index=False, encoding='utf-8-sig')

    plot_shap_aggregation(shap_df, os.path.join(l1_dir, 'shap_modality_aggregation.png'))
    plot_shap_beeswarm(shap_vals, X_sample, feature_names,
                       os.path.join(l1_dir, 'shap_beeswarm_top20.png'))

    # ---------- L2: Drop-One ----------
    print('\n[L2/4] Drop-One-Modality（SVM + Bootstrap CI）...')
    baseline_auc = {'cv': fusion_cv, 'test': fusion_test}
    drop_df = drop_one_modality(X_train, y_train, X_test, y_test, feature_names,
                                baseline_auc, fusion_proba)
    drop_df.to_csv(os.path.join(l2_dir, 'drop_one_modality.csv'),
                   index=False, encoding='utf-8-sig')
    print(drop_df.to_string(index=False, float_format=lambda x: f'{x:.4f}'))
    plot_drop_one(drop_df, fusion_test, (svm_out['ci_lo'], svm_out['ci_hi']),
                  os.path.join(l2_dir, 'drop_one_modality.png'))

    # ---------- L3: Lasso 稀疏代理 ----------
    print('\n[L3/4] Lasso L1 代理模型（蒸馏 SVM predict_proba）...')
    # 用 SVM 训练集 predict_proba 作为 Lasso 蒸馏目标
    preproc = fusion_model.named_steps['preprocessor']
    X_train_proc = preproc.transform(X_train)
    X_test_proc = preproc.transform(X_test)
    y_train_pred_proba = fusion_proba_train = fusion_model.predict_proba(X_train)[:, 1]
    y_test_pred_proba = fusion_proba

    print('\n  [3a] 扫描 L1 正则路径...')
    alpha_list = [0.001, 0.002, 0.005, 0.01, 0.02, 0.05, 0.1]
    path_df, _ = evaluate_lasso_path(
        X_train_proc, y_train_pred_proba, X_test_proc, y_test, y_test_pred_proba,
        feature_names, alpha_list)

    print('\n  [3b] 寻找拐点...')
    knee = find_knee(path_df, fusion_test, FIDELITY_THRESHOLD)
    print(f'  ✓ 拐点: alpha={knee["Alpha"]}, 特征数={int(knee["N_Features"])}, '
          f'Surrogate AUC={knee["Surrogate_AUC"]:.4f} '
          f'(SVM={fusion_test:.4f}, Δ={knee["AUC_Drop_vs_SVM"]:+.4f})')

    print(f'\n  [3c] 稳定性选择（bootstrap {N_BOOTSTRAP_STABILITY} 次，'
          f'阈值 {STABILITY_THRESHOLD:.0%}）...')
    freq = stability_selection(X_train_proc, y_train_pred_proba, knee['Alpha'])
    stable_mask = freq >= STABILITY_THRESHOLD
    stable_features = [feature_names[i] for i in range(len(feature_names)) if stable_mask[i]]
    print(f'  ✓ 稳定特征（频率≥{STABILITY_THRESHOLD:.0%}）: {len(stable_features)} 个')

    print('\n  [3d] 输出最终可解释核心特征清单...')
    if not stable_features:
        stable_features = knee['Selected_Features']
        stable_mask = np.array([f in stable_features for f in feature_names])
    X_stable_train = X_train_proc[:, stable_mask]
    X_stable_test = X_test_proc[:, stable_mask]
    final_lasso = Lasso(alpha=knee['Alpha'], max_iter=10000,
                        random_state=RANDOM_SEED, selection='cyclic')
    final_lasso.fit(X_stable_train, y_train_pred_proba)
    final_coef = final_lasso.coef_.ravel()
    final_auc = roc_auc_score(y_test, final_lasso.predict(X_stable_test))

    final_df = pd.DataFrame({
        'Feature': stable_features,
        'Modality': [assign_modality(f) for f in stable_features],
        'Coefficient': final_coef,
        'Abs_Coefficient': np.abs(final_coef),
        'Selection_Frequency': freq[stable_mask],
    }).sort_values('Abs_Coefficient', ascending=False).reset_index(drop=True)

    print(final_df.to_string(index=False, float_format=lambda x: f'{x:.4f}'))
    print('\n模态分布:')
    mod_summary = final_df.groupby('Modality').agg(
        N_Features=('Feature', 'count'),
        Sum_AbsCoef=('Abs_Coefficient', 'sum'),
        Mean_AbsCoef=('Abs_Coefficient', 'mean'),
    )
    print(mod_summary.to_string(float_format=lambda x: f'{x:.4f}'))

    print('\n性能对比:')
    print(f'  SVM 三模态（锁定）  : AUC = {fusion_test:.4f} (121 特征)')
    print(f'  Lasso 代理(完整稀疏): AUC = {knee["Surrogate_AUC"]:.4f} ({int(knee["N_Features"])} 特征)')
    print(f'  Lasso 最终稳定版    : AUC = {final_auc:.4f} ({len(final_df)} 特征)')
    print(f'  可解释性增益: 特征数 121 → {len(final_df)} (压缩比 {len(final_df) / 121 * 100:.1f}%)')
    print(f'  性能损失: AUC 下降 {fusion_test - final_auc:+.4f}')

    path_df.drop(columns=['Selected_Features']).to_csv(
        os.path.join(l3_dir, 'lasso_regularization_path.csv'),
        index=False, encoding='utf-8-sig')
    final_df.to_csv(os.path.join(l3_dir, 'interpretable_core_features.csv'),
                    index=False, encoding='utf-8-sig')
    mod_summary.to_csv(os.path.join(l3_dir, 'modality_distribution_in_core.csv'),
                       encoding='utf-8-sig')
    pd.DataFrame({
        'Feature': feature_names,
        'Modality': [assign_modality(f) for f in feature_names],
        'Selection_Frequency': freq,
    }).sort_values('Selection_Frequency', ascending=False).to_csv(
        os.path.join(l3_dir, 'all_features_stability_frequency.csv'),
        index=False, encoding='utf-8-sig')

    plot_lasso_results(path_df, knee,
                       os.path.join(l3_dir, 'lasso_surrogate_analysis.png'))
    plot_core_features(final_df,
                       os.path.join(l3_dir, 'core_features_coefficients.png'),
                       svm_auc=fusion_test, final_auc=final_auc)

    # ---------- 汇总 ----------
    print('\n' + '=' * 70)
    print('三层证据链汇总')
    print('=' * 70)
    print(f'SVM | CV AUC = {fusion_cv:.4f} ± {svm_out["cv_std"]:.4f} | '
          f'Test AUC = {fusion_test:.4f} '
          f'(95% CI: {svm_out["ci_lo"]:.3f}-{svm_out["ci_hi"]:.3f})')
    print(f'\n[L1] SHAP 模态贡献 (Sum |SHAP| share):')
    for _, row in shap_df.iterrows():
        print(f'  {row["Modality"]:<10s}: {row["Sum_Percent"]:.2f}% '
              f'(Mean per feature = {row["Mean_SHAP"]:.4f})')
    print(f'\n[L2] Drop-One 功能增量 (ΔAUC share, 95% Bootstrap CI):')
    for _, row in drop_df.iterrows():
        print(f'  w/o {row["Dropped_Modality"]:<8s}: ΔAUC = {row["Delta_Test_AUC"]:+.4f} '
              f'[{row["Delta_CI_Lower"]:+.4f}, {row["Delta_CI_Upper"]:+.4f}] '
              f'({row["Incremental_Value_Percent"]:.1f}% of total)')
    print(f'\n[L3] Lasso 稀疏代理: {len(final_df)} 个核心特征, '
          f'AUC = {final_auc:.4f} vs SVM {fusion_test:.4f}')

    print(f'\n结果保存至: {os.path.abspath(output_dir)}')
    print(f'  L1_SHAP/    (SHAP 边际)')
    print(f'    - fusion_performance.csv')
    print(f'    - modality_shap_summary.csv')
    print(f'    - top20_features_shap.csv')
    print(f'    - shap_modality_aggregation.png')
    print(f'    - shap_beeswarm_top20.png')
    print(f'  L2_DropOne/ (Drop-One 功能消融)')
    print(f'    - drop_one_modality.csv')
    print(f'    - drop_one_modality.png')
    print(f'  L3_Lasso/   (Lasso 稀疏代理)')
    print(f'    - lasso_regularization_path.csv')
    print(f'    - interpretable_core_features.csv')
    print(f'    - modality_distribution_in_core.csv')
    print(f'    - all_features_stability_frequency.csv')
    print(f'    - lasso_surrogate_analysis.png')
    print(f'    - core_features_coefficients.png')


if __name__ == '__main__':
    # Repo-relative defaults via config.py — no hardcoded absolute paths.
    import sys as _sys
    from pathlib import Path as _Path
    _sys.path.insert(0, str(_Path(__file__).resolve().parent))
    from config import FULL_DEV_CSV, RESULTS_DIR

    import argparse
    parser = argparse.ArgumentParser(description='三层证据链合一：SHAP + Drop-One + Lasso 代理 '
                                                 '(3-layer evidence chain: SHAP + Drop-One + LASSO surrogate)')
    parser.add_argument('--data', default=str(FULL_DEV_CSV),
                        help='开发队列 CSV 路径 (dev cohort CSV). Default: repo full dev matrix.')
    parser.add_argument('--out', default=str(RESULTS_DIR / 'three_layer_evidence'),
                        help='输出根目录（自动建 L1_SHAP / L2_DropOne / L3_Lasso 子目录）. '
                             'Default: results/three_layer_evidence')
    args = parser.parse_args()
    main(args.data, args.out, target_column='target')
