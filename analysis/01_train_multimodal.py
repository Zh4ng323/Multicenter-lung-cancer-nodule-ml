# -*- coding: utf-8 -*-
"""
Multimodal lung-cancer vs benign pulmonary nodule pipeline: nine-algorithm
10-fold CV, locked RBF-SVM (C=1.0, gamma=scale, class_weight=balanced),
internal/external evaluation, SHAP interpretation, and a deployable artifact.
"""

# ==================== Environment Configuration ====================
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
import re
import shutil
import traceback
from scipy import stats
from scipy.stats import norm
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.base import clone
from sklearn.model_selection import train_test_split, StratifiedKFold, GridSearchCV, cross_val_score
from sklearn.metrics import (roc_auc_score, accuracy_score, f1_score, precision_score,
                             recall_score, average_precision_score, confusion_matrix,
                             brier_score_loss, classification_report, roc_curve,
                             precision_recall_curve, auc, make_scorer)
from sklearn.calibration import calibration_curve, CalibratedClassifierCV
from sklearn.ensemble import (RandomForestClassifier, GradientBoostingClassifier)
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
import xgboost as xgb
import lightgbm as lgb
import shap
import joblib
import argparse
import warnings
import json
from datetime import datetime
import matplotlib.colors as mcolors
from matplotlib.patches import Patch
import matplotlib.cm as cm
from sklearn.isotonic import IsotonicRegression
from statsmodels.stats.multitest import multipletests

warnings.filterwarnings('ignore')

RANDOM_SEED = 42
RNG = np.random.RandomState(RANDOM_SEED)
sns.set_style("whitegrid")

# ==================== 统一字体设置 (全局生效，无无效参数) ====================
FONT_FAMILY = 'Times New Roman'
TITLE_FONT_SIZE = 20        # 图表主标题字号
AXIS_LABEL_FONT_SIZE = 20   # 坐标轴标签字号
TICK_FONT_SIZE = 18         # 刻度标签字号
LEGEND_FONT_SIZE = 12       # 图例字号（基础值）
ANNOTATION_FONT_SIZE = 14   # 图中额外标注字号

# 热图专用字号（如需与全局统一，可将以下变量改为与上面对应）
HEATMAP_TICK_FONT_SIZE = 18
HEATMAP_ANNOTATION_FONT_SIZE = 20
HEATMAP_LEGEND_FONT_SIZE = 16
HEATMAP_CBAR_FONT_SIZE = 16

plt.rcParams['font.family'] = FONT_FAMILY
plt.rcParams['font.size'] = AXIS_LABEL_FONT_SIZE
plt.rcParams['axes.labelsize'] = AXIS_LABEL_FONT_SIZE
plt.rcParams['axes.titlesize'] = TITLE_FONT_SIZE
plt.rcParams['figure.titlesize'] = TITLE_FONT_SIZE
plt.rcParams['xtick.labelsize'] = TICK_FONT_SIZE
plt.rcParams['ytick.labelsize'] = TICK_FONT_SIZE
plt.rcParams['legend.fontsize'] = LEGEND_FONT_SIZE
plt.rcParams['lines.linewidth'] = 2
plt.rcParams['axes.linewidth'] = 1.5
plt.rcParams['grid.linewidth'] = 1

plt.rcParams['axes.labelweight'] = 'normal'
plt.rcParams['axes.titleweight'] = 'bold'
plt.rcParams['font.weight'] = 'normal'

plt.rcParams['mathtext.fontset'] = 'stix'
plt.rcParams['pdf.fonttype'] = 42
plt.rcParams['ps.fonttype'] = 42

# 颜色方案
VIBRANT_COLORS = [
    '#e41a1c', '#377eb8', '#4daf4a', '#984ea3', '#ff7f00',
    '#ffff33', '#a65628', '#f781bf', '#999999',
]
SET2_COLORS = [
    '#66c2a5', '#fc8d62', '#8da0cb', '#e78ac3', '#a6d854',
    '#ffd92f', '#e5c494', '#b3b3b3',
]
MEAN_CURVE_COLOR = '#0000FF'


def get_fold_colors(n_folds=10):
    return [cm.tab10(i / n_folds) for i in range(n_folds)]


# ==================== DeLong 检验函数 ====================
def compute_auc_variance_delong(y_true, y_pred):
    y_true = np.asarray(y_true).ravel()
    y_pred = np.asarray(y_pred).ravel()
    if len(np.unique(y_true)) != 2:
        raise ValueError("DeLong检验要求二分类标签")
    pos_idx = y_true == 1
    neg_idx = y_true == 0
    pos_scores = y_pred[pos_idx]
    neg_scores = y_pred[neg_idx]
    n_pos = len(pos_scores)
    n_neg = len(neg_scores)
    if n_pos == 0 or n_neg == 0:
        raise ValueError("正类或负类样本数量为0")
    auc_value = roc_auc_score(y_true, y_pred)
    pos_scores_sorted = np.sort(pos_scores)
    neg_scores_sorted = np.sort(neg_scores)
    V10 = np.zeros(n_pos)
    V01 = np.zeros(n_neg)
    for i, score in enumerate(pos_scores):
        V10[i] = np.searchsorted(neg_scores_sorted, score, side='right') / n_neg
    for j, score in enumerate(neg_scores):
        V01[j] = np.searchsorted(pos_scores_sorted, score, side='left') / n_pos
    var10 = np.var(V10, ddof=1) / n_pos if n_pos > 1 else 0
    var01 = np.var(V01, ddof=1) / n_neg if n_neg > 1 else 0
    S10 = V10 - auc_value
    S01 = V01 - auc_value
    var_auc = var10 + var01
    var_auc = max(var_auc, 1e-10)
    auc_std = np.sqrt(var_auc)
    return auc_value, var_auc, auc_std, S10, S01


def delong_test_correct(y_true, y_pred1, y_pred2):
    y_true = np.asarray(y_true).ravel()
    y_pred1 = np.asarray(y_pred1).ravel()
    y_pred2 = np.asarray(y_pred2).ravel()
    if len(np.unique(y_true)) != 2:
        raise ValueError("标签必须是二分类")
    n_pos = np.sum(y_true == 1)
    n_neg = np.sum(y_true == 0)
    if n_pos < 10 or n_neg < 10:
        print(f"  ⚠ 警告: 正类样本量={n_pos}, 负类样本量={n_neg}, DeLong检验结果可能不可靠")
    try:
        auc1, var1, std1, S10_1, S01_1 = compute_auc_variance_delong(y_true, y_pred1)
        auc2, var2, std2, S10_2, S01_2 = compute_auc_variance_delong(y_true, y_pred2)
        cov10 = np.cov(S10_1, S10_2)[0, 1] / len(S10_1) if len(S10_1) == len(S10_2) and len(S10_1) > 0 else 0
        cov01 = np.cov(S01_1, S01_2)[0, 1] / len(S01_1) if len(S01_1) == len(S01_2) and len(S01_1) > 0 else 0
        cov_auc = cov10 + cov01
        std_err = np.sqrt(max(var1 + var2 - 2 * cov_auc, 1e-10))
        if std_err < 1e-10:
            auc_diff = 0
            z_score = 0
            p_value = 1.0
        else:
            auc_diff = auc1 - auc2
            z_score = auc_diff / std_err
            p_value = 2 * (1 - norm.cdf(abs(z_score)))
        p_value = min(max(p_value, 0), 1.0)
        return {
            'auc1': auc1, 'auc2': auc2, 'auc_diff': auc_diff, 'std_err': std_err,
            'z_score': z_score, 'p_value': p_value, 'var1': var1, 'var2': var2,
            'n_pos': n_pos, 'n_neg': n_neg
        }
    except Exception as e:
        print(f"DeLong检验计算失败: {str(e)}")
        return {
            'auc1': roc_auc_score(y_true, y_pred1), 'auc2': roc_auc_score(y_true, y_pred2),
            'auc_diff': 0, 'std_err': 1.0, 'z_score': 0, 'p_value': 1.0,
            'var1': 0, 'var2': 0, 'n_pos': n_pos, 'n_neg': n_neg
        }


def get_significance_symbol(p_value):
    if pd.isna(p_value) or p_value >= 1:
        return 'ns'
    elif p_value < 0.001:
        return '***'
    elif p_value < 0.01:
        return '**'
    elif p_value < 0.05:
        return '*'
    else:
        return 'ns'


def perform_pairwise_delong_tests_with_correction(model_probas, y_true, output_dir="results", phase="Test"):
    models = list(model_probas.keys())
    n_models = len(models)
    p_matrix_raw = pd.DataFrame(np.ones((n_models, n_models)), index=models, columns=models)
    auc_values = {}
    comparisons = []
    p_values_raw = []
    comparison_pairs = []
    print(f"\n{'=' * 70}")
    print(f"Pairwise DeLong检验 - {phase}集 (所有模型两两比较)")
    print(f"{'=' * 70}")
    for model_name in models:
        auc_values[model_name] = roc_auc_score(y_true, model_probas[model_name])
        print(f"  {model_name}: AUC = {auc_values[model_name]:.4f}")
    print("\n执行DeLong检验...")
    for i, model_i in enumerate(models):
        for j, model_j in enumerate(models):
            if i < j:
                try:
                    delong_result = delong_test_correct(y_true, model_probas[model_i], model_probas[model_j])
                    p_value_raw = delong_result['p_value']
                    p_matrix_raw.loc[model_i, model_j] = p_value_raw
                    p_matrix_raw.loc[model_j, model_i] = p_value_raw
                    comparisons.append({
                        'model1': model_i, 'model2': model_j,
                        'p_value_raw': p_value_raw, 'auc_diff': delong_result['auc_diff']
                    })
                    p_values_raw.append(p_value_raw)
                    comparison_pairs.append((model_i, model_j))
                except Exception as e:
                    print(f"  {model_i} vs {model_j} DeLong检验失败: {str(e)}")
                    p_matrix_raw.loc[model_i, model_j] = np.nan
                    p_matrix_raw.loc[model_j, model_i] = np.nan
    n_comparisons = len(p_values_raw)
    if n_comparisons > 0:
        rejected, p_values_corrected, alpha_corrected, _ = multipletests(p_values_raw, alpha=0.05, method='bonferroni')
        p_matrix_corrected = p_matrix_raw.copy()
        for idx, (model_i, model_j) in enumerate(comparison_pairs):
            p_matrix_corrected.loc[model_i, model_j] = p_values_corrected[idx]
            p_matrix_corrected.loc[model_j, model_i] = p_values_corrected[idx]
        print(f"\n多重比较校正信息:")
        print(f"  总比较次数: {n_comparisons}")
        print(f"  Bonferroni校正后显著性阈值: α = 0.05/{n_comparisons} = {0.05 / n_comparisons:.6f}")
        print(f"  原始显著比较数: {sum(np.array(p_values_raw) < 0.05)}")
        print(f"  校正后显著比较数: {sum(rejected)}")
        print(f"\n详细比较结果 (Bonferroni校正):")
        for idx, (model_i, model_j) in enumerate(comparison_pairs):
            p_raw = p_values_raw[idx]
            p_corr = p_values_corrected[idx]
            sig_raw = get_significance_symbol(p_raw)
            sig_corr = get_significance_symbol(p_corr)
            status = "✓ 显著" if rejected[idx] else "✗ 不显著"
            print(f"  {model_i} vs {model_j}: AUC diff = {comparisons[idx]['auc_diff']:.4f}, "
                  f"P_raw = {p_raw:.6f} {sig_raw}, P_corrected = {p_corr:.6f} {sig_corr} [{status}]")
        os.makedirs(output_dir, exist_ok=True)
        p_matrix_raw.to_csv(os.path.join(output_dir, f"pairwise_delong_pvalues_raw_{phase}.csv"), index=True, encoding='utf-8-sig')
        p_matrix_corrected.to_csv(os.path.join(output_dir, f"pairwise_delong_pvalues_corrected_{phase}.csv"), index=True, encoding='utf-8-sig')
        correction_summary = pd.DataFrame({
            'Comparison': [f"{m1} vs {m2}" for m1, m2 in comparison_pairs],
            'Model_1': [c['model1'] for c in comparisons],
            'Model_2': [c['model2'] for c in comparisons],
            'AUC_Difference': [c['auc_diff'] for c in comparisons],
            'P_Value_Raw': p_values_raw,
            'P_Value_Bonferroni': p_values_corrected,
            'Significant_Raw': [p < 0.05 for p in p_values_raw],
            'Significant_Bonferroni': rejected.tolist()
        })
        correction_summary.to_csv(os.path.join(output_dir, f"delong_multiple_comparison_correction_{phase}.csv"), index=False, encoding='utf-8-sig')
    else:
        p_matrix_corrected = p_matrix_raw
    auc_df = pd.DataFrame(list(auc_values.items()), columns=['Model', 'AUC']).sort_values('AUC', ascending=False)
    auc_df.to_csv(os.path.join(output_dir, f"model_auc_values_{phase}.csv"), index=False, encoding='utf-8-sig')
    return p_matrix_corrected, auc_values


def perform_delong_tests_with_reference_and_correction(model_probas, y_true, reference_model_name,
                                                       output_dir="results", phase="Test"):
    models = list(model_probas.keys())
    if reference_model_name not in model_probas:
        raise ValueError(f"基准模型 '{reference_model_name}' 不存在")
    reference_proba = model_probas[reference_model_name]
    reference_auc = roc_auc_score(y_true, reference_proba)
    results = []
    p_values_raw = []
    comparisons = []
    print(f"\n{'=' * 70}")
    print(f"DeLong检验 - {phase}集 (以基准模型 [{reference_model_name}] 为参照)")
    print(f"基准模型 AUC = {reference_auc:.4f}")
    print(f"{'=' * 70}")
    for model_name in models:
        if model_name == reference_model_name:
            continue
        try:
            delong_result = delong_test_correct(y_true, reference_proba, model_probas[model_name])
            result = {
                'Phase': phase, 'Reference_Model': reference_model_name, 'Compared_Model': model_name,
                'Reference_AUC': delong_result['auc1'], 'Compared_AUC': delong_result['auc2'],
                'AUC_Difference': delong_result['auc_diff'], 'Standard_Error': delong_result['std_err'],
                'Z_Score': delong_result['z_score'], 'P_Value_Raw': delong_result['p_value'],
                'Significant_Raw': 'Yes' if delong_result['p_value'] < 0.05 else 'No',
                'Significance_Level_Raw': get_significance_symbol(delong_result['p_value'])
            }
            results.append(result)
            p_values_raw.append(delong_result['p_value'])
            comparisons.append(model_name)
            sig_symbol = result['Significance_Level_Raw']
            print(f"  {reference_model_name} vs {model_name}: AUC diff = {delong_result['auc_diff']:.4f}, "
                  f"P_raw = {delong_result['p_value']:.6f} {sig_symbol}")
        except Exception as e:
            print(f"  {reference_model_name} vs {model_name} DeLong检验失败: {str(e)}")
            results.append({
                'Phase': phase, 'Reference_Model': reference_model_name, 'Compared_Model': model_name,
                'Reference_AUC': np.nan, 'Compared_AUC': np.nan, 'AUC_Difference': np.nan,
                'Standard_Error': np.nan, 'Z_Score': np.nan, 'P_Value_Raw': np.nan,
                'Significant_Raw': 'Error', 'Significance_Level_Raw': 'Error'
            })
            p_values_raw.append(np.nan)
            comparisons.append(model_name)
    n_comparisons = len([p for p in p_values_raw if not pd.isna(p)])
    if n_comparisons > 0:
        p_values_valid = [p for p in p_values_raw if not pd.isna(p)]
        rejected, p_values_corrected, _, _ = multipletests(p_values_valid, alpha=0.05, method='bonferroni')
        corrected_idx = 0
        for i, result in enumerate(results):
            if not pd.isna(p_values_raw[i]):
                result['P_Value_Corrected'] = p_values_corrected[corrected_idx]
                result['Significant_Corrected'] = 'Yes' if rejected[corrected_idx] else 'No'
                result['Significance_Level_Corrected'] = get_significance_symbol(p_values_corrected[corrected_idx])
                corrected_idx += 1
            else:
                result['P_Value_Corrected'] = np.nan
                result['Significant_Corrected'] = 'Error'
                result['Significance_Level_Corrected'] = 'Error'
        print(f"\n多重比较校正 (Bonferroni):比较次数: {n_comparisons}, 校正后显著数: {sum(rejected)}")
        print(f"\n详细结果 (Bonferroni校正):")
        for result in results:
            if 'P_Value_Corrected' in result:
                print(f"  {reference_model_name} vs {result['Compared_Model']}: AUC diff = {result['AUC_Difference']:.4f}, "
                      f"P_raw = {result['P_Value_Raw']:.6f}, P_corr = {result['P_Value_Corrected']:.6f} {result['Significance_Level_Corrected']}")
    else:
        for result in results:
            result['P_Value_Corrected'] = result['P_Value_Raw']
            result['Significant_Corrected'] = result['Significant_Raw']
            result['Significance_Level_Corrected'] = result['Significance_Level_Raw']
    results_df = pd.DataFrame(results)
    os.makedirs(output_dir, exist_ok=True)
    results_df.to_csv(os.path.join(output_dir, f"delong_test_results_{phase}_with_reference_{reference_model_name}.csv"), index=False, encoding='utf-8-sig')
    plot_delong_comparison_bar(results_df, reference_model_name, reference_auc, output_dir, phase)
    return results_df, reference_auc


# ==================== 数据加载与预处理 ====================
def load_and_preprocess_data(filepath, target_column='target'):
    data = pd.read_csv(filepath)
    if target_column not in data.columns:
        raise ValueError(f"Dataset must contain a target column named '{target_column}'")
    X = data.drop(target_column, axis=1)
    y = data[target_column]
    feature_names = X.columns.tolist()
    class_ratio = y.value_counts(normalize=True)
    print(f"Class distribution: Negative={class_ratio.get(0, 0):.2%}, Positive={class_ratio.get(1, 0):.2%}")
    return X, y, feature_names


def load_external_test_data(filepath, feature_names, target_column='target'):
    data_external = pd.read_csv(filepath)
    if target_column not in data_external.columns:
        raise ValueError(f"External test dataset must contain a target column named '{target_column}'")
    X_external = data_external.drop(target_column, axis=1).copy()
    y_external = data_external[target_column]
    missing_features = set(feature_names) - set(X_external.columns)
    if missing_features:
        print(f"Warning: External test set is missing features: {missing_features}")
        for feature in missing_features:
            X_external[feature] = np.nan
    X_external = X_external[feature_names]
    print(f"External test set loaded: {X_external.shape[0]} samples, {X_external.shape[1]} features")
    print(f"External test class distribution: Negative={(y_external == 0).mean():.2%}, Positive={(y_external == 1).mean():.2%}")
    return X_external, y_external


# ==================== 风险分层函数（阈值0.40/0.80） ====================
def compute_risk_thresholds_from_training(y_train_proba, method='clinical_driven'):
    if method != 'clinical_driven':
        print(f"  ⚠ 注意: 指定的方法 '{method}' 已被覆盖，强制使用临床驱动法(clinical_driven)")
    low_threshold = 0.40
    high_threshold = 0.80
    print(f"\n风险分层阈值 (临床驱动法 - Clinical Driven Method) [v25更新] :")
    print(f"  Low-risk: ≤ {low_threshold:.2f} (临床低风险)")
    print(f"  Medium-risk: {low_threshold:.2f} < prob ≤ {high_threshold:.2f}")
    print(f"  High-risk: > {high_threshold:.2f} (临床高风险)")
    return {'low_threshold': low_threshold, 'high_threshold': high_threshold, 'method': 'clinical_driven'}


def add_risk_stratification_fixed(predictions_df, calibrated_prob_col='Calibrated_Probability',
                                  thresholds=None, y_true_for_validation=None,
                                  risk_labels=None, output_dir=None, dataset_name="Dataset"):
    if risk_labels is None:
        risk_labels = ['low-risk', 'medium-risk', 'high-risk']
    if calibrated_prob_col not in predictions_df.columns:
        raise ValueError(f"DataFrame does not contain column '{calibrated_prob_col}'")
    calibrated_probs = predictions_df[calibrated_prob_col].values
    if thresholds is None:
        if y_true_for_validation is None:
            raise ValueError(f"Risk stratification for {dataset_name} requires thresholds.")
        print(f"\n{'=' * 50}\nRisk Stratification for {dataset_name} (TRAINING SET - Computing Thresholds)\n{'=' * 50}")
        thresholds = compute_risk_thresholds_from_training(calibrated_probs, method='clinical_driven')
    else:
        print(f"\n{'=' * 50}\nRisk Stratification for {dataset_name} (using fixed thresholds from training)\n{'=' * 50}")
    low_threshold = thresholds.get('low_threshold', 0.40)
    high_threshold = thresholds.get('high_threshold', 0.80)
    print(f"  Low-risk cutoff: ≤ {low_threshold:.4f}")
    print(f"  Medium-risk cutoff: {low_threshold:.4f} < prob ≤ {high_threshold:.4f}")
    print(f"  High-risk cutoff: > {high_threshold:.4f}")
    risk_levels_list = []
    risk_scores = []
    for prob in calibrated_probs:
        if prob <= low_threshold:
            risk_levels_list.append(risk_labels[0])
            risk_scores.append(1)
        elif prob <= high_threshold:
            risk_levels_list.append(risk_labels[1])
            risk_scores.append(2)
        else:
            risk_levels_list.append(risk_labels[2])
            risk_scores.append(3)
    predictions_df['Risk_Level'] = risk_levels_list
    predictions_df['Risk_Score'] = risk_scores
    risk_distribution = predictions_df['Risk_Level'].value_counts()
    risk_percentages = predictions_df['Risk_Level'].value_counts(normalize=True) * 100
    print(f"\n{'=' * 50}\nRisk Distribution for {dataset_name}\n{'=' * 50}")
    for risk_level in risk_labels:
        count = risk_distribution.get(risk_level, 0)
        percentage = risk_percentages.get(risk_level, 0)
        print(f"  {risk_level}: {count} samples ({percentage:.1f}%)")
    risk_ratio = None
    low_risk_pos_rate = None
    high_risk_pos_rate = None
    if 'True_Label' in predictions_df.columns:
        print(f"\n{'=' * 50}\nRisk Distribution by True Label for {dataset_name}\n{'=' * 50}")
        for label in sorted(predictions_df['True_Label'].unique()):
            label_name = "Negative (0)" if label == 0 else "Positive (1)"
            label_data = predictions_df[predictions_df['True_Label'] == label]
            print(f"\n  {label_name} (n={len(label_data)}):")
            for risk_level in risk_labels:
                count = (label_data['Risk_Level'] == risk_level).sum()
                percentage = count / len(label_data) * 100 if len(label_data) > 0 else 0
                print(f"    {risk_level}: {count} ({percentage:.1f}%)")
        low_risk_data = predictions_df[predictions_df['Risk_Level'] == risk_labels[0]]
        high_risk_data = predictions_df[predictions_df['Risk_Level'] == risk_labels[2]]
        if len(low_risk_data) > 0 and len(high_risk_data) > 0:
            low_risk_pos_rate = low_risk_data['True_Label'].mean() * 100
            high_risk_pos_rate = high_risk_data['True_Label'].mean() * 100
            risk_ratio = high_risk_pos_rate / low_risk_pos_rate if low_risk_pos_rate > 0 else np.inf
            print(f"\n{'=' * 50}\nRisk Stratification Performance Evaluation for {dataset_name}\n{'=' * 50}")
            print(f"Positive rate in low-risk group: {low_risk_pos_rate:.1f}%")
            print(f"Positive rate in high-risk group: {high_risk_pos_rate:.1f}%")
            print(f"Risk Ratio (High vs Low): {risk_ratio:.2f}")
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
        pd.DataFrame([thresholds]).to_csv(os.path.join(output_dir, f"{dataset_name.replace(' ', '_')}_risk_stratification_thresholds.csv"), index=False, encoding='utf-8-sig')
        risk_stats = pd.DataFrame({'Risk_Level': risk_labels, 'Count': [risk_distribution.get(level, 0) for level in risk_labels],
                                   'Percentage': [risk_percentages.get(level, 0) for level in risk_labels]})
        risk_stats.to_csv(os.path.join(output_dir, f"{dataset_name.replace(' ', '_')}_risk_distribution_stats.csv"), index=False, encoding='utf-8-sig')
        if 'True_Label' in predictions_df.columns and risk_ratio is not None:
            pd.DataFrame([{'Dataset': dataset_name, 'Low_Risk_Positive_Rate_Percent': low_risk_pos_rate,
                           'High_Risk_Positive_Rate_Percent': high_risk_pos_rate, 'Risk_Ratio': risk_ratio,
                           'Low_Threshold': low_threshold, 'High_Threshold': high_threshold,
                           'Threshold_Method': thresholds.get('method', 'clinical_driven')}]) \
                .to_csv(os.path.join(output_dir, f"{dataset_name.replace(' ', '_')}_risk_ratio.csv"), index=False, encoding='utf-8-sig')
    return predictions_df, thresholds, risk_distribution, risk_ratio


def plot_risk_stratification(predictions_df, calibrated_prob_col='Calibrated_Probability',
                             risk_level_col='Risk_Level', output_dir=None, dataset_name="Test Set"):
    os.makedirs(output_dir, exist_ok=True)
    risk_colors = {'low-risk': '#2ecc71', 'medium-risk': '#f39c12', 'high-risk': '#e74c3c'}
    risk_order = ['low-risk', 'medium-risk', 'high-risk']

    # 条形图
    plt.figure(figsize=(10, 6))
    risk_counts = predictions_df[risk_level_col].value_counts().reindex(risk_order)
    colors = [risk_colors[risk] for risk in risk_counts.index]
    bars = plt.bar(risk_counts.index, risk_counts.values, color=colors, edgecolor='black', linewidth=1.5)
    plt.xlabel('Risk Level'); plt.ylabel('Number of Samples'); plt.title(f'{dataset_name} - Risk Distribution', pad=20)
    for bar, count in zip(bars, risk_counts.values):
        plt.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 5,
                 f'{count}\n({count / len(predictions_df) * 100:.1f}%)', ha='center', va='bottom')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f"{dataset_name.replace(' ', '_')}_risk_distribution_bar.JPG"), dpi=300, bbox_inches='tight')
    plt.close()

    # 饼图
    plt.figure(figsize=(8, 8))
    plt.pie(risk_counts.values, labels=risk_counts.index, autopct='%1.1f%%', colors=colors, startangle=90, explode=(0.05,0.05,0.05))
    plt.title(f'{dataset_name} - Risk Distribution (Pie Chart)', pad=20)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f"{dataset_name.replace(' ', '_')}_risk_distribution_pie.JPG"), dpi=300, bbox_inches='tight')
    plt.close()

    # 箱线图
    plt.figure(figsize=(10, 6))
    risk_data = [predictions_df[predictions_df[risk_level_col] == risk][calibrated_prob_col].values for risk in risk_order]
    bp = plt.boxplot(risk_data, labels=risk_order, patch_artist=True,
                     boxprops=dict(linewidth=2), whiskerprops=dict(linewidth=2),
                     capprops=dict(linewidth=2), medianprops=dict(linewidth=2, color='black'))
    for patch, color in zip(bp['boxes'], colors):
        patch.set_facecolor(color); patch.set_alpha(0.7)
    plt.xlabel('Risk Level'); plt.ylabel('Calibrated Probability'); plt.title(f'{dataset_name} - Calibrated Probability by Risk Level', pad=20)
    plt.grid(True, linestyle='--', alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f"{dataset_name.replace(' ', '_')}_probability_by_risk_boxplot.JPG"), dpi=300, bbox_inches='tight')
    plt.close()

    # 概率分布直方图
    plt.figure(figsize=(12, 6))
    plt.hist(predictions_df[calibrated_prob_col].values, bins=30, alpha=0.5, color='gray', edgecolor='black', label='All Samples')
    for risk_level, color in risk_colors.items():
        risk_probs = predictions_df[predictions_df[risk_level_col] == risk_level][calibrated_prob_col].values
        if len(risk_probs) > 0:
            plt.hist(risk_probs, bins=20, alpha=0.7, color=color, edgecolor='black', label=f'{risk_level}', density=False)
    plt.xlabel('Calibrated Probability'); plt.ylabel('Frequency'); plt.title(f'{dataset_name} - Calibrated Probability Distribution with Risk Thresholds', pad=20)
    plt.legend(loc='upper right'); plt.grid(True, linestyle='--', alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f"{dataset_name.replace(' ', '_')}_probability_distribution.JPG"), dpi=300, bbox_inches='tight')
    plt.close()

    if 'True_Label' in predictions_df.columns:
        # 按标签分组堆叠条形图
        label_0_data = predictions_df[predictions_df['True_Label'] == 0]
        label_1_data = predictions_df[predictions_df['True_Label'] == 1]
        label_0_counts = [label_0_data[label_0_data[risk_level_col] == risk].shape[0] for risk in risk_order]
        label_1_counts = [label_1_data[label_1_data[risk_level_col] == risk].shape[0] for risk in risk_order]
        x = np.arange(len(risk_order)); width = 0.35
        fig, ax = plt.subplots(figsize=(10, 6))
        bars1 = ax.bar(x - width/2, label_0_counts, width, label='Negative (0)', color='#3498db', edgecolor='black')
        bars2 = ax.bar(x + width/2, label_1_counts, width, label='Positive (1)', color='#e74c3c', edgecolor='black')
        ax.set_xlabel('Risk Level'); ax.set_ylabel('Number of Samples'); ax.set_title(f'{dataset_name} - Risk Level Distribution by True Label', pad=20)
        ax.set_xticks(x); ax.set_xticklabels(risk_order); ax.legend(); ax.grid(True, linestyle='--', alpha=0.3, axis='y')
        for bars in [bars1, bars2]:
            for bar in bars:
                height = bar.get_height()
                if height > 0:
                    ax.text(bar.get_x() + bar.get_width()/2., height + 0.5, f'{int(height)}', ha='center', va='bottom')
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, f"{dataset_name.replace(' ', '_')}_risk_by_label_stacked_bar.JPG"), dpi=300, bbox_inches='tight')
        plt.close()

        # 阳性率条形图
        positive_rates = []
        for risk in risk_order:
            risk_data = predictions_df[predictions_df[risk_level_col] == risk]
            pos_rate = risk_data['True_Label'].mean() * 100 if len(risk_data) > 0 else 0
            positive_rates.append(pos_rate)
        low_risk_rate = positive_rates[0]; high_risk_rate = positive_rates[2]
        risk_ratio = high_risk_rate / low_risk_rate if low_risk_rate > 0 else np.inf
        plt.figure(figsize=(10, 6))
        bars = plt.bar(risk_order, positive_rates, color=[risk_colors[r] for r in risk_order], edgecolor='black', linewidth=1.5)
        for bar, rate in zip(bars, positive_rates):
            plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1, f'{rate:.1f}%', ha='center', va='bottom', fontweight='bold')
        plt.text(0.5, 0.5, f'Risk Ratio (High vs Low): {risk_ratio:.2f}', transform=plt.gca().transAxes,
                 ha='center', va='center', fontweight='bold', bbox=dict(boxstyle='round,pad=0.5', facecolor='white', alpha=0.85))
        plt.xlabel('Risk Level'); plt.ylabel('Positive Rate (%)'); plt.title(f'{dataset_name} - Positive Rate by Risk Level', pad=20)
        plt.ylim(0, max(positive_rates) * 1.15 if max(positive_rates) > 0 else 100)
        plt.grid(True, linestyle='--', alpha=0.3, axis='y')
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, f"{dataset_name.replace(' ', '_')}_positive_rate_by_risk.JPG"), dpi=300, bbox_inches='tight')
        plt.close()

        # 散点图
        plt.figure(figsize=(12, 8))
        jitter = np.random.normal(0, 0.08, len(predictions_df))
        x_jitter = predictions_df['Risk_Score'] + jitter
        colors_scatter = ['#3498db' if label == 0 else '#e74c3c' for label in predictions_df['True_Label']]
        plt.scatter(x_jitter, predictions_df[calibrated_prob_col], c=colors_scatter, alpha=0.6, s=60, edgecolors='black', linewidth=0.5)
        low_threshold_plot = 0.40; high_threshold_plot = 0.80
        plt.axhline(y=low_threshold_plot, color='green', linestyle='--', linewidth=2.5, alpha=0.8)
        plt.axhline(y=high_threshold_plot, color='orange', linestyle='--', linewidth=2.5, alpha=0.8)
        plt.xticks([1,2,3], ['Low Risk', 'Medium Risk', 'High Risk'])
        plt.xlabel('Risk Level'); plt.ylabel('Calibrated Probability'); plt.title(f'Distribution of Calibrated Probabilities by Risk Level - {dataset_name}', pad=20)
        plt.grid(True, linestyle='--', alpha=0.3)
        from matplotlib.lines import Line2D
        legend_elements = [Patch(facecolor='#3498db', label='Negative (0)', alpha=0.6),
                           Patch(facecolor='#e74c3c', label='Positive (1)', alpha=0.6),
                           Line2D([0],[0], color='green', linestyle='--', linewidth=2.5, label=f'Low/Medium Risk Threshold: {low_threshold_plot:.2f}'),
                           Line2D([0],[0], color='orange', linestyle='--', linewidth=2.5, label=f'Medium/High Risk Threshold: {high_threshold_plot:.2f}')]
        plt.legend(handles=legend_elements, frameon=True, framealpha=0.9, loc='upper left')
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, f"{dataset_name.replace(' ', '_')}_risk_score_scatter.JPG"), dpi=300, bbox_inches='tight')
        plt.close()

    print(f"\nRisk stratification plots saved to: {output_dir}")


# ==================== 正态性检验与基线特征表 ====================
def perform_normality_test(data, feature_name, alpha=0.05):
    data_clean = data[feature_name].dropna()
    if len(data_clean) < 3:
        return False, "样本量不足(<3)"
    try:
        stat, p_value = stats.shapiro(data_clean)
        is_normal = p_value > alpha
        skewness = stats.skew(data_clean)
        kurtosis = stats.kurtosis(data_clean)
        if abs(skewness) > 1 or abs(kurtosis) > 1:
            is_normal = False
        return is_normal, p_value
    except Exception as e:
        print(f"正态性检验出错 ({feature_name}): {str(e)}")
        return False, "检验失败"


def save_split_datasets(X_train, y_train, X_test, y_test, feature_names, target_column='target', output_dir="results"):
    os.makedirs(output_dir, exist_ok=True)
    train_data = X_train.copy(); train_data[target_column] = y_train; train_data['数据分区'] = '训练集'
    test_data = X_test.copy(); test_data[target_column] = y_test; test_data['数据分区'] = '测试集'
    full_split_data = pd.concat([train_data, test_data], axis=0, ignore_index=True)
    full_split_data.to_csv(os.path.join(output_dir, "数据集分割详情.csv"), index=False, encoding='utf-8-sig')
    train_path = os.path.join(output_dir, "训练集数据.csv")
    test_path = os.path.join(output_dir, "测试集数据.csv")
    train_data.to_csv(train_path, index=False, encoding='utf-8-sig')
    test_data.to_csv(test_path, index=False, encoding='utf-8-sig')
    print(f"训练集数据已保存至: {train_path} (样本量: {len(train_data)})")
    print(f"测试集数据已保存至: {test_path} (样本量: {len(test_data)})")
    generate_baseline_summary_with_normality_test(train_data, test_data, feature_names, target_column, output_dir)
    return train_path, test_path


def generate_baseline_summary_with_normality_test(train_data, test_data, feature_names, target_column, output_dir):
    summary_dir = os.path.join(output_dir, "基线特征统计")
    os.makedirs(summary_dir, exist_ok=True)
    summary_results = []
    normality_results = []
    for feature in feature_names + [target_column]:
        if feature in train_data.columns:
            train_vals = train_data[feature]
            test_vals = test_data[feature]
            if pd.api.types.is_numeric_dtype(train_vals):
                train_normal, train_p = perform_normality_test(train_data, feature)
                test_normal, test_p = perform_normality_test(test_data, feature)
                is_both_normal = train_normal and test_normal
                normality_results.append({
                    '特征': feature,
                    '训练集_正态性': '是' if train_normal else '否',
                    '训练集_Shapiro_p值': train_p if isinstance(train_p, (int, float)) else train_p,
                    '测试集_正态性': '是' if test_normal else '否',
                    '测试集_Shapiro_p值': test_p if isinstance(test_p, (int, float)) else test_p,
                    '统计方法': '参数检验(t检验)' if is_both_normal else '非参数检验(秩和检验)'
                })
                if is_both_normal:
                    train_stats = {
                        '特征': feature, '数据类型': '连续型(正态)', '统计方法': 't检验',
                        '训练集_均值±标准差': f"{train_vals.mean():.2f} ± {train_vals.std():.2f}",
                        '训练集_中位数(IQR)': f"{train_vals.median():.2f} ({train_vals.quantile(0.25):.2f}-{train_vals.quantile(0.75):.2f})",
                        '训练集_范围': f"{train_vals.min():.2f} - {train_vals.max():.2f}",
                        '测试集_均值±标准差': f"{test_vals.mean():.2f} ± {test_vals.std():.2f}",
                        '测试集_中位数(IQR)': f"{test_vals.median():.2f} ({test_vals.quantile(0.25):.2f}-{test_vals.quantile(0.75):.2f})",
                        '测试集_范围': f"{test_vals.min():.2f} - {test_vals.max():.2f}",
                    }
                    if feature != target_column:
                        try:
                            t_stat, p_value = stats.ttest_ind(train_vals.dropna(), test_vals.dropna())
                            train_stats['统计量'] = f"t={t_stat:.3f}"
                            train_stats['P值'] = f"{p_value:.4f}"
                            n1, n2 = len(train_vals.dropna()), len(test_vals.dropna())
                            s_pooled = np.sqrt(((n1-1)*train_vals.std()**2 + (n2-1)*test_vals.std()**2)/(n1+n2-2))
                            cohens_d = (train_vals.mean() - test_vals.mean()) / s_pooled
                            train_stats["效应量(Cohen's d)"] = f"{cohens_d:.3f}"
                        except Exception as e:
                            train_stats['P值'] = f"检验失败: {str(e)}"
                            train_stats["效应量(Cohen's d)"] = "NA"
                else:
                    train_stats = {
                        '特征': feature, '数据类型': '连续型(非正态)', '统计方法': 'Mann-Whitney U检验',
                        '训练集_均值±标准差': f"{train_vals.mean():.2f} ± {train_vals.std():.2f}",
                        '训练集_中位数(IQR)': f"{train_vals.median():.2f} ({train_vals.quantile(0.25):.2f}-{train_vals.quantile(0.75):.2f})",
                        '训练集_范围': f"{train_vals.min():.2f} - {train_vals.max():.2f}",
                        '测试集_均值±标准差': f"{test_vals.mean():.2f} ± {test_vals.std():.2f}",
                        '测试集_中位数(IQR)': f"{test_vals.median():.2f} ({test_vals.quantile(0.25):.2f}-{test_vals.quantile(0.75):.2f})",
                        '测试集_范围': f"{test_vals.min():.2f} - {test_vals.max():.2f}",
                    }
                    if feature != target_column:
                        try:
                            u_stat, p_value = stats.mannwhitneyu(train_vals.dropna(), test_vals.dropna())
                            train_stats['统计量'] = f"U={u_stat:.1f}"
                            train_stats['P值'] = f"{p_value:.4f}"
                            n1, n2 = len(train_vals.dropna()), len(test_vals.dropna())
                            rbc = 1 - (2*u_stat)/(n1*n2)
                            train_stats["效应量(Rank-biserial)"] = f"{rbc:.3f}"
                        except Exception as e:
                            train_stats['P值'] = f"检验失败: {str(e)}"
                            train_stats["效应量(Rank-biserial)"] = "NA"
                summary_results.append(train_stats)
            else:
                train_counts = train_vals.value_counts().sort_index()
                test_counts = test_vals.value_counts().sort_index()
                for i, (category, train_count) in enumerate(train_counts.items()):
                    test_count = test_counts.get(category, 0)
                    train_pct = train_count / len(train_vals) * 100
                    test_pct = test_count / len(test_vals) * 100
                    category_stats = {
                        '特征': feature if i==0 else f"{feature}_{category}",
                        '数据类型': '分类变量', '统计方法': '卡方检验', '类别': str(category),
                        '训练集_n(%)': f"{train_count} ({train_pct:.1f}%)",
                        '测试集_n(%)': f"{test_count} ({test_pct:.1f}%)",
                    }
                    if i == 0 and feature != target_column:
                        try:
                            contingency_table = pd.DataFrame({'训练集': train_counts, '测试集': test_counts}).fillna(0)
                            chi2, p_value, _, _ = stats.chi2_contingency(contingency_table)
                            category_stats['统计量'] = f"χ²={chi2:.3f}"
                            category_stats['P值'] = f"{p_value:.4f}"
                            n = len(train_vals) + len(test_vals)
                            min_dim = min(contingency_table.shape) - 1
                            cramers_v = np.sqrt(chi2/(n*min_dim))
                            category_stats["效应量(Cramer's V)"] = f"{cramers_v:.3f}"
                        except Exception as e:
                            category_stats['P值'] = f"检验失败: {str(e)}"
                            category_stats["效应量(Cramer's V)"] = "NA"
                    summary_results.append(category_stats)
    normality_df = pd.DataFrame(normality_results)
    normality_df.to_csv(os.path.join(summary_dir, "正态性检验结果.csv"), index=False, encoding='utf-8-sig')
    summary_df = pd.DataFrame(summary_results)
    summary_df.to_csv(os.path.join(summary_dir, "基线特征统计表.csv"), index=False, encoding='utf-8-sig')
    generate_publication_table_with_normality(summary_df, train_data, test_data, feature_names, target_column, summary_dir, normality_df)
    print(f"基线特征统计表已保存至: {os.path.join(summary_dir, '基线特征统计表.csv')}")


def generate_publication_table_with_normality(summary_df, train_data, test_data, feature_names, target_column, output_dir, normality_df):
    pub_table_data = []
    pub_table_data.append(["表1. 开发队列基线特征分布"])
    pub_table_data.append(["", "", f"训练集 (n={len(train_data)})", f"测试集 (n={len(test_data)})", "统计方法", "P值"])
    pub_table_data.append(["特征", "", "", "", "", ""])
    for feature in feature_names:
        if pd.api.types.is_numeric_dtype(train_data[feature]):
            norm_info = normality_df[normality_df['特征'] == feature]
            is_normal = not norm_info.empty and norm_info['统计方法'].values[0] == '参数检验(t检验)'
            if is_normal:
                train_desc = f"{train_data[feature].mean():.2f} ± {train_data[feature].std():.2f}"
                test_desc = f"{test_data[feature].mean():.2f} ± {test_data[feature].std():.2f}"
                stat_method = "t检验"
            else:
                train_median = train_data[feature].median()
                train_q1, train_q3 = train_data[feature].quantile(0.25), train_data[feature].quantile(0.75)
                test_median = test_data[feature].median()
                test_q1, test_q3 = test_data[feature].quantile(0.25), test_data[feature].quantile(0.75)
                train_desc = f"{train_median:.2f} ({train_q1:.2f}-{train_q3:.2f})"
                test_desc = f"{test_median:.2f} ({test_q1:.2f}-{test_q3:.2f})"
                stat_method = "秩和检验"
            p_row = summary_df[summary_df['特征'] == feature]
            p_value = p_row['P值'].values[0] if not p_row.empty else "NA"
            pub_table_data.append([feature, "", train_desc, test_desc, stat_method, p_value])
        else:
            train_counts = train_data[feature].value_counts().sort_index()
            test_counts = test_data[feature].value_counts().sort_index()
            for i, (cat, train_cnt) in enumerate(train_counts.items()):
                test_cnt = test_counts.get(cat, 0)
                train_pct = train_cnt / len(train_data) * 100
                test_pct = test_cnt / len(test_data) * 100
                if i == 0:
                    p_row = summary_df[summary_df['特征'] == feature]
                    p_value = p_row['P值'].values[0] if not p_row.empty else "NA"
                    stat_method = "卡方检验"
                else:
                    p_value = ""; stat_method = ""
                pub_table_data.append([feature if i==0 else "", str(cat), f"{train_cnt} ({train_pct:.1f}%)", f"{test_cnt} ({test_pct:.1f}%)", stat_method, p_value])
    pub_table_data.append(["", "", "", "", "", ""])
    pub_table_data.append(["目标变量", "", "", "", "", ""])
    for label in [0,1]:
        train_cnt = (train_data[target_column] == label).sum()
        test_cnt = (test_data[target_column] == label).sum()
        train_pct = train_cnt / len(train_data) * 100
        test_pct = test_cnt / len(test_data) * 100
        label_name = "阴性" if label == 0 else "阳性"
        contingency = pd.DataFrame({'训练集': [train_cnt, len(train_data)-train_cnt], '测试集': [test_cnt, len(test_data)-test_cnt]}, index=[label, 1-label])
        try:
            chi2, p, _, _ = stats.chi2_contingency(contingency)
            p_str = f"{p:.4f}"
            stat_method = "卡方检验"
        except:
            p_str = "NA"; stat_method = "卡方检验"
        var_name = target_column if label == 0 else ""
        pub_table_data.append([var_name, label_name, f"{train_cnt} ({train_pct:.1f}%)", f"{test_cnt} ({test_pct:.1f}%)", stat_method, p_str if label==0 else ""])
    pub_df = pd.DataFrame(pub_table_data)
    pub_csv_path = os.path.join(output_dir, "论文用基线特征表.csv")
    pub_excel_path = os.path.join(output_dir, "论文用基线特征表.xlsx")
    pub_df.to_csv(pub_csv_path, index=False, header=False, encoding='utf-8-sig')
    try:
        with pd.ExcelWriter(pub_excel_path, engine='openpyxl') as writer:
            pub_df.to_excel(writer, index=False, header=False, sheet_name='基线特征')
            workbook = writer.book
            worksheet = writer.sheets['基线特征']
            for column in worksheet.columns:
                max_len = 0
                col_letter = column[0].column_letter
                for cell in column:
                    try:
                        if len(str(cell.value)) > max_len:
                            max_len = len(str(cell.value))
                    except:
                        pass
                worksheet.column_dimensions[col_letter].width = min(max_len+2, 50)
    except Exception as e:
        print(f"Excel文件保存失败，但CSV文件已保存: {str(e)}")
    generate_markdown_table_with_normality(pub_table_data, output_dir)


def generate_markdown_table_with_normality(table_data, output_dir):
    md_lines = []
    for i, row in enumerate(table_data):
        if i == 0:
            md_lines.append(f"**{row[0]}**")
            md_lines.append("")
        elif i == 1:
            md_lines.append("| " + " | ".join(row) + " |")
            md_lines.append("|" + "|".join(["---"]*len(row)) + "|")
        else:
            md_lines.append("| " + " | ".join(row) + " |")
    with open(os.path.join(output_dir, "论文用基线特征表.md"), 'w', encoding='utf-8') as f:
        f.write("\n".join(md_lines))


def generate_external_baseline_comparison(train_data, test_data, external_data, feature_names, target_column, output_dir):
    os.makedirs(output_dir, exist_ok=True)
    print("\n[统计分析] 生成开发队列与外部测试集基线特征对比...")
    development = pd.concat([train_data, test_data], ignore_index=True)
    development['队列'] = '开发队列'
    external = external_data.copy()
    external['队列'] = '外部验证队列'
    comparison_results = []
    normality_results = []
    for feature in feature_names + [target_column]:
        if feature in development.columns and feature in external.columns:
            dev_vals = development[feature]
            ext_vals = external[feature]
            if pd.api.types.is_numeric_dtype(dev_vals):
                dev_norm, dev_p = perform_normality_test(development, feature)
                ext_norm, ext_p = perform_normality_test(external, feature)
                is_both_norm = dev_norm and ext_norm
                normality_results.append({
                    '特征': feature,
                    '开发队列_正态性': '是' if dev_norm else '否',
                    '开发队列_Shapiro_p值': dev_p if isinstance(dev_p, (int,float)) else dev_p,
                    '外部队列_正态性': '是' if ext_norm else '否',
                    '外部队列_Shapiro_p值': ext_p if isinstance(ext_p, (int,float)) else ext_p,
                    '统计方法': '参数检验(t检验)' if is_both_norm else '非参数检验(秩和检验)'
                })
                if is_both_norm:
                    dev_desc = f"{dev_vals.mean():.2f} ± {dev_vals.std():.2f}"
                    ext_desc = f"{ext_vals.mean():.2f} ± {ext_vals.std():.2f}"
                    stat_method = "t检验"
                    try:
                        t_stat, p_val = stats.ttest_ind(dev_vals.dropna(), ext_vals.dropna())
                        stat_value = f"t={t_stat:.3f}"
                        n1, n2 = len(dev_vals.dropna()), len(ext_vals.dropna())
                        s_pooled = np.sqrt(((n1-1)*dev_vals.std()**2 + (n2-1)*ext_vals.std()**2)/(n1+n2-2))
                        cohens_d = (dev_vals.mean() - ext_vals.mean()) / s_pooled
                        effect = f"d={cohens_d:.3f}"
                    except:
                        p_val = np.nan; stat_value = "检验失败"; effect = "NA"
                else:
                    dev_med, dev_q1, dev_q3 = dev_vals.median(), dev_vals.quantile(0.25), dev_vals.quantile(0.75)
                    ext_med, ext_q1, ext_q3 = ext_vals.median(), ext_vals.quantile(0.25), ext_vals.quantile(0.75)
                    dev_desc = f"{dev_med:.2f} ({dev_q1:.2f}-{dev_q3:.2f})"
                    ext_desc = f"{ext_med:.2f} ({ext_q1:.2f}-{ext_q3:.2f})"
                    stat_method = "秩和检验"
                    try:
                        u_stat, p_val = stats.mannwhitneyu(dev_vals.dropna(), ext_vals.dropna())
                        stat_value = f"U={u_stat:.1f}"
                        n1, n2 = len(dev_vals.dropna()), len(ext_vals.dropna())
                        rbc = 1 - (2*u_stat)/(n1*n2)
                        effect = f"r={rbc:.3f}"
                    except:
                        p_val = np.nan; stat_value = "检验失败"; effect = "NA"
                comparison_results.append({
                    '特征': feature, '数据类型': '连续型',
                    f'开发队列(n={len(development)})': dev_desc,
                    f'外部验证队列(n={len(external)})': ext_desc,
                    '统计方法': stat_method, '统计量': stat_value,
                    'P值': f"{p_val:.4f}" if not np.isnan(p_val) else "NA", '效应量': effect
                })
            else:
                dev_counts = dev_vals.value_counts().sort_index()
                ext_counts = ext_vals.value_counts().sort_index()
                for i, (cat, dev_cnt) in enumerate(dev_counts.items()):
                    ext_cnt = ext_counts.get(cat, 0)
                    dev_pct = dev_cnt / len(dev_vals) * 100
                    ext_pct = ext_cnt / len(ext_vals) * 100
                    if i == 0:
                        try:
                            cont = pd.DataFrame({'开发队列': dev_counts, '外部验证队列': ext_counts}).fillna(0)
                            chi2, p_val, _, _ = stats.chi2_contingency(cont)
                            stat_value = f"χ²={chi2:.3f}"
                            n = len(dev_vals)+len(ext_vals)
                            min_dim = min(cont.shape)-1
                            cramers_v = np.sqrt(chi2/(n*min_dim))
                            effect = f"V={cramers_v:.3f}"
                        except:
                            p_val = np.nan; stat_value = "检验失败"; effect = "NA"
                    else:
                        stat_value = ""; effect = ""; p_val = np.nan
                    comparison_results.append({
                        '特征': feature if i==0 else '',
                        '数据类型': '分类变量', '类别': str(cat),
                        f'开发队列(n={len(development)})': f"{dev_cnt} ({dev_pct:.1f}%)",
                        f'外部验证队列(n={len(external)})': f"{ext_cnt} ({ext_pct:.1f}%)",
                        '统计方法': "卡方检验" if i==0 else "",
                        '统计量': stat_value if i==0 else "",
                        'P值': f"{p_val:.4f}" if i==0 and not np.isnan(p_val) else "",
                        '效应量': effect if i==0 else ""
                    })
    pd.DataFrame(normality_results).to_csv(os.path.join(output_dir, "外部验证_正态性检验结果.csv"), index=False, encoding='utf-8-sig')
    comp_df = pd.DataFrame(comparison_results)
    comp_df.to_csv(os.path.join(output_dir, "开发队列_外部验证队列_基线特征对比.csv"), index=False, encoding='utf-8-sig')
    generate_external_publication_table(comp_df, development, external, feature_names, target_column, output_dir)
    return comp_df


def generate_external_publication_table(comp_df, dev_data, ext_data, feature_names, target_column, output_dir):
    pub_table = []
    pub_table.append(["表2. 开发队列与外部验证队列基线特征比较"])
    pub_table.append(["", "", f"开发队列 (n={len(dev_data)})", f"外部验证队列 (n={len(ext_data)})", "统计方法", "P值"])
    pub_table.append(["特征", "", "", "", "", ""])
    for feature in feature_names:
        if pd.api.types.is_numeric_dtype(dev_data[feature]):
            rows = comp_df[comp_df['特征'] == feature]
            if not rows.empty:
                row = rows.iloc[0]
                pub_table.append([feature, "", row[f'开发队列(n={len(dev_data)})'], row[f'外部验证队列(n={len(ext_data)})'], row['统计方法'], row['P值']])
        else:
            dev_counts = dev_data[feature].value_counts().sort_index()
            ext_counts = ext_data[feature].value_counts().sort_index()
            for i, (cat, dev_cnt) in enumerate(dev_counts.items()):
                ext_cnt = ext_counts.get(cat, 0)
                dev_pct = dev_cnt / len(dev_data) * 100
                ext_pct = ext_cnt / len(ext_data) * 100
                if i == 0:
                    rows = comp_df[comp_df['特征'] == feature]
                    p_val = rows.iloc[0]['P值'] if not rows.empty else "NA"
                    stat_method = "卡方检验"
                else:
                    p_val = ""; stat_method = ""
                pub_table.append([feature if i==0 else "", str(cat), f"{dev_cnt} ({dev_pct:.1f}%)", f"{ext_cnt} ({ext_pct:.1f}%)", stat_method, p_val])
    pub_table.append(["", "", "", "", "", ""])
    pub_table.append(["目标变量", "", "", "", "", ""])
    for label in [0,1]:
        dev_cnt = (dev_data[target_column] == label).sum()
        ext_cnt = (ext_data[target_column] == label).sum()
        dev_pct = dev_cnt / len(dev_data) * 100
        ext_pct = ext_cnt / len(ext_data) * 100
        label_name = "阴性" if label == 0 else "阳性"
        cont = pd.DataFrame({'开发队列': [dev_cnt, len(dev_data)-dev_cnt], '外部验证队列': [ext_cnt, len(ext_data)-ext_cnt]}, index=[label, 1-label])
        try:
            chi2, p_val, _, _ = stats.chi2_contingency(cont)
            p_str = f"{p_val:.4f}"
            stat_method = "卡方检验"
        except:
            p_str = "NA"; stat_method = "卡方检验"
        var_name = target_column if label == 0 else ""
        pub_table.append([var_name, label_name, f"{dev_cnt} ({dev_pct:.1f}%)", f"{ext_cnt} ({ext_pct:.1f}%)", stat_method, p_str if label==0 else ""])
    pub_df = pd.DataFrame(pub_table)
    pub_df.to_csv(os.path.join(output_dir, "论文用_外部验证基线特征对比表.csv"), index=False, header=False, encoding='utf-8-sig')
    try:
        with pd.ExcelWriter(os.path.join(output_dir, "论文用_外部验证基线特征对比表.xlsx"), engine='openpyxl') as writer:
            pub_df.to_excel(writer, index=False, header=False, sheet_name='外部验证基线特征')
            workbook = writer.book
            worksheet = writer.sheets['外部验证基线特征']
            for col in worksheet.columns:
                max_len = 0
                col_letter = col[0].column_letter
                for cell in col:
                    try:
                        if len(str(cell.value)) > max_len:
                            max_len = len(str(cell.value))
                    except:
                        pass
                worksheet.column_dimensions[col_letter].width = min(max_len+2, 50)
    except Exception as e:
        print(f"Excel文件保存失败: {str(e)}")
    md_lines = []
    for i, row in enumerate(pub_table):
        if i == 0:
            md_lines.append(f"**{row[0]}**\n")
        elif i == 1:
            md_lines.append("| " + " | ".join(row) + " |")
            md_lines.append("|" + "|".join(["---"]*len(row)) + "|")
        else:
            md_lines.append("| " + " | ".join(row) + " |")
    with open(os.path.join(output_dir, "论文用_外部验证基线特征对比表.md"), 'w', encoding='utf-8') as f:
        f.write("\n".join(md_lines))


# ==================== 模型配置 ====================
def get_model_configurations():
    models = {
        'XGBoost': {
            'model': xgb.XGBClassifier(use_label_encoder=False, eval_metric='logloss', random_state=RANDOM_SEED, scale_pos_weight=1),
            'params': {
                'model__n_estimators': [100,200], 'model__learning_rate': [0.01,0.1], 'model__max_depth': [3,5],
                'model__subsample': [0.8,1.0], 'model__scale_pos_weight': [1,2,3,5]
            }
        },
        'LightGBM': {
            'model': lgb.LGBMClassifier(random_state=RANDOM_SEED, verbose=-1, class_weight='balanced'),
            'params': {
                'model__num_leaves': [31,50], 'model__max_depth': [-1,5], 'model__learning_rate': [0.05,0.1],
                'model__n_estimators': [100,200], 'model__class_weight': ['balanced',None]
            }
        },
        'RandomForest': {
            'model': RandomForestClassifier(class_weight='balanced', random_state=RANDOM_SEED),
            'params': {
                'model__n_estimators': [100,200], 'model__max_depth': [10,None], 'model__min_samples_split': [2,5],
                'model__class_weight': ['balanced','balanced_subsample']
            }
        },
        'LogisticRegression': {
            'model': LogisticRegression(max_iter=1000, class_weight='balanced', random_state=RANDOM_SEED),
            'params': {'model__C': [0.1,1.0,10.0], 'model__penalty': ['l2'], 'model__solver': ['lbfgs','saga'], 'model__class_weight': ['balanced',None]}
        },
        'ANN': {
            'model': MLPClassifier(max_iter=2000, early_stopping=True, random_state=RANDOM_SEED),
            'params': {
                'model__hidden_layer_sizes': [(64,32),(128,64),(256,128)], 'model__activation': ['relu','tanh'],
                'model__alpha': [0.0001,0.001,0.01], 'model__learning_rate': ['constant','adaptive'],
                'model__learning_rate_init': [0.001,0.01]
            }
        },
        'MLP': {
            'model': MLPClassifier(max_iter=1000, early_stopping=True, random_state=RANDOM_SEED),
            'params': {'model__hidden_layer_sizes': [(50,),(100,)], 'model__alpha': [0.0001,0.001], 'model__activation': ['relu']}
        },
        'SVM': {
            'model': SVC(probability=True, class_weight='balanced', random_state=RANDOM_SEED),
            'params': {'model__C': [0.1,1.0], 'model__kernel': ['rbf','linear'], 'model__gamma': ['scale','auto'], 'model__class_weight': ['balanced',None]}
        },
        'KNN': {
            'model': KNeighborsClassifier(),
            'params': {'model__n_neighbors': [3,5,7], 'model__weights': ['uniform','distance']}
        },
        'GBDT': {
            'model': GradientBoostingClassifier(random_state=RANDOM_SEED),
            'params': {'model__n_estimators': [100,200], 'model__learning_rate': [0.01,0.1], 'model__max_depth': [3,5], 'model__subsample': [0.8,1.0]}
        }
    }
    return models


# ==================== 基础绘图函数（全局字体，无局部覆盖）====================
def plot_roc_curve(fpr, tpr, roc_auc, model_name, phase="Test", save_path=None):
    plt.figure(figsize=(10,8))
    plt.plot(fpr, tpr, color=VIBRANT_COLORS[0], lw=3, alpha=0.9, label=f'ROC curve (AUC = {roc_auc:.3f})')
    plt.fill_between(fpr, tpr, color=VIBRANT_COLORS[0], alpha=0.3)
    plt.plot([0,1],[0,1], color='navy', lw=2, linestyle='--', alpha=0.8)
    plt.xlim([0.0,1.0]); plt.ylim([0.0,1.05])
    plt.xlabel('False Positive Rate (1 - Specificity)')
    plt.ylabel('True Positive Rate (Sensitivity)')
    plt.title(f'{model_name} - {phase} ROC Curve', pad=20)
    plt.legend(loc="lower right")
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
    else:
        plt.show()


def plot_confusion_matrix(y_true, y_pred, model_name, phase="Test", save_path=None):
    cm = confusion_matrix(y_true, y_pred)
    labels = ['True Negative', 'False Positive', 'False Negative', 'True Positive']
    counts = [f"{v}" for v in cm.flatten()]
    percents = [f"{v/sum(cm.flatten()):.1%}" for v in cm.flatten()]
    annotations = np.asarray([f"{l}\n{c}\n({p})" for l,c,p in zip(labels,counts,percents)]).reshape(2,2)
    plt.figure(figsize=(10,8))
    ax = sns.heatmap(cm, annot=False, fmt="d", cmap='Blues', cbar=False, linewidths=2, linecolor='black')
    norm = mcolors.Normalize(vmin=cm.min(), vmax=cm.max())
    cmap = plt.cm.Blues
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            cell_color = cmap(norm(cm[i,j]))
            brightness = 0.299*cell_color[0] + 0.587*cell_color[1] + 0.114*cell_color[2]
            text_color = 'white' if brightness < 0.5 else 'black'
            plt.text(j+0.5, i+0.5, annotations[i,j], ha='center', va='center', fontsize=ANNOTATION_FONT_SIZE+2, fontweight='bold', color=text_color)
    plt.xlabel('Predicted Label'); plt.ylabel('True Label')
    plt.title(f'{model_name} - {phase} Confusion Matrix', pad=20)
    ax.set_xticklabels(['Negative','Positive'], fontweight='bold')
    ax.set_yticklabels(['Negative','Positive'], rotation=0, fontweight='bold')
    for spine in ax.spines.values():
        spine.set_visible(True); spine.set_linewidth(2); spine.set_color('black')
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
    else:
        plt.show()


def plot_pr_curve(precision, recall, average_precision, model_name, phase="Test", save_path=None):
    plt.figure(figsize=(10,8))
    plt.step(recall, precision, color=VIBRANT_COLORS[1], alpha=0.9, where='post', lw=3, label=f'PR curve (AP = {average_precision:.3f})')
    plt.fill_between(recall, precision, color=VIBRANT_COLORS[1], alpha=0.2)
    plt.xlabel('Recall (Sensitivity)'); plt.ylabel('Precision')
    plt.ylim([0.0,1.05]); plt.xlim([0.0,1.0])
    plt.title(f'{model_name} - {phase} Precision-Recall Curve', pad=20)
    plt.legend(loc="lower right")
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
    else:
        plt.show()


def plot_cv_roc_curve_enhanced(mean_fpr, tprs, aucs, mean_tpr, mean_auc, std_auc, model_name, save_path=None):
    plt.figure(figsize=(13,11))
    std_tpr = np.std(tprs, axis=0)
    tprs_upper = np.minimum(mean_tpr + std_tpr, 1)
    tprs_lower = np.maximum(mean_tpr - std_tpr, 0)
    fold_colors = get_fold_colors(10)
    for i, tpr in enumerate(tprs):
        plt.plot(mean_fpr, tpr, lw=2.5, alpha=0.85, color=fold_colors[i%len(fold_colors)], label=f'Fold {i+1} (AUC={aucs[i]:.3f})')
    plt.plot(mean_fpr, mean_tpr, color=MEAN_CURVE_COLOR, lw=4, label=f'Mean ROC (AUC = {mean_auc:.3f} ± {std_auc:.3f})')
    plt.fill_between(mean_fpr, tprs_lower, tprs_upper, color=MEAN_CURVE_COLOR, alpha=0.2, label=r'±1 std. dev.')
    plt.plot([0,1],[0,1], linestyle='--', lw=2.5, color='#DC143C', alpha=0.7, label='Random (AUC=0.5)')
    plt.xlim([-0.02,1.02]); plt.ylim([-0.02,1.02])
    plt.xlabel('False Positive Rate (1 - Specificity)')
    plt.ylabel('True Positive Rate (Sensitivity)')
    plt.title(f'{model_name} - 10-Fold Cross-Validation ROC Curves', pad=25)
    legend = plt.legend(loc="lower right", frameon=True, fancybox=True, shadow=True, ncol=2)
    legend.get_frame().set_alpha(0.9); legend.get_frame().set_edgecolor('black'); legend.get_frame().set_linewidth(1)
    plt.grid(True, linestyle='--', alpha=0.4)
    ax = plt.gca()
    for spine in ax.spines.values():
        spine.set_linewidth(1.5); spine.set_color('black')
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
    else:
        plt.show()
    print(f"交叉验证ROC曲线已保存至: {save_path}")


def plot_cv_pr_curve_enhanced(mean_recall, precisions, average_precisions, mean_precision, mean_ap, std_ap, prevalence, model_name, save_path=None):
    plt.figure(figsize=(13,11))
    std_prec = np.std(precisions, axis=0)
    prec_upper = np.minimum(mean_precision + std_prec, 1)
    prec_lower = np.maximum(mean_precision - std_prec, 0)
    fold_colors = get_fold_colors(10)
    for i, prec in enumerate(precisions):
        plt.step(mean_recall, prec, lw=2.5, alpha=0.85, where='post', color=fold_colors[i%len(fold_colors)], label=f'Fold {i+1} (AP={average_precisions[i]:.3f})')
    plt.step(mean_recall, mean_precision, color=MEAN_CURVE_COLOR, lw=4, where='post', label=f'Mean PR (AP = {mean_ap:.3f} ± {std_ap:.3f})')
    plt.fill_between(mean_recall, prec_lower, prec_upper, color=MEAN_CURVE_COLOR, alpha=0.2, label=r'±1 std. dev.')
    plt.plot([0,1],[prevalence,prevalence], linestyle='--', lw=2.5, color='#2E7D32', alpha=0.8, label=f'Baseline (Prevalence={prevalence:.3f})')
    plt.xlabel('Recall (Sensitivity)'); plt.ylabel('Precision')
    plt.ylim([0.0,1.05]); plt.xlim([0.0,1.0])
    plt.title(f'{model_name} - 10-Fold Cross-Validation Precision-Recall Curves', pad=25)
    legend = plt.legend(loc="lower right", frameon=True, fancybox=True, shadow=True, ncol=2)
    legend.get_frame().set_alpha(0.9); legend.get_frame().set_edgecolor('black'); legend.get_frame().set_linewidth(1)
    plt.grid(True, linestyle='--', alpha=0.4)
    ax = plt.gca()
    for spine in ax.spines.values():
        spine.set_linewidth(1.5); spine.set_color('black')
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
    else:
        plt.show()
    print(f"交叉验证PR曲线已保存至: {save_path}")


def plot_cv_calibration_curve(all_y_true, all_y_proba, model_name, save_path=None, n_bins=10):
    frac_pos, mean_pred = calibration_curve(all_y_true, all_y_proba, n_bins=n_bins, strategy='quantile')
    plt.figure(figsize=(10,8))
    plt.plot(mean_pred, frac_pos, "s-", lw=3, markersize=8, color=VIBRANT_COLORS[2], alpha=0.9, label=f"{model_name}")
    plt.plot([0,1],[0,1], "k:", lw=2, label="Ideal Calibration")
    plt.xlabel("Mean Predicted Probability"); plt.ylabel("Fraction of Positives")
    plt.title(f"{model_name} - Cross-Validation Calibration Curve", pad=20)
    plt.legend(loc="lower right"); plt.grid(True, linestyle='--', alpha=0.5)
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
    else:
        plt.show()


def plot_all_cv_calibration_curves(model_data, save_path=None, n_bins=10):
    plt.figure(figsize=(12,10))
    colors = VIBRANT_COLORS[:len(model_data)]
    markers = ['o','s','^','D','v','<','>','p','*']
    for idx, (name, data) in enumerate(model_data.items()):
        frac, mean = calibration_curve(data['y_true'], data['y_proba'], n_bins=n_bins, strategy='quantile')
        plt.plot(mean, frac, marker=markers[idx%len(markers)], color=colors[idx], lw=3, markersize=8, label=name)
    plt.plot([0,1],[0,1], "k:", lw=2, label="Ideal Calibration")
    plt.xlabel("Mean Predicted Probability"); plt.ylabel("Fraction of Positives")
    plt.title("Cross-Validation Calibration Curves (All Models)", pad=20)
    plt.legend(loc="lower right", frameon=True, facecolor='white')
    plt.grid(True, linestyle='--', alpha=0.5); plt.xlim(-0.05,1.05); plt.ylim(-0.05,1.05)
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
    else:
        plt.show()


def plot_cv_decision_curve(all_y_true, all_y_proba, model_name, save_path=None):
    thresholds = np.linspace(0.01,0.99,99)
    n = len(all_y_true); prevalence = np.mean(all_y_true)
    net_benefits = []
    for pt in thresholds:
        y_pred = (all_y_proba >= pt).astype(int)
        tp = np.sum((all_y_true==1) & (y_pred==1))
        fp = np.sum((all_y_true==0) & (y_pred==1))
        nb = tp/n - fp/n * (pt/(1-pt))
        net_benefits.append(nb)
    treat_all = [prevalence - (1-prevalence)*(pt/(1-pt)) for pt in thresholds]
    treat_none = [0]*len(thresholds)
    plt.figure(figsize=(10,8))
    plt.plot(thresholds, net_benefits, label=model_name, lw=3, color=VIBRANT_COLORS[3], alpha=0.9)
    plt.plot(thresholds, treat_all, 'k--', lw=2, label="Treat All")
    plt.plot(thresholds, treat_none, 'k-.', lw=2, label="Treat None")
    plt.xlabel("Threshold Probability"); plt.ylabel("Net Benefit")
    plt.title(f"{model_name} - Cross-Validation Decision Curve", pad=20)
    plt.legend(loc='upper right'); plt.grid(True, linestyle='--', alpha=0.5)
    max_benefit = max(max(net_benefits), max(treat_all), 0.1)
    plt.ylim([-0.05, max_benefit*1.1])
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
    else:
        plt.show()


def plot_all_cv_decision_curves(model_data, save_path=None):
    plt.figure(figsize=(12,10))
    colors = VIBRANT_COLORS[:len(model_data)]
    first_data = next(iter(model_data.values()))
    prevalence = np.mean(first_data['y_true'])
    thresholds = np.linspace(0.01,0.99,99)
    for idx, (name, data) in enumerate(model_data.items()):
        n = len(data['y_true'])
        nb = []
        for pt in thresholds:
            y_pred = (data['y_proba'] >= pt).astype(int)
            tp = np.sum((data['y_true']==1) & (y_pred==1))
            fp = np.sum((data['y_true']==0) & (y_pred==1))
            nb.append(tp/n - fp/n * (pt/(1-pt)))
        plt.plot(thresholds, nb, color=colors[idx], lw=2.5, label=name)
    treat_all = [prevalence - (1-prevalence)*(pt/(1-pt)) for pt in thresholds]
    treat_none = [0]*len(thresholds)
    plt.plot(thresholds, treat_all, 'k--', lw=2, label="Treat All")
    plt.plot(thresholds, treat_none, 'k-.', lw=2, label="Treat None")
    plt.xlabel("Threshold Probability"); plt.ylabel("Net Benefit")
    plt.title("Cross-Validation Decision Curves (All Models)", pad=20)
    plt.legend(loc='upper right', frameon=True, facecolor='white')
    plt.grid(True, linestyle='--', alpha=0.5)
    all_nb = []
    for data in model_data.values():
        n = len(data['y_true'])
        nb = [np.sum((data['y_true']==1) & ((data['y_proba']>=pt).astype(int)==1))/n -
              np.sum((data['y_true']==0) & ((data['y_proba']>=pt).astype(int)==1))/n * (pt/(1-pt)) for pt in thresholds]
        all_nb.extend(nb)
    max_benefit = max(max(all_nb), max(treat_all), 0.1)
    plt.ylim([-0.05, max_benefit*1.1])
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
    else:
        plt.show()


def plot_all_roc_curves(all_roc_data, save_path=None):
    n_models = len(all_roc_data)
    n_cols = 3; n_rows = (n_models + n_cols -1)//n_cols
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(18, 6*n_rows))
    if n_rows == 1: axes = np.array([axes])
    axes = axes.flatten()
    for idx, (name, data) in enumerate(all_roc_data.items()):
        ax = axes[idx]
        mean_fpr = data['mean_fpr']; tprs = data['tprs']; aucs = data['aucs']
        mean_tpr = data['mean_tpr']; mean_auc = data['mean_auc']; std_auc = data['std_auc']
        std_tpr = np.std(tprs, axis=0)
        tprs_upper = np.minimum(mean_tpr + std_tpr, 1); tprs_lower = np.maximum(mean_tpr - std_tpr, 0)
        fold_colors = get_fold_colors(10)
        for i, tpr in enumerate(tprs):
            ax.plot(mean_fpr, tpr, lw=1.8, alpha=0.6, color=fold_colors[i%len(fold_colors)], label=f'Fold {i+1} (AUC={aucs[i]:.3f})')
        ax.plot(mean_fpr, mean_tpr, color=MEAN_CURVE_COLOR, lw=3, label=f'Mean (AUC={mean_auc:.3f} ± {std_auc:.3f})')
        ax.fill_between(mean_fpr, tprs_lower, tprs_upper, color=MEAN_CURVE_COLOR, alpha=0.2, label=r'±1 std. dev.')
        ax.plot([0,1],[0,1], linestyle='--', lw=2, color='r', alpha=0.8, label='Random (AUC=0.5)')
        ax.set_title(name); ax.set_xlabel('False Positive Rate'); ax.set_ylabel('True Positive Rate')
        ax.set_xlim(-0.05,1.05); ax.set_ylim(-0.05,1.05); ax.grid(True, linestyle='--', alpha=0.5)
        ax.legend(loc="lower right", frameon=True, fancybox=True, shadow=True, ncol=2)
    for idx in range(n_models, len(axes)): fig.delaxes(axes[idx])
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
    else:
        plt.show()
    print(f"All models ROC curves saved to: {save_path}")


def plot_all_pr_curves(all_pr_data, prevalence, save_path=None):
    n_models = len(all_pr_data)
    n_cols = 3; n_rows = (n_models + n_cols -1)//n_cols
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(18, 6*n_rows))
    if n_rows == 1: axes = np.array([axes])
    axes = axes.flatten()
    for idx, (name, data) in enumerate(all_pr_data.items()):
        ax = axes[idx]
        mean_recall = data['mean_recall']; precisions = data['precisions']; aps = data['average_precisions']
        mean_prec = data['mean_precision']; mean_ap = data['mean_ap']; std_ap = data['std_ap']
        std_prec = np.std(precisions, axis=0)
        prec_upper = np.minimum(mean_prec + std_prec, 1); prec_lower = np.maximum(mean_prec - std_prec, 0)
        fold_colors = get_fold_colors(10)
        for i, prec in enumerate(precisions):
            ax.step(mean_recall, prec, lw=1.8, alpha=0.6, where='post', color=fold_colors[i%len(fold_colors)], label=f'Fold {i+1} (AP={aps[i]:.3f})')
        ax.step(mean_recall, mean_prec, color=MEAN_CURVE_COLOR, lw=3, where='post', label=f'Mean (AP={mean_ap:.3f} ± {std_ap:.3f})')
        ax.fill_between(mean_recall, prec_lower, prec_upper, color=MEAN_CURVE_COLOR, alpha=0.2, label=r'±1 std. dev.')
        ax.plot([0,1],[prevalence,prevalence], linestyle='--', color='r', lw=2, label=f'Baseline (Prevalence={prevalence:.3f})')
        ax.set_title(name); ax.set_xlabel('Recall'); ax.set_ylabel('Precision')
        ax.set_xlim(0.0,1.0); ax.set_ylim(0.0,1.05); ax.grid(True, linestyle='--', alpha=0.5)
        ax.legend(loc="lower right", frameon=True, fancybox=True, shadow=True, ncol=2)
    for idx in range(n_models, len(axes)): fig.delaxes(axes[idx])
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
    else:
        plt.show()
    print(f"All models PR curves saved to: {save_path}")


def plot_decision_curve(y_true, y_proba, model_name, output_dir="results"):
    os.makedirs(output_dir, exist_ok=True)
    thresholds = np.linspace(0.01,0.99,99)
    n = len(y_true); prevalence = np.mean(y_true)
    net_benefits = []
    for pt in thresholds:
        y_pred = (y_proba >= pt).astype(int)
        tp = np.sum((y_true==1) & (y_pred==1))
        fp = np.sum((y_true==0) & (y_pred==1))
        nb = tp/n - fp/n * (pt/(1-pt))
        net_benefits.append(nb)
    treat_all = [prevalence - (1-prevalence)*(pt/(1-pt)) for pt in thresholds]
    treat_none = [0]*len(thresholds)
    plt.figure(figsize=(10,8))
    plt.plot(thresholds, net_benefits, label=model_name, lw=3, color=VIBRANT_COLORS[2], alpha=0.9)
    plt.plot(thresholds, treat_all, 'k--', lw=2, label="Treat All")
    plt.plot(thresholds, treat_none, 'k-.', lw=2, label="Treat None")
    plt.xlabel("Threshold Probability"); plt.ylabel("Net Benefit")
    plt.title("Decision Curve Analysis (DCA)", pad=20)
    plt.legend(loc='upper right'); plt.grid(True, linestyle='--', alpha=0.5)
    max_benefit = max(max(net_benefits), max(treat_all), 0.1)
    plt.ylim([-0.05, max_benefit*1.1])
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f"{model_name}_decision_curve.JPG"), dpi=300, bbox_inches='tight')
    plt.close()
    pos_nb = [pt for pt, nb in zip(thresholds, net_benefits) if nb > 0]
    if pos_nb:
        print(f"Model clinically useful threshold range: {min(pos_nb):.2f} - {max(pos_nb):.2f}")
    else:
        print("Model shows no net clinical benefit in given threshold range")


# ==================== 交叉验证模型训练 ====================
def train_and_optimize_models(X_train, y_train, models, output_dir="results"):
    from sklearn.base import clone as sklearn_clone
    preprocessor = Pipeline([('imputer', SimpleImputer(strategy='median')), ('scaler', StandardScaler())])
    kfold = StratifiedKFold(n_splits=10, shuffle=True, random_state=RANDOM_SEED)
    scorers = {'AUC': 'roc_auc', 'AP': 'average_precision', 'Accuracy': 'accuracy', 'Precision': 'precision',
               'Sensitivity': 'recall', 'Specificity': make_scorer(recall_score, pos_label=0), 'F1': 'f1'}
    best_models = {}
    cv_results = []
    model_data = {}
    all_roc_data = {}
    all_pr_data = {}
    optimal_parameters = {}
    os.makedirs(output_dir, exist_ok=True)
    cv_plots_dir = os.path.join(output_dir, "cross_validation_plots")
    comparison_plots_dir = os.path.join(output_dir, "model_comparison_plots")
    cv_fold_results_dir = os.path.join(output_dir, "cv_fold_results")
    params_dir = os.path.join(output_dir, "optimal_parameters")
    os.makedirs(cv_plots_dir, exist_ok=True); os.makedirs(comparison_plots_dir, exist_ok=True)
    os.makedirs(cv_fold_results_dir, exist_ok=True); os.makedirs(params_dir, exist_ok=True)

    for name, config in models.items():
        print(f"\n{'='*50}\nTraining and optimizing model: {name}\n{'='*50}")
        full_pipeline = Pipeline([('preprocessor', sklearn_clone(preprocessor)), ('model', config['model'])])
        grid_search = GridSearchCV(full_pipeline, config['params'], cv=kfold, scoring=scorers, refit='AUC', n_jobs=-1, verbose=1, return_train_score=True)
        grid_search.fit(X_train, y_train)

        best_models[name] = {'pipeline': grid_search.best_estimator_, 'params': grid_search.best_params_,
                             'cv_results': grid_search.cv_results_, 'best_index': grid_search.best_index_,
                             'original_model': config['model']}
        optimal_parameters[name] = grid_search.best_params_
        pd.DataFrame([grid_search.best_params_]).to_csv(os.path.join(params_dir, f"{name}_optimal_parameters.csv"), index=False)
        print(f"\n{name} optimal parameters:", *[f"  {k}: {v}" for k,v in grid_search.best_params_.items()], sep='\n')
        cv_metrics = { 'Model': name, 'AUC': grid_search.best_score_,
                       'AP': grid_search.cv_results_['mean_test_AP'][grid_search.best_index_],
                       'Accuracy': grid_search.cv_results_['mean_test_Accuracy'][grid_search.best_index_],
                       'Precision': grid_search.cv_results_['mean_test_Precision'][grid_search.best_index_],
                       'Sensitivity': grid_search.cv_results_['mean_test_Sensitivity'][grid_search.best_index_],
                       'Specificity': grid_search.cv_results_['mean_test_Specificity'][grid_search.best_index_],
                       'F1': grid_search.cv_results_['mean_test_F1'][grid_search.best_index_] }
        cv_results.append(cv_metrics)
        print(f"{name} best CV AUC: {grid_search.best_score_:.4f}")

        # 保存每折结果
        fold_list = []
        for fold in range(10):
            fold_list.append({'Model': name, 'Fold': fold+1,
                              'AUC': grid_search.cv_results_[f'split{fold}_test_AUC'][grid_search.best_index_],
                              'AP': grid_search.cv_results_[f'split{fold}_test_AP'][grid_search.best_index_],
                              'Accuracy': grid_search.cv_results_[f'split{fold}_test_Accuracy'][grid_search.best_index_],
                              'Precision': grid_search.cv_results_[f'split{fold}_test_Precision'][grid_search.best_index_],
                              'Sensitivity': grid_search.cv_results_[f'split{fold}_test_Sensitivity'][grid_search.best_index_],
                              'Specificity': grid_search.cv_results_[f'split{fold}_test_Specificity'][grid_search.best_index_],
                              'F1': grid_search.cv_results_[f'split{fold}_test_F1'][grid_search.best_index_] })
        pd.DataFrame(fold_list).to_csv(os.path.join(cv_fold_results_dir, f"{name}_fold_results.csv"), index=False)
        print(f"{name} 10-fold results saved.")

        # 提取最佳模型参数并克隆
        original_model = config['model']
        best_params = grid_search.best_params_
        cleaned = {k.replace('model__',''): v for k,v in best_params.items() if k.startswith('model__')}
        tuned_model = sklearn_clone(original_model)
        valid_params = {k: v for k,v in cleaned.items() if hasattr(tuned_model, k)}
        tuned_model.set_params(**valid_params)

        all_y_true = []; all_y_proba = []
        fold_tprs = []; fold_aucs = []; fold_precisions = []; fold_aps = []
        mean_fpr = np.linspace(0,1,100); mean_recall = np.linspace(0,1,100)

        for train_idx, val_idx in kfold.split(X_train, y_train):
            fold_preproc = Pipeline([('imputer', SimpleImputer(strategy='median')), ('scaler', StandardScaler())])
            fold_model = sklearn_clone(tuned_model)
            X_train_fold = X_train.iloc[train_idx]; y_train_fold = y_train.iloc[train_idx]
            X_val_fold = X_train.iloc[val_idx]; y_val_fold = y_train.iloc[val_idx]
            X_train_fold_pp = fold_preproc.fit_transform(X_train_fold)
            X_val_fold_pp = fold_preproc.transform(X_val_fold)
            fold_model.fit(X_train_fold_pp, y_train_fold)
            y_proba_fold = fold_model.predict_proba(X_val_fold_pp)[:,1]
            all_y_true.extend(y_val_fold.values); all_y_proba.extend(y_proba_fold)
            fpr, tpr, _ = roc_curve(y_val_fold, y_proba_fold)
            fold_aucs.append(auc(fpr, tpr))
            interp_tpr = np.interp(mean_fpr, fpr, tpr); interp_tpr[0]=0.0; fold_tprs.append(interp_tpr)
            precision, recall, _ = precision_recall_curve(y_val_fold, y_proba_fold)
            fold_aps.append(average_precision_score(y_val_fold, y_proba_fold))
            interp_prec = np.interp(mean_recall, recall[::-1], precision[::-1]); fold_precisions.append(interp_prec)

        all_y_true = np.array(all_y_true); all_y_proba = np.array(all_y_proba)
        model_data[name] = {'y_true': all_y_true, 'y_proba': all_y_proba}
        mean_tpr = np.mean(fold_tprs, axis=0); mean_tpr[-1]=1.0
        mean_auc = np.mean(fold_aucs); std_auc = np.std(fold_aucs)
        all_roc_data[name] = {'mean_fpr': mean_fpr, 'tprs': fold_tprs, 'aucs': fold_aucs, 'mean_tpr': mean_tpr, 'mean_auc': mean_auc, 'std_auc': std_auc}
        mean_prec = np.mean(fold_precisions, axis=0); mean_ap = np.mean(fold_aps); std_ap = np.std(fold_aps)
        all_pr_data[name] = {'mean_recall': mean_recall, 'precisions': fold_precisions, 'average_precisions': fold_aps,
                             'mean_precision': mean_prec, 'mean_ap': mean_ap, 'std_ap': std_ap}

        plot_cv_roc_curve_enhanced(mean_fpr, fold_tprs, fold_aucs, mean_tpr, mean_auc, std_auc, name, save_path=os.path.join(cv_plots_dir, f"{name}_cross_val_roc_curve.JPG"))
        plot_cv_pr_curve_enhanced(mean_recall, fold_precisions, fold_aps, mean_prec, mean_ap, std_ap, np.mean(y_train), name, save_path=os.path.join(cv_plots_dir, f"{name}_cross_val_pr_curve.JPG"))
        plot_cv_calibration_curve(all_y_true, all_y_proba, name, save_path=os.path.join(cv_plots_dir, f"{name}_cross_val_calibration_curve.JPG"))
        plot_cv_decision_curve(all_y_true, all_y_proba, name, save_path=os.path.join(cv_plots_dir, f"{name}_cross_val_decision_curve.JPG"))
        print(f"  {name} 10折交叉验证完成，平均AUC: {mean_auc:.4f} ± {std_auc:.4f}")

    # 汇总所有模型的折叠结果
    all_fold = []
    for name in models.keys():
        fold_df = pd.read_csv(os.path.join(cv_fold_results_dir, f"{name}_fold_results.csv"))
        all_fold.append(fold_df)
    pd.concat(all_fold, ignore_index=True).to_csv(os.path.join(output_dir, "all_models_cv_fold_summary.csv"), index=False)

    # 基于 pooled OOF 预测计算 CV 级别的 95% CI (与测试集 CI 方法学一致)
    print("\n" + "="*50)
    print("计算各模型 CV 汇总指标的 95% CI (基于 OOF 预测)")
    print("="*50)
    cv_ci_extended = []
    cv_ci_publication = []
    for name in models.keys():
        if name not in model_data:
            continue
        oof_true = model_data[name]['y_true']
        oof_proba = model_data[name]['y_proba']
        oof_pred = (oof_proba >= 0.5).astype(int)
        print(f"  {name}: 计算 CV 95% CI (DeLong AUC + Bootstrap {BOOTSTRAP_ITERATIONS}x)...")
        ci_ext, ci_pub, _ = build_metrics_with_ci(oof_true, oof_pred, oof_proba, n_bootstrap=BOOTSTRAP_ITERATIONS)
        cv_ci_extended.append({'Model': name, **ci_ext})
        cv_ci_publication.append({'Model': name, **ci_pub})
        print(f"    CV AUC (95% CI): {ci_pub['AUC (95% CI)']}")

    cv_results_df = pd.DataFrame(cv_results)
    if cv_ci_extended:
        cv_ci_df = pd.DataFrame(cv_ci_extended)
        cv_results_with_ci = cv_results_df.merge(cv_ci_df, on='Model', how='left')
        cv_results_with_ci.to_csv(os.path.join(output_dir, "cv_results_summary_with_ci.csv"), index=False, encoding='utf-8-sig')
        print(f"  ✓ CV 指标带 95% CI 表已保存: cv_results_summary_with_ci.csv")
    if cv_ci_publication:
        cv_pub_df = pd.DataFrame(cv_ci_publication)
        cv_pub_cols = ['Model', 'AUC (95% CI)', 'AP (95% CI)', 'Accuracy (95% CI)',
                       'Precision (95% CI)', 'Sensitivity (95% CI)', 'Specificity (95% CI)',
                       'F1 (95% CI)']
        cv_pub_df = cv_pub_df.reindex(columns=[c for c in cv_pub_cols if c in cv_pub_df.columns])
        cv_pub_df.to_csv(os.path.join(output_dir, "cv_results_summary_publication_format.csv"), index=False, encoding='utf-8-sig')
        try:
            with pd.ExcelWriter(os.path.join(output_dir, "cv_results_summary_publication_format.xlsx"),
                                engine='openpyxl') as writer:
                cv_pub_df.to_excel(writer, index=False, sheet_name='CV_CI')
                ws = writer.sheets['CV_CI']
                for col in ws.columns:
                    col_letter = col[0].column_letter
                    max_len = max((len(str(c.value)) for c in col), default=8)
                    ws.column_dimensions[col_letter].width = min(max_len + 4, 60)
        except Exception as e:
            print(f"  ⚠ CV 论文格式 Excel 保存失败 ({e})，CSV 已生成")

    # 参数汇总
    params_list = []
    for model_name, params in optimal_parameters.items():
        cleaned = {k.replace('model__',''): v for k,v in params.items()}
        params_list.append({'Model': model_name, **cleaned})
    pd.DataFrame(params_list).to_csv(os.path.join(output_dir, "all_models_optimal_parameters_summary.csv"), index=False)

    plot_all_cv_calibration_curves(model_data, save_path=os.path.join(comparison_plots_dir, "all_models_calibration_curve.JPG"))
    plot_all_cv_decision_curves(model_data, save_path=os.path.join(comparison_plots_dir, "all_models_decision_curve.JPG"))
    plot_all_roc_curves(all_roc_data, save_path=os.path.join(comparison_plots_dir, "all_models_roc_curves.JPG"))
    plot_all_pr_curves(all_pr_data, np.mean(y_train), save_path=os.path.join(comparison_plots_dir, "all_models_pr_curves.JPG"))
    return best_models, cv_results_df, optimal_parameters


def select_and_retrain_best_model(X_train, y_train, best_models, cv_results):
    cv_results = cv_results.sort_values('AUC', ascending=False)
    best_name = cv_results.iloc[0]['Model']
    best_auc = cv_results.iloc[0]['AUC']
    print(f"\n{'='*50}\nBest model from CV: {best_name} (CV AUC={best_auc:.4f})\n{'='*50}")
    final_model = best_models[best_name]['pipeline']
    final_model.fit(X_train, y_train)
    return final_model, best_name, cv_results, best_auc


def evaluate_model(model, X_test, y_test, model_name, output_dir="results", phase="Test"):
    os.makedirs(output_dir, exist_ok=True)
    test_plots_dir = os.path.join(output_dir, f"{phase.lower()}_plots")
    os.makedirs(test_plots_dir, exist_ok=True)
    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:,1]
    tn, fp, fn, tp = confusion_matrix(y_test, y_pred).ravel()
    specificity = tn/(tn+fp)
    metrics = {'AUC': roc_auc_score(y_test, y_proba), 'AP': average_precision_score(y_test, y_proba),
               'Accuracy': accuracy_score(y_test, y_pred), 'Precision': precision_score(y_test, y_pred),
               'Sensitivity': recall_score(y_test, y_pred), 'Specificity': specificity,
               'F1': f1_score(y_test, y_pred), 'Brier': brier_score_loss(y_test, y_proba),
               'Confusion_Matrix': confusion_matrix(y_test, y_pred)}
    print(f"\n{phase} Set Classification Report:\n", classification_report(y_test, y_pred))
    print(f"\n{phase} Set Performance Metrics:")
    for k,v in metrics.items():
        if k != 'Confusion_Matrix': print(f"{k}: {v:.4f}")
    # 95% CI
    print(f"  Computing 95% CI (AUC=DeLong, others=Bootstrap {BOOTSTRAP_ITERATIONS}x)...")
    ci_ext, ci_pub, _ = build_metrics_with_ci(y_test, y_pred, y_proba, n_bootstrap=BOOTSTRAP_ITERATIONS)
    metrics_with_ci = {**metrics, **ci_ext}
    print(f"  AUC (95% CI): {ci_pub['AUC (95% CI)']}")
    # 保存单模型 CI 表
    pd.DataFrame([metrics_with_ci]).to_csv(
        os.path.join(output_dir, f"{model_name}_{phase}_metrics_with_ci.csv"),
        index=False, encoding='utf-8-sig')
    pd.DataFrame([{'Model': model_name, **ci_pub}]).to_csv(
        os.path.join(output_dir, f"{model_name}_{phase}_metrics_publication.csv"),
        index=False, encoding='utf-8-sig')
    fpr, tpr, _ = roc_curve(y_test, y_proba)
    plot_roc_curve(fpr, tpr, metrics['AUC'], model_name, phase, save_path=os.path.join(test_plots_dir, f"{model_name}_{phase.lower()}_roc_curve.JPG"))
    prec, rec, _ = precision_recall_curve(y_test, y_proba)
    plot_pr_curve(prec, rec, metrics['AP'], model_name, phase, save_path=os.path.join(test_plots_dir, f"{model_name}_{phase.lower()}_pr_curve.JPG"))
    plot_confusion_matrix(y_test, y_pred, model_name, phase, save_path=os.path.join(test_plots_dir, f"{model_name}_{phase.lower()}_confusion_matrix.JPG"))
    return metrics, y_proba


# ==================== 校准曲线 ====================
def select_calibration_method(n_samples, n_positive=None):
    if n_positive is None: n_positive = n_samples//2
    if n_samples < 500 or n_positive < 100:
        return 'sigmoid', f"样本量较小或正类样本较少(正类={n_positive})，使用Platt scaling (sigmoid)校准"
    else:
        return 'isotonic', f"样本量充足且正类样本足够(正类={n_positive})，使用Isotonic Regression校准"


def plot_calibration_curve_safe(y_true_train, y_proba_train, y_true_test, y_proba_test, model_name, output_dir="results", n_bins=10, calibration_method='auto'):
    os.makedirs(output_dir, exist_ok=True)
    frac_train_orig, mean_train_orig = calibration_curve(y_true_train, y_proba_train, n_bins=n_bins, strategy='quantile')
    frac_test_orig, mean_test_orig = calibration_curve(y_true_test, y_proba_test, n_bins=n_bins, strategy='quantile')
    calibrated_test = y_proba_test.copy()
    try:
        n_calib = min(len(y_true_train), max(500, int(0.5*len(y_true_train))))
        n_calib = max(n_calib, 10)
        n_pos_calib = np.sum(y_true_train[:n_calib])
        if calibration_method == 'auto':
            method, _ = select_calibration_method(n_calib, n_pos_calib)
        else:
            method = calibration_method
        print(f"  正在对 {model_name} 进行校准（仅使用训练集数据）... 方法: {method}")
        if n_calib >= 10:
            idx = RNG.permutation(len(y_true_train))[:n_calib]
            calib_proba = y_proba_train[idx]; calib_label = y_true_train[idx]
            if method == 'isotonic':
                calibrator = IsotonicRegression(out_of_bounds='clip', increasing=True)
                calibrator.fit(calib_proba, calib_label)
                calibrated_test = calibrator.transform(y_proba_test)
                calibrated_test = np.clip(calibrated_test, 0.001, 0.999)
                calibrated_train = calibrator.transform(y_proba_train)
                calibrated_train = np.clip(calibrated_train, 0.001, 0.999)
            elif method == 'sigmoid':
                calibrator = LogisticRegression(C=1e10, solver='lbfgs', max_iter=1000, random_state=RANDOM_SEED)
                calibrator.fit(calib_proba.reshape(-1,1), calib_label)
                calibrated_test = calibrator.predict_proba(y_proba_test.reshape(-1,1))[:,1]
                calibrated_test = np.clip(calibrated_test, 0.001, 0.999)
                calibrated_train = calibrator.predict_proba(y_proba_train.reshape(-1,1))[:,1]
                calibrated_train = np.clip(calibrated_train, 0.001, 0.999)
            else:
                return calibrated_test
            frac_train_cal, mean_train_cal = calibration_curve(y_true_train, calibrated_train, n_bins=n_bins, strategy='quantile')
            frac_test_cal, mean_test_cal = calibration_curve(y_true_test, calibrated_test, n_bins=n_bins, strategy='quantile')

            plt.figure(figsize=(12,10))
            plt.plot(mean_train_orig, frac_train_orig, "s--", lw=2, ms=6, color=VIBRANT_COLORS[0], alpha=0.6, label="Training (Original)")
            plt.plot(mean_train_cal, frac_train_cal, "o--", lw=2, ms=6, color=VIBRANT_COLORS[1], alpha=0.6, label="Training (Calibrated)")
            plt.plot(mean_test_orig, frac_test_orig, "s-", lw=3, ms=8, color=VIBRANT_COLORS[0], alpha=0.9, label="Test (Original)")
            plt.plot(mean_test_cal, frac_test_cal, "o-", lw=3, ms=8, color=VIBRANT_COLORS[1], alpha=0.9, label="Test (Calibrated - No Leakage)")
            plt.plot([0,1],[0,1], "k:", lw=2, label="Ideal Calibration")
            plt.xlabel("Mean Predicted Probability"); plt.ylabel("Fraction of Positives")
            plt.title(f"{model_name} - Calibration Curves (No Data Leakage, {method})", pad=20)
            plt.legend(loc="lower right"); plt.grid(True, linestyle='--', alpha=0.5)
            plt.tight_layout()
            plt.savefig(os.path.join(output_dir, f"{model_name}_calibration_curve_safe.JPG"), dpi=300, bbox_inches='tight')
            plt.close()

            plt.figure(figsize=(10,8))
            plt.plot(mean_test_orig, frac_test_orig, "s-", lw=3, ms=8, color=VIBRANT_COLORS[0], alpha=0.7, label=f"{model_name} (Test, Original)")
            plt.plot(mean_test_cal, frac_test_cal, "o-", lw=3, ms=8, color=VIBRANT_COLORS[1], alpha=0.9, label=f"{model_name} (Test, Calibrated)")
            plt.plot([0,1],[0,1], "k:", lw=2, label="Ideal Calibration")
            plt.xlabel("Mean Predicted Probability"); plt.ylabel("Fraction of Positives")
            plt.title(f"{model_name} - Test Set Calibration (No Data Leakage, {method})", pad=20)
            plt.legend(loc="lower right"); plt.grid(True, linestyle='--', alpha=0.5)
            plt.tight_layout()
            plt.savefig(os.path.join(output_dir, f"{model_name}_calibration_curve_test_only.JPG"), dpi=300, bbox_inches='tight')
            plt.close()
            print(f"  {model_name} 安全校准完成（方法: {method}）")
        else:
            print(f"  训练样本量不足（{n_calib} < 10），跳过校准")
            plt.figure(figsize=(10,8))
            plt.plot(mean_test_orig, frac_test_orig, "s-", lw=3, ms=8, color=VIBRANT_COLORS[0], alpha=0.9, label=f"{model_name} (Original)")
            plt.plot([0,1],[0,1], "k:", lw=2, label="Ideal Calibration")
            plt.xlabel("Mean Predicted Probability"); plt.ylabel("Fraction of Positives")
            plt.title(f"{model_name} Calibration Curve", pad=20)
            plt.legend(loc="lower right"); plt.grid(True, linestyle='--', alpha=0.5)
            plt.tight_layout()
            plt.savefig(os.path.join(output_dir, f"{model_name}_calibration_curve.JPG"), dpi=300, bbox_inches='tight')
            plt.close()
    except Exception as e:
        print(f"  校准失败: {str(e)}")
        plt.figure(figsize=(10,8))
        plt.plot(mean_test_orig, frac_test_orig, "s-", lw=3, ms=8, color=VIBRANT_COLORS[0], alpha=0.9, label=f"{model_name} (Original)")
        plt.plot([0,1],[0,1], "k:", lw=2, label="Ideal Calibration")
        plt.xlabel("Mean Predicted Probability"); plt.ylabel("Fraction of Positives")
        plt.title(f"{model_name} Calibration Curve", pad=20)
        plt.legend(loc="lower right"); plt.grid(True, linestyle='--', alpha=0.5)
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, f"{model_name}_calibration_curve.JPG"), dpi=300, bbox_inches='tight')
        plt.close()
        return y_proba_test
    return calibrated_test


def calibration_curve_only(y_true, y_proba, model_name, output_dir="results", phase="Test", n_bins=10):
    os.makedirs(output_dir, exist_ok=True)
    frac, mean = calibration_curve(y_true, y_proba, n_bins=n_bins, strategy='quantile')
    plt.figure(figsize=(10,8))
    plt.plot(mean, frac, "s-", lw=3, ms=8, color=VIBRANT_COLORS[0], alpha=0.9, label=f"{model_name} (Original)")
    plt.plot([0,1],[0,1], "k:", lw=2, label="Ideal Calibration")
    plt.xlabel("Mean Predicted Probability"); plt.ylabel("Fraction of Positives")
    plt.title(f"{model_name} - {phase} Calibration Curve", pad=20)
    plt.legend(loc="lower right"); plt.grid(True, linestyle='--', alpha=0.5)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f"{model_name}_calibration_curve.JPG"), dpi=300, bbox_inches='tight')
    plt.close()


# ==================== 95% 置信区间 (CI) 计算 ====================
# DeLong 法用于 AUC (精确解析法)，Bootstrap 用于其他指标
BOOTSTRAP_ITERATIONS = 1000
Z_95 = norm.ppf(0.975)  # ≈ 1.95996


def compute_auc_ci_delong(y_true, y_proba, alpha=0.05):
    """基于 DeLong 方差计算 AUC 的 95% 置信区间。
    复用 compute_auc_variance_delong 已有实现。
    返回: (auc, se, lower, upper)
    """
    y_true = np.asarray(y_true).ravel()
    y_proba = np.asarray(y_proba).ravel()
    try:
        auc_value, var_auc, auc_std, _, _ = compute_auc_variance_delong(y_true, y_proba)
        z = norm.ppf(1 - alpha / 2)
        lower = auc_value - z * auc_std
        upper = auc_value + z * auc_std
        lower = min(max(lower, 0.0), 1.0)
        upper = min(max(upper, 0.0), 1.0)
        return auc_value, auc_std, lower, upper
    except Exception as e:
        auc_value = roc_auc_score(y_true, y_proba)
        print(f"  ⚠ DeLong AUC CI 计算失败 ({e})，仅返回点估计")
        return auc_value, np.nan, np.nan, np.nan


def compute_metrics_bootstrap_ci(y_true, y_pred, y_proba, n_bootstrap=1000,
                                  alpha=0.05, random_seed=RANDOM_SEED, verbose=False):
    """分层 Bootstrap 计算 AP/Accuracy/Precision/Sensitivity/Specificity/F1/Brier 的 95% CI。
    返回: {metric: (lower, point, upper)}
    """
    y_true = np.asarray(y_true).ravel()
    y_pred = np.asarray(y_pred).ravel()
    y_proba = np.asarray(y_proba).ravel()
    n = len(y_true)
    rng = np.random.RandomState(random_seed)

    pos_idx = np.where(y_true == 1)[0]
    neg_idx = np.where(y_true == 0)[0]
    n_pos = len(pos_idx)
    n_neg = len(neg_idx)

    point_estimates = {
        'AP': average_precision_score(y_true, y_proba),
        'Accuracy': accuracy_score(y_true, y_pred),
        'Precision': precision_score(y_true, y_pred, zero_division=0),
        'Sensitivity': recall_score(y_true, y_pred, zero_division=0),
        'Specificity': recall_score(y_true, y_pred, pos_label=0, zero_division=0),
        'F1': f1_score(y_true, y_pred, zero_division=0),
        'Brier': brier_score_loss(y_true, y_proba),
    }

    boot_samples = {m: [] for m in point_estimates}
    n_valid = 0
    n_skipped = 0

    for _ in range(n_bootstrap):
        boot_pos = rng.choice(pos_idx, size=n_pos, replace=True)
        boot_neg = rng.choice(neg_idx, size=n_neg, replace=True)
        boot_idx = np.concatenate([boot_pos, boot_neg])

        y_true_b = y_true[boot_idx]
        y_pred_b = y_pred[boot_idx]
        y_proba_b = y_proba[boot_idx]

        if len(np.unique(y_true_b)) < 2 or len(np.unique(y_pred_b)) < 2:
            n_skipped += 1
            continue

        try:
            boot_samples['AP'].append(average_precision_score(y_true_b, y_proba_b))
            boot_samples['Accuracy'].append(accuracy_score(y_true_b, y_pred_b))
            boot_samples['Precision'].append(precision_score(y_true_b, y_pred_b, zero_division=0))
            boot_samples['Sensitivity'].append(recall_score(y_true_b, y_pred_b, zero_division=0))
            boot_samples['Specificity'].append(recall_score(y_true_b, y_pred_b, pos_label=0, zero_division=0))
            boot_samples['F1'].append(f1_score(y_true_b, y_pred_b, zero_division=0))
            boot_samples['Brier'].append(brier_score_loss(y_true_b, y_proba_b))
            n_valid += 1
        except Exception:
            n_skipped += 1
            continue

    if n_valid < max(100, n_bootstrap * 0.5):
        print(f"  ⚠ Bootstrap 有效样本不足 ({n_valid}/{n_bootstrap})，CI 可能不可靠")

    if verbose:
        print(f"  Bootstrap: 有效={n_valid}, 跳过={n_skipped}")

    lower_pct = 100 * (alpha / 2)
    upper_pct = 100 * (1 - alpha / 2)

    ci_dict = {}
    for m, point in point_estimates.items():
        if len(boot_samples[m]) >= 10:
            arr = np.array(boot_samples[m])
            lower = float(np.percentile(arr, lower_pct))
            upper = float(np.percentile(arr, upper_pct))
            if m in ('AP', 'Accuracy', 'Precision', 'Sensitivity', 'Specificity', 'F1'):
                lower = min(max(lower, 0.0), 1.0)
                upper = min(max(upper, 0.0), 1.0)
        else:
            lower = np.nan
            upper = np.nan
        ci_dict[m] = (lower, point, upper)
    return ci_dict


def format_metric_with_ci(point, lower, upper, decimals=3):
    """格式化为 'point (lower-upper)' 字符串，如 '0.85 (0.80-0.90)'。"""
    if pd.isna(point) or pd.isna(lower) or pd.isna(upper):
        return f"{point:.{decimals}f}" if not pd.isna(point) else "NA"
    return f"{point:.{decimals}f} ({lower:.{decimals}f}-{upper:.{decimals}f})"


def build_metrics_with_ci(y_true, y_pred, y_proba, n_bootstrap=1000, random_seed=RANDOM_SEED):
    """整合 DeLong AUC CI + 其他指标 Bootstrap CI。
    返回:
      ci_extended_dict: {'AUC_Lower':..., 'AUC_Upper':..., 'AP_Lower':..., ...}
      ci_publication_dict: {'AUC (95% CI)': '0.85 (0.80-0.90)', ...}
    """
    y_true = np.asarray(y_true).ravel()
    y_pred = np.asarray(y_pred).ravel()
    y_proba = np.asarray(y_proba).ravel()

    auc_val, auc_se, auc_lo, auc_hi = compute_auc_ci_delong(y_true, y_proba)
    boot_ci = compute_metrics_bootstrap_ci(y_true, y_pred, y_proba,
                                            n_bootstrap=n_bootstrap, random_seed=random_seed)

    extended = {
        'AUC_Lower': auc_lo, 'AUC_Upper': auc_hi, 'AUC_SE': auc_se,
        'AP_Lower': boot_ci['AP'][0], 'AP_Upper': boot_ci['AP'][2],
        'Accuracy_Lower': boot_ci['Accuracy'][0], 'Accuracy_Upper': boot_ci['Accuracy'][2],
        'Precision_Lower': boot_ci['Precision'][0], 'Precision_Upper': boot_ci['Precision'][2],
        'Sensitivity_Lower': boot_ci['Sensitivity'][0], 'Sensitivity_Upper': boot_ci['Sensitivity'][2],
        'Specificity_Lower': boot_ci['Specificity'][0], 'Specificity_Upper': boot_ci['Specificity'][2],
        'F1_Lower': boot_ci['F1'][0], 'F1_Upper': boot_ci['F1'][2],
        'Brier_Lower': boot_ci['Brier'][0], 'Brier_Upper': boot_ci['Brier'][2],
    }
    publication = {
        'AUC (95% CI)': format_metric_with_ci(auc_val, auc_lo, auc_hi),
        'AP (95% CI)': format_metric_with_ci(boot_ci['AP'][1], boot_ci['AP'][0], boot_ci['AP'][2]),
        'Accuracy (95% CI)': format_metric_with_ci(boot_ci['Accuracy'][1], boot_ci['Accuracy'][0], boot_ci['Accuracy'][2]),
        'Precision (95% CI)': format_metric_with_ci(boot_ci['Precision'][1], boot_ci['Precision'][0], boot_ci['Precision'][2]),
        'Sensitivity (95% CI)': format_metric_with_ci(boot_ci['Sensitivity'][1], boot_ci['Sensitivity'][0], boot_ci['Sensitivity'][2]),
        'Specificity (95% CI)': format_metric_with_ci(boot_ci['Specificity'][1], boot_ci['Specificity'][0], boot_ci['Specificity'][2]),
        'F1 (95% CI)': format_metric_with_ci(boot_ci['F1'][1], boot_ci['F1'][0], boot_ci['F1'][2]),
        'Brier (95% CI)': format_metric_with_ci(boot_ci['Brier'][1], boot_ci['Brier'][0], boot_ci['Brier'][2]),
    }
    return extended, publication, {'AUC_point': auc_val, 'AUC_SE': auc_se}


def evaluate_all_models_on_test_set(best_models, X_test, y_test, reference_model_name, output_dir="results", phase="Test"):
    os.makedirs(output_dir, exist_ok=True)
    phase_dir = os.path.join(output_dir, f"{phase.lower()}_all_models")
    os.makedirs(phase_dir, exist_ok=True)
    all_metrics = []
    all_metrics_ci = []
    all_metrics_pub = []
    all_probas = {}
    for name, info in best_models.items():
        print(f"\n{'='*60}\nEvaluating {name} on {phase} set\n{'='*60}")
        model = info['pipeline']
        y_pred = model.predict(X_test)
        y_proba = model.predict_proba(X_test)[:,1]
        all_probas[name] = y_proba
        tn, fp, fn, tp = confusion_matrix(y_test, y_pred).ravel()
        spec = tn/(tn+fp)
        metrics = {'Model': name, 'AUC': roc_auc_score(y_test, y_proba), 'AP': average_precision_score(y_test, y_proba),
                   'Accuracy': accuracy_score(y_test, y_pred), 'Precision': precision_score(y_test, y_pred),
                   'Sensitivity': recall_score(y_test, y_pred), 'Specificity': spec, 'F1': f1_score(y_test, y_pred),
                   'Brier': brier_score_loss(y_test, y_proba), 'TP': tp, 'FP': fp, 'TN': tn, 'FN': fn}
        all_metrics.append(metrics)
        pd.DataFrame({'True_Label': y_test.values, 'Predicted_Label': y_pred, 'Predicted_Probability': y_proba}) \
            .to_csv(os.path.join(phase_dir, f"{name}_predictions.csv"), index=False)
        print(f"\n{name} {phase} Set Performance:")
        for k,v in metrics.items():
            if k not in ['Model','TP','FP','TN','FN']: print(f"  {k}: {v:.4f}")
        # 计算 95% CI（AUC 用 DeLong，其他指标用 Bootstrap）
        print(f"  Computing 95% CI for {name} (AUC=DeLong, others=Bootstrap {BOOTSTRAP_ITERATIONS}x)...")
        ci_ext, ci_pub, _ = build_metrics_with_ci(y_test, y_pred, y_proba, n_bootstrap=BOOTSTRAP_ITERATIONS)
        all_metrics_ci.append({'Model': name, **ci_ext})
        all_metrics_pub.append({'Model': name, **ci_pub})
        print(f"  AUC (95% CI): {ci_pub['AUC (95% CI)']}")
        model_plots = os.path.join(phase_dir, "individual_plots")
        os.makedirs(model_plots, exist_ok=True)
        fpr,tpr,_ = roc_curve(y_test, y_proba)
        plot_roc_curve(fpr,tpr, metrics['AUC'], name, phase, save_path=os.path.join(model_plots, f"{name}_roc_curve.JPG"))
        prec, rec, _ = precision_recall_curve(y_test, y_proba)
        plot_pr_curve(prec, rec, metrics['AP'], name, phase, save_path=os.path.join(model_plots, f"{name}_pr_curve.JPG"))
        plot_confusion_matrix(y_test, y_pred, name, phase, save_path=os.path.join(model_plots, f"{name}_confusion_matrix.JPG"))
        calibration_curve_only(y_test, y_proba, name, output_dir=model_plots, phase=phase)
        plot_decision_curve(y_test, y_proba, name, output_dir=model_plots)
        try:
            feature_dir = os.path.join(phase_dir, "feature_importance")
            os.makedirs(feature_dir, exist_ok=True)
            plot_feature_importance(model, X_test.columns.tolist(), name, output_dir=feature_dir)
        except Exception as e:
            print(f"Feature importance for {name} failed: {str(e)}")
    metrics_df = pd.DataFrame(all_metrics)
    metrics_df.to_csv(os.path.join(phase_dir, "all_models_metrics_summary.csv"), index=False)
    # 保存带 95% CI 的扩展表与论文格式表
    if all_metrics_ci:
        ci_df = pd.DataFrame(all_metrics_ci)
        ci_cols_order = ['Model', 'AUC_Lower', 'AUC_Upper', 'AUC_SE',
                         'AP_Lower', 'AP_Upper', 'Accuracy_Lower', 'Accuracy_Upper',
                         'Precision_Lower', 'Precision_Upper', 'Sensitivity_Lower', 'Sensitivity_Upper',
                         'Specificity_Lower', 'Specificity_Upper', 'F1_Lower', 'F1_Upper',
                         'Brier_Lower', 'Brier_Upper']
        ci_df = ci_df.reindex(columns=[c for c in ci_cols_order if c in ci_df.columns])
        ci_df.to_csv(os.path.join(phase_dir, "all_models_metrics_summary_with_ci.csv"), index=False, encoding='utf-8-sig')
        # 合并点估计与 CI（含点估计 + 上下界）
        merged = metrics_df.merge(ci_df, on='Model', how='left')
        merged.to_csv(os.path.join(phase_dir, "all_models_metrics_summary_full.csv"), index=False, encoding='utf-8-sig')
    if all_metrics_pub:
        pub_df = pd.DataFrame(all_metrics_pub)
        pub_cols_order = ['Model', 'AUC (95% CI)', 'AP (95% CI)', 'Accuracy (95% CI)',
                          'Precision (95% CI)', 'Sensitivity (95% CI)', 'Specificity (95% CI)',
                          'F1 (95% CI)', 'Brier (95% CI)']
        pub_df = pub_df.reindex(columns=[c for c in pub_cols_order if c in pub_df.columns])
        pub_df.to_csv(os.path.join(phase_dir, "all_models_metrics_summary_publication_format.csv"), index=False, encoding='utf-8-sig')
        try:
            with pd.ExcelWriter(os.path.join(phase_dir, "all_models_metrics_summary_publication_format.xlsx"),
                                engine='openpyxl') as writer:
                pub_df.to_excel(writer, index=False, sheet_name=f'{phase}_CI')
                ws = writer.sheets[f'{phase}_CI']
                for col in ws.columns:
                    col_letter = col[0].column_letter
                    max_len = max((len(str(c.value)) for c in col), default=8)
                    ws.column_dimensions[col_letter].width = min(max_len + 4, 60)
        except Exception as e:
            print(f"  ⚠ 论文格式 Excel 保存失败 ({e})，CSV 已生成")
    print(f"\nModel Ranking by AUC on {phase} Set:")
    for i, (_, row) in enumerate(metrics_df.sort_values('AUC', ascending=False).iterrows(), 1):
        auc_ci_str = next((p['AUC (95% CI)'] for p in all_metrics_pub if p['Model'] == row['Model']), 'NA')
        print(f"{i}. {row['Model']}: AUC={row['AUC']:.4f} (95% CI: {auc_ci_str})")

    generate_model_comparison_plots(metrics_df, phase_dir, phase)

    combined_dir = os.path.join(phase_dir, "combined_plots")
    os.makedirs(combined_dir, exist_ok=True)
    plot_test_roc_curves_all_models(all_probas, y_test, phase, save_path=os.path.join(combined_dir, f"all_models_roc_curves_{phase}.JPG"))
    plot_test_pr_curves_all_models(all_probas, y_test, phase, save_path=os.path.join(combined_dir, f"all_models_pr_curves_{phase}.JPG"))
    plot_test_calibration_curves_all_models(all_probas, y_test, phase, save_path=os.path.join(combined_dir, f"all_models_calibration_curves_{phase}.JPG"))
    plot_test_decision_curves_all_models(all_probas, y_test, phase, save_path=os.path.join(combined_dir, f"all_models_decision_curves_{phase}.JPG"))

    print(f"\nPerforming pairwise DeLong tests on {phase} set (Bonferroni)...")
    p_mat, aucs = perform_pairwise_delong_tests_with_correction(all_probas, y_test, output_dir=phase_dir, phase=phase)
    # 保存热力图为JPG文件
    heatmap_save_path = os.path.join(phase_dir, f"pairwise_delong_heatmap_{phase}.JPG")
    plot_pairwise_delong_heatmap(p_mat, aucs, phase=phase, save_path=heatmap_save_path)
    print(f"\nDeLong test with reference model '{reference_model_name}'...")
    delong_res, _ = perform_delong_tests_with_reference_and_correction(all_probas, y_test, reference_model_name, output_dir=phase_dir, phase=phase)
    return metrics_df, None, all_probas, delong_res


def plot_pairwise_delong_heatmap(p_matrix, auc_values, phase="Test", save_path=None):
    models = p_matrix.index.tolist()
    n_models = len(models)
    sorted_models = sorted(models, key=lambda x: auc_values[x], reverse=True)
    p_sorted = p_matrix.loc[sorted_models, sorted_models]
    heatmap_data = p_sorted.copy().astype(float)
    for i in range(n_models):
        for j in range(n_models):
            if i <= j: heatmap_data.iloc[i,j] = np.nan
    sig_mat = pd.DataFrame('', index=sorted_models, columns=sorted_models)
    for i in range(n_models):
        for j in range(n_models):
            if i > j:
                p_val = p_sorted.iloc[i,j]
                if not pd.isna(p_val):
                    sig_mat.iloc[i,j] = get_significance_symbol(p_val)
    fig, ax = plt.subplots(figsize=(16,14))
    colors_rdbl = ['#d73027','#f46d43','#fdae61','#fee090','#e0f3f8','#abd9e9','#74add1','#4575b4']
    cmap = mcolors.LinearSegmentedColormap.from_list('RedBlue', colors_rdbl)
    im = ax.imshow(heatmap_data, cmap=cmap, aspect='auto', vmin=0, vmax=0.1, interpolation='nearest')
    cbar = plt.colorbar(im, ax=ax, shrink=0.7, fraction=0.046, pad=0.04)
    cbar.set_label('P Value (Bonferroni corrected)')
    cbar.ax.tick_params(labelsize=HEATMAP_CBAR_FONT_SIZE)
    ax.set_xticks(np.arange(n_models)); ax.set_yticks(np.arange(n_models))
    ax.set_xticklabels(sorted_models, rotation=45, ha='right', fontsize=HEATMAP_TICK_FONT_SIZE, fontweight='bold')
    ax.set_yticklabels(sorted_models, fontsize=HEATMAP_TICK_FONT_SIZE, fontweight='bold')
    ax.set_title(f'{phase} Set - Pairwise DeLong Test P-Values', pad=30)
    # 绘制网格线
    for i in range(n_models):
        for j in range(n_models):
            if i > j:
                rect = plt.Rectangle((j-0.5, i-0.5), 1, 1, fill=False, edgecolor='black', linewidth=1.5)
                ax.add_patch(rect)
    for i in range(n_models):
        for j in range(n_models):
            if i <= j:
                rect = plt.Rectangle((j-0.5, i-0.5), 1, 1, fill=True, facecolor='white', edgecolor='none', zorder=-1)
                ax.add_patch(rect)
    for i in range(n_models):
        for j in range(n_models):
            if i > j:
                sym = sig_mat.iloc[i,j]
                if sym:
                    p_val = p_sorted.iloc[i,j]
                    if not pd.isna(p_val):
                        text_color = 'white' if p_val < 0.05 else 'black'
                        if sym == '***': fsize = HEATMAP_ANNOTATION_FONT_SIZE+6
                        elif sym == '**': fsize = HEATMAP_ANNOTATION_FONT_SIZE+4
                        elif sym == '*': fsize = HEATMAP_ANNOTATION_FONT_SIZE+2
                        else: fsize = HEATMAP_ANNOTATION_FONT_SIZE
                        ax.text(j, i, sym, ha='center', va='center', fontsize=fsize, fontweight='bold', color=text_color)
    for i in range(n_models):
        ax.text(i, i, '-', ha='center', va='center', fontsize=HEATMAP_ANNOTATION_FONT_SIZE+6, fontweight='bold', color='black')
    for spine in ax.spines.values():
        spine.set_visible(True); spine.set_linewidth(2); spine.set_color('black')
    leg_elements = [Patch(facecolor='none', edgecolor='none', label='*** p < 0.001'),
                    Patch(facecolor='none', edgecolor='none', label='** p < 0.01'),
                    Patch(facecolor='none', edgecolor='none', label='* p < 0.05'),
                    Patch(facecolor='none', edgecolor='none', label='ns not significant')]
    legend = ax.legend(handles=leg_elements, loc='upper right', fontsize=HEATMAP_LEGEND_FONT_SIZE, frameon=True, framealpha=0.9, title='Significance Levels (Bonferroni)', title_fontsize=HEATMAP_LEGEND_FONT_SIZE+2)
    legend.get_title().set_fontweight('bold')
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
    else:
        plt.show()


# ==================== 修改的 plot_delong_comparison_bar 函数（更紧凑、字号增大）====================
def plot_delong_comparison_bar(delong_results, ref_name, ref_auc, output_dir, phase="Test"):
    results = []
    for _, row in delong_results.iterrows():
        p_val = row.get('P_Value_Corrected', row.get('P_Value_Raw',1.0))
        sig = row.get('Significance_Level_Corrected', row.get('Significance_Level_Raw','ns'))
        results.append({'Model': row['Compared_Model'], 'AUC': row['Compared_AUC'], 'AUC_Diff': row['AUC_Difference'],
                        'P_Value': p_val, 'Significance_Level': sig})
    df = pd.DataFrame(results).sort_values('AUC', ascending=False)
    models = df['Model'].tolist(); aucs = df['AUC'].tolist(); sigs = df['Significance_Level'].tolist()
    fig, ax = plt.subplots(1,1, figsize=(12,9))
    plt.subplots_adjust(bottom=0.15)
    x = np.arange(len(models))
    bars = ax.bar(x, aucs, color=VIBRANT_COLORS[:len(models)], alpha=0.8, edgecolor='black', linewidth=2)
    ax.axhline(y=ref_auc, color='red', linestyle='--', lw=2.5, alpha=0.8, label=f'Reference: {ref_name} (AUC={ref_auc:.3f})')
    for i, (bar, auc_val, sig) in enumerate(zip(bars, aucs, sigs)):
        ax.text(bar.get_x()+bar.get_width()/2, bar.get_height()+0.01, f'{auc_val:.3f}', ha='center', va='bottom', fontsize=ANNOTATION_FONT_SIZE, fontweight='bold')
        if sig not in ['ns','Error']:
            ax.text(bar.get_x()+bar.get_width()/2, bar.get_height()+0.035, sig, ha='center', va='bottom', fontsize=ANNOTATION_FONT_SIZE+4, fontweight='bold', color='red')
    ax.set_xlabel('Model'); ax.set_ylabel('AUC')
    ax.set_title(f'{phase} Set - AUC Comparison (vs {ref_name})', pad=20)
    ax.set_xticks(x); ax.set_xticklabels(models, rotation=45, ha='right')
    ax.set_ylim(0,1.08); ax.legend(loc='lower right', frameon=True); ax.grid(True, alpha=0.3, axis='y')
    # ---------- 修改：图例内容更紧凑，去掉冗余的 (Bonferroni)，并增大字号 ----------
    leg_elements = [Patch(facecolor='none',edgecolor='none',label='*** p < 0.001'),
                    Patch(facecolor='none',edgecolor='none',label='** p < 0.01'),
                    Patch(facecolor='none',edgecolor='none',label='* p < 0.05'),
                    Patch(facecolor='none',edgecolor='none',label='ns not significant')]
    fig.legend(handles=leg_elements, loc='lower center', ncol=4, fontsize=LEGEND_FONT_SIZE+6,  # 字号从12增大到16
               frameon=True, bbox_to_anchor=(0.5, -0.1))  # 略微上移使布局更紧凑
    plt.subplots_adjust(bottom=0.2, top=0.93)
    plt.savefig(os.path.join(output_dir, f"delong_comparison_{phase}_with_reference_bonferroni.JPG"), dpi=300, bbox_inches='tight')
    plt.close()


def generate_model_comparison_plots(metrics_df, output_dir, phase="Test"):
    os.makedirs(output_dir, exist_ok=True)
    metrics = ['AUC','Accuracy','Precision','Sensitivity','Specificity','F1']
    fig, axes = plt.subplots(2,3, figsize=(24,16))
    axes = axes.flatten()
    for idx, m in enumerate(metrics):
        ax = axes[idx]
        sorted_df = metrics_df.sort_values(m, ascending=False)
        bars = ax.barh(sorted_df['Model'], sorted_df[m], color=VIBRANT_COLORS[:len(sorted_df)], alpha=0.8, height=0.7)
        ax.set_xlabel(m); ax.set_title(f'Model Comparison by {m}', pad=20)
        for bar in bars:
            width = bar.get_width()
            ax.text(width*1.01, bar.get_y()+bar.get_height()/2, f'{width:.3f}', va='center', fontsize=ANNOTATION_FONT_SIZE, fontweight='bold')
        ax.set_xlim(0, max(sorted_df[m])*1.15); ax.grid(True, alpha=0.3, axis='x')
    for idx in range(len(metrics), len(axes)): fig.delaxes(axes[idx])
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f"model_comparison_{phase}.JPG"), dpi=300, bbox_inches='tight')
    plt.close()

    top4 = metrics_df.nlargest(4, 'AUC')
    radar_metrics = ['AUC','Accuracy','Precision','Sensitivity','Specificity','F1']
    radar_data = top4[radar_metrics].copy()
    for m in radar_metrics:
        if radar_data[m].std() > 0:
            radar_data[m] = (radar_data[m] - radar_data[m].min()) / (radar_data[m].max() - radar_data[m].min())
        else:
            radar_data[m] = 0.5
    angles = np.linspace(0, 2*np.pi, len(radar_metrics), endpoint=False).tolist()
    angles += angles[:1]
    fig = plt.figure(figsize=(14,12))
    ax = fig.add_subplot(111, polar=True)
    for i, (_, row) in enumerate(top4.iterrows()):
        values = radar_data.iloc[i].tolist() + [radar_data.iloc[i].tolist()[0]]
        ax.plot(angles, values, 'o-', lw=3, ms=8, color=VIBRANT_COLORS[i], label=row['Model'])
        ax.fill(angles, values, color=VIBRANT_COLORS[i], alpha=0.1)
    ax.set_xticks(angles[:-1]); ax.set_xticklabels(radar_metrics)
    ax.set_ylim(0,1); ax.set_title(f'Top 4 Models Performance Radar ({phase})', size=TITLE_FONT_SIZE, fontweight='bold', y=1.1)
    ax.legend(loc='upper right', bbox_to_anchor=(1.3,1.1), frameon=True, framealpha=0.9)
    ax.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f"model_radar_{phase}.JPG"), dpi=300, bbox_inches='tight')
    plt.close()


def plot_test_roc_curves_all_models(model_probas, y_true, phase="Test", save_path=None):
    plt.figure(figsize=(12,10))
    colors = VIBRANT_COLORS[:len(model_probas)]
    for idx, (name, proba) in enumerate(model_probas.items()):
        fpr, tpr, _ = roc_curve(y_true, proba)
        auc_val = auc(fpr, tpr)
        plt.plot(fpr, tpr, color=colors[idx], lw=3, alpha=0.9, label=f'{name} (AUC = {auc_val:.3f})')
    plt.plot([0,1],[0,1], 'navy', lw=2, linestyle='--', alpha=0.8, label='Random')
    plt.xlim(0,1); plt.ylim(0,1.05)
    plt.xlabel('False Positive Rate (1 - Specificity)'); plt.ylabel('True Positive Rate (Sensitivity)')
    plt.title(f'{phase} Set ROC Curves (All Models)', pad=20)
    plt.legend(loc="lower right"); plt.grid(True, linestyle='--', alpha=0.5)
    plt.tight_layout()
    if save_path: plt.savefig(save_path, dpi=300, bbox_inches='tight'); plt.close()
    else: plt.show()


def plot_test_pr_curves_all_models(model_probas, y_true, phase="Test", save_path=None):
    plt.figure(figsize=(12,10))
    colors = VIBRANT_COLORS[:len(model_probas)]
    for idx, (name, proba) in enumerate(model_probas.items()):
        prec, rec, _ = precision_recall_curve(y_true, proba)
        ap = average_precision_score(y_true, proba)
        plt.step(rec, prec, where='post', lw=3, alpha=0.9, color=colors[idx], label=f'{name} (AP = {ap:.3f})')
    prevalence = np.mean(y_true)
    plt.plot([0,1], [prevalence, prevalence], 'r--', lw=2, label=f'Baseline (Prevalence={prevalence:.2f})')
    plt.xlabel('Recall (Sensitivity)'); plt.ylabel('Precision')
    plt.ylim(0,1.05); plt.xlim(0,1)
    plt.title(f'{phase} Set Precision-Recall Curves (All Models)', pad=20)
    plt.legend(loc="lower right"); plt.grid(True, linestyle='--', alpha=0.5)
    plt.tight_layout()
    if save_path: plt.savefig(save_path, dpi=300, bbox_inches='tight'); plt.close()
    else: plt.show()


def plot_test_calibration_curves_all_models(model_probas, y_true, phase="Test", save_path=None, n_bins=10):
    plt.figure(figsize=(12,10))
    colors = VIBRANT_COLORS[:len(model_probas)]
    markers = ['o','s','^','D','v','<','>','p','*']
    for idx, (name, proba) in enumerate(model_probas.items()):
        frac, mean = calibration_curve(y_true, proba, n_bins=n_bins, strategy='quantile')
        plt.plot(mean, frac, marker=markers[idx%len(markers)], color=colors[idx], lw=3, ms=8, label=name)
    plt.plot([0,1],[0,1], 'k:', lw=2, label="Ideal Calibration")
    plt.xlabel("Mean Predicted Probability"); plt.ylabel("Fraction of Positives")
    plt.title(f"{phase} Set Calibration Curves (All Models)", pad=20)
    plt.legend(loc="lower right", frameon=True, facecolor='white')
    plt.grid(True, linestyle='--', alpha=0.5); plt.xlim(-0.05,1.05); plt.ylim(-0.05,1.05)
    plt.tight_layout()
    if save_path: plt.savefig(save_path, dpi=300, bbox_inches='tight'); plt.close()
    else: plt.show()


def plot_test_decision_curves_all_models(model_probas, y_true, phase="Test", save_path=None):
    plt.figure(figsize=(12,10))
    colors = VIBRANT_COLORS[:len(model_probas)]
    prevalence = np.mean(y_true)
    thresholds = np.linspace(0.01,0.99,99)
    for idx, (name, proba) in enumerate(model_probas.items()):
        n = len(y_true)
        nb = []
        for pt in thresholds:
            y_pred = (proba >= pt).astype(int)
            tp = np.sum((y_true==1)&(y_pred==1))
            fp = np.sum((y_true==0)&(y_pred==1))
            nb.append(tp/n - fp/n * (pt/(1-pt)))
        plt.plot(thresholds, nb, color=colors[idx], lw=2.5, label=name)
    treat_all = [prevalence - (1-prevalence)*(pt/(1-pt)) for pt in thresholds]
    treat_none = [0]*len(thresholds)
    plt.plot(thresholds, treat_all, 'k--', lw=2, label="Treat All")
    plt.plot(thresholds, treat_none, 'k-.', lw=2, label="Treat None")
    plt.xlabel("Threshold Probability"); plt.ylabel("Net Benefit")
    plt.title(f"{phase} Set Decision Curves (All Models)", pad=20)
    plt.legend(loc='upper right', frameon=True, facecolor='white')
    plt.grid(True, linestyle='--', alpha=0.5)
    all_nb = []
    for proba in model_probas.values():
        n = len(y_true)
        nb = [np.sum((y_true==1)&((proba>=pt).astype(int)==1))/n - np.sum((y_true==0)&((proba>=pt).astype(int)==1))/n * (pt/(1-pt)) for pt in thresholds]
        all_nb.extend(nb)
    max_b = max(max(all_nb), max(treat_all), 0.1)
    plt.ylim(-0.05, max_b*1.1)
    plt.tight_layout()
    if save_path: plt.savefig(save_path, dpi=300, bbox_inches='tight'); plt.close()
    else: plt.show()


def evaluate_external_test_fixed(model, X_external, y_external, model_name, feature_names, train_thresholds, output_dir="results"):
    print(f"\n{'='*60}\nExternal Independent Test Set Validation\n{'='*60}")
    external_dir = os.path.join(output_dir, "external_test")
    os.makedirs(external_dir, exist_ok=True)
    missing = set(feature_names) - set(X_external.columns)
    if missing:
        for f in missing: X_external[f] = np.nan
    X_external = X_external[feature_names]
    external_metrics, external_proba = evaluate_model(model, X_external, y_external, model_name, output_dir=external_dir, phase="External_Test")
    # 基线特征对比
    try:
        ext_data = X_external.copy(); ext_data['target'] = y_external.values
        train_df = pd.read_csv(os.path.join(output_dir, "训练集数据.csv"))
        test_df = pd.read_csv(os.path.join(output_dir, "测试集数据.csv"))
        generate_external_baseline_comparison(train_df, test_df, ext_data, feature_names, 'target', external_dir)
    except Exception as e: print(f"基线特征分析失败: {e}")
    # 校准曲线
    cal_dir = os.path.join(external_dir, "external_test_plots")
    os.makedirs(cal_dir, exist_ok=True)
    calibration_curve_only(y_external, external_proba, model_name, output_dir=cal_dir, phase="External_Test")
    plot_decision_curve(y_external, external_proba, model_name, output_dir=cal_dir)
    # 风险分层
    results_df = pd.DataFrame({'True_Label': y_external.values, 'Predicted_Label': model.predict(X_external),
                               'Predicted_Probability': external_proba, 'Calibrated_Probability': external_proba})
    if train_thresholds is None:
        train_thresholds = {'low_threshold':0.40,'high_threshold':0.80,'method':'clinical_driven'}
    results_with_risk, _, _, _ = add_risk_stratification_fixed(results_df, calibrated_prob_col='Calibrated_Probability', thresholds=train_thresholds,
                                                               risk_labels=['low-risk','medium-risk','high-risk'], output_dir=external_dir, dataset_name="External_Test_Set")
    results_with_risk.to_csv(os.path.join(external_dir, "external_test_set_predictions.csv"), index=False)
    plot_risk_stratification(results_with_risk, calibrated_prob_col='Calibrated_Probability', risk_level_col='Risk_Level',
                             output_dir=os.path.join(external_dir, "risk_stratification_plots"), dataset_name="External Test Set")
    # 保存指标
    pd.DataFrame([external_metrics]).to_csv(os.path.join(external_dir, "external_test_metrics_summary.csv"), index=False)
    # 内外对比
    internal_path = os.path.join(output_dir, "test_all_models", "all_models_metrics_summary.csv")
    if os.path.exists(internal_path):
        internal_df = pd.read_csv(internal_path)
        if not internal_df.empty:
            internal_row = internal_df.iloc[0]
            comp = []
            for metric in ['AUC','AP','Accuracy','Precision','Sensitivity','Specificity','F1','Brier']:
                if metric in internal_row and metric in external_metrics:
                    comp.append({'Metric':metric, 'Internal_Test':internal_row[metric], 'External_Test':external_metrics[metric],
                                 'Difference':external_metrics[metric]-internal_row[metric]})
            pd.DataFrame(comp).to_csv(os.path.join(external_dir, "performance_comparison.csv"), index=False)
    return external_metrics, external_proba


def evaluate_all_models_on_external_test_fixed(best_models, X_external, y_external, feature_names, reference_model_name, train_thresholds, output_dir="results"):
    print(f"\n{'='*60}\nExternal Independent Test Set Validation for ALL Models\n{'='*60}")
    ext_dir = os.path.join(output_dir, "external_test_all_models")
    os.makedirs(ext_dir, exist_ok=True)
    missing = set(feature_names) - set(X_external.columns)
    if missing:
        for f in missing: X_external[f] = np.nan
    X_external = X_external[feature_names]
    metrics_df, _, _, _ = evaluate_all_models_on_test_set(best_models, X_external, y_external, reference_model_name, output_dir=ext_dir, phase="External_Test")
    best = metrics_df.nlargest(1, 'AUC').iloc[0]['Model']
    best_model = best_models[best]['pipeline']
    # 最佳模型风险分层
    try:
        y_pred = best_model.predict(X_external)
        y_proba = best_model.predict_proba(X_external)[:,1]
        res = pd.DataFrame({'True_Label': y_external.values, 'Predicted_Label': y_pred,
                            'Predicted_Probability': y_proba, 'Calibrated_Probability': y_proba})
        _, _, _, _ = add_risk_stratification_fixed(res, calibrated_prob_col='Calibrated_Probability', thresholds=train_thresholds,
                                                   risk_labels=['low-risk','medium-risk','high-risk'],
                                                   output_dir=os.path.join(ext_dir, "best_model_risk_stratification"),
                                                   dataset_name=f"External_Test_Set_Best_Model_{best}")
        res.to_csv(os.path.join(ext_dir, f"best_model_{best}_external_predictions_with_risk.csv"), index=False)
        plot_risk_stratification(res, calibrated_prob_col='Calibrated_Probability', risk_level_col='Risk_Level',
                                 output_dir=os.path.join(ext_dir, "best_model_risk_stratification_plots"),
                                 dataset_name=f"External Test Set - Best Model ({best})")
    except Exception as e: print(f"最佳模型风险分层失败: {e}")
    # 外部基线特征分析（新增）
    try:
        ext_data = X_external.copy()
        ext_data['target'] = y_external.values
        train_data_path = os.path.join(output_dir, "训练集数据.csv")
        test_data_path = os.path.join(output_dir, "测试集数据.csv")
        if os.path.exists(train_data_path) and os.path.exists(test_data_path):
            train_data = pd.read_csv(train_data_path)
            test_data = pd.read_csv(test_data_path)
            _ = generate_external_baseline_comparison(train_data, test_data, ext_data, feature_names, 'target', ext_dir)
            print("✓ 外部验证队列基线特征分析完成")
        else:
            print("警告：未找到训练集或测试集数据文件，跳过外部验证基线特征分析")
    except Exception as e:
        print(f"外部验证基线特征分析失败: {str(e)}")
    # 内部外部对比图
    internal_path = os.path.join(output_dir, "test_all_models", "all_models_metrics_summary.csv")
    if os.path.exists(internal_path):
        internal_df = pd.read_csv(internal_path)
        comp = []
        for model in set(internal_df['Model']).intersection(set(metrics_df['Model'])):
            int_row = internal_df[internal_df['Model']==model].iloc[0]
            ext_row = metrics_df[metrics_df['Model']==model].iloc[0]
            for metric in ['AUC','AP','Accuracy','Precision','Sensitivity','Specificity','F1']:
                if int_row[metric] != 0:
                    comp.append({'Model':model, 'Metric':metric,
                                 'Internal_Test':int_row[metric], 'External_Test':ext_row[metric],
                                 'Difference':ext_row[metric]-int_row[metric],
                                 'Percent_Change':((ext_row[metric]-int_row[metric])/int_row[metric]*100)})
        pd.DataFrame(comp).to_csv(os.path.join(ext_dir, "internal_external_comparison.csv"), index=False)
        generate_internal_external_comparison_plot(pd.DataFrame(comp), ext_dir)
    return metrics_df


def generate_internal_external_comparison_plot(comparison_df, output_dir):
    auc_data = comparison_df[comparison_df['Metric'] == 'AUC']
    if auc_data.empty: return
    fig, (ax1, ax2) = plt.subplots(1,2, figsize=(20,10))
    models = auc_data['Model'].values
    internal = auc_data['Internal_Test'].values
    external = auc_data['External_Test'].values
    x = np.arange(len(models)); width=0.35
    ax1.bar(x-width/2, internal, width, label='Internal Test', color=VIBRANT_COLORS[0], edgecolor='black')
    ax1.bar(x+width/2, external, width, label='External Test', color=VIBRANT_COLORS[1], edgecolor='black')
    ax1.set_xlabel('Model'); ax1.set_ylabel('AUC'); ax1.set_title('AUC Comparison: Internal vs External Test', pad=20)
    ax1.set_xticks(x); ax1.set_xticklabels(models, rotation=45, ha='right')
    ax1.legend(); ax1.grid(True, alpha=0.3, axis='y')
    for i, v in enumerate(internal): ax1.text(i-width/2, v+0.01, f'{v:.3f}', ha='center', va='bottom', fontsize=ANNOTATION_FONT_SIZE, fontweight='bold')
    for i, v in enumerate(external): ax1.text(i+width/2, v+0.01, f'{v:.3f}', ha='center', va='bottom', fontsize=ANNOTATION_FONT_SIZE, fontweight='bold')
    change = auc_data['Percent_Change'].values
    colors = ['red' if c<0 else 'green' for c in change]
    bars = ax2.bar(models, change, color=colors, edgecolor='black')
    ax2.set_xlabel('Model'); ax2.set_ylabel('Performance Change (%)'); ax2.set_title('Performance Change from Internal to External Test', pad=20)
    ax2.set_xticklabels(models, rotation=45, ha='right')
    ax2.axhline(0, color='black', alpha=0.3)
    ax2.grid(True, alpha=0.3, axis='y')
    for bar, val in zip(bars, change):
        if not np.isnan(val):
            ypos = val+0.5 if val>=0 else val-0.5
            va = 'bottom' if val>=0 else 'top'
            ax2.text(bar.get_x()+bar.get_width()/2, ypos, f'{val:.1f}%', ha='center', va=va, fontsize=ANNOTATION_FONT_SIZE, fontweight='bold')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "internal_external_comparison.JPG"), dpi=300, bbox_inches='tight')
    plt.close()


def plot_feature_importance(model, feature_names, model_name, output_dir="results"):
    model_est = model.named_steps['model']
    supported = (xgb.XGBClassifier, lgb.LGBMClassifier, RandomForestClassifier, LogisticRegression, GradientBoostingClassifier)
    if not isinstance(model_est, supported):
        print(f"{model_name} does not support feature importance analysis, skipping.")
        return False
    try:
        if hasattr(model_est, 'feature_importances_'):
            imp = model_est.feature_importances_
        elif hasattr(model_est, 'coef_'):
            coef = model_est.coef_
            imp = np.abs(coef).mean(axis=0) if len(coef.shape)>1 else np.abs(coef)
        else:
            return False
        feat_imp = pd.DataFrame({'Feature': feature_names, 'Importance': imp}).sort_values('Importance', ascending=False).head(15)
        plt.figure(figsize=(12,8))
        ax = sns.barplot(x='Importance', y='Feature', data=feat_imp, palette='viridis')
        plt.title(f"{model_name} - Feature Importance", pad=20)
        plt.xlabel("Importance Score"); plt.ylabel("Features")
        max_imp = feat_imp['Importance'].max()
        if max_imp>0: plt.xlim(0, max_imp*1.15)
        for p in ax.patches:
            width = p.get_width()
            plt.text(width*1.02, p.get_y()+p.get_height()/2, f'{width:.4f}', ha='left', va='center', fontsize=ANNOTATION_FONT_SIZE)
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, f"{model_name}_feature_importance.JPG"), dpi=300, bbox_inches='tight')
        plt.close()
        return True
    except Exception as e:
        print(f"Feature importance failed: {e}")
        return False


# ==================== SHAP Force Strip (典型阴性/阳性病例) ====================
def _select_force_case_index(y_true, y_proba, target_label, p_lo, p_hi):
    """从样本子集中选择典型病例用于 Force Strip。
    target_label: 0=典型阴性病例, 1=高风险阳性病例
    p_lo, p_hi: 目标预测概率范围
    返回: 样本在子集中的索引，若不存在返回 None
    """
    y_true = np.asarray(y_true).ravel()
    y_proba = np.asarray(y_proba).ravel()
    mask = (y_true == target_label)
    if mask.sum() == 0:
        return None
    indices = np.where(mask)[0]
    probs = y_proba[indices]
    in_range = (probs >= p_lo) & (probs <= p_hi)
    if in_range.sum() > 0:
        target_idx = indices[in_range]
        target_probs = probs[in_range]
        if target_label == 0:
            return int(target_idx[np.argmax(target_probs)])  # 阴性：选概率最高（最接近边界）
        else:
            return int(target_idx[np.argmin(target_probs)])  # 阳性：选概率最低（最接近边界）
    # 降级：选概率最接近决策边界的样本
    if target_label == 0:
        return int(indices[np.argmax(probs)])
    else:
        return int(indices[np.argmin(probs)])


def plot_force_strip_sample(model_name, sample_idx, y_true, y_proba,
                             X_orig_sample, shap_vals, base_value, feature_names,
                             output_dir, sample_type='negative',
                             output_space='margin', top_k=7):
    """绘制单个样本的 Force Strip（横向贡献条）。

    改进点：
    1. Top-K（默认 7）按 |SHAP| 排序，不分方向
    2. 其余特征按方向合并为两段：其余正贡献合计 / 其余负贡献合计（仅在非零时追加）
       不再单独显示 "other_sum" 标签
    3. 画布加大到 (26, 12)，避免 f(x) 标签与 x 轴标签碰撞
    4. 文件名带 output_space 后缀（_margin / _probability / _logodds）
    5. 添加贡献图例

    sample_type: 'negative' (典型阴性病例) 或 'positive' (高风险阳性病例)
    output_space: 'margin' (decision_function) / 'probability' (predict_proba) / 'logodds' (TreeExplainer)
    """
    os.makedirs(output_dir, exist_ok=True)

    def _sigmoid(x):
        return 1.0 / (1.0 + np.exp(-np.clip(x, -50, 50)))

    shap_sample = np.asarray(shap_vals[sample_idx]).ravel().astype(float)
    X_sample = np.asarray(X_orig_sample[sample_idx]).ravel().astype(float)

    # Top-K 重要特征（按绝对 SHAP 值，不分方向）
    abs_shap = np.abs(shap_sample)
    k = min(top_k, len(shap_sample))
    top_idx = np.argsort(abs_shap)[-k:][::-1]
    top_shap = shap_sample[top_idx]
    top_features = [str(feature_names[i]) for i in top_idx]
    top_X = X_sample[top_idx]

    # 其余特征：按方向合并为两段（不再使用 "other_sum" 标签）
    other_mask = np.ones(len(shap_sample), dtype=bool)
    other_mask[top_idx] = False
    rest_shap = shap_sample[other_mask]
    rest_pos_sum = float(rest_shap[rest_shap > 0].sum())
    rest_neg_sum = float(rest_shap[rest_shap < 0].sum())
    rest_pos_n = int((rest_shap > 0).sum())
    rest_neg_n = int((rest_shap < 0).sum())

    # 构建 SHAP 数组：top-K + 至多 2 个方向合并段
    display_shap_list = list(top_shap)
    display_features = []
    for fname, fval, sval in zip(top_features, top_X, top_shap):
        sign = "+" if sval >= 0 else ""
        display_features.append(f"{fname}={fval:.3f}\n(SHAP={sign}{sval:.3f})")

    if rest_pos_sum > 0:
        display_shap_list.append(rest_pos_sum)
        display_features.append(f"其余正贡献合计\n(n={rest_pos_n}, SHAP=+{rest_pos_sum:.3f})")
    if rest_neg_sum < 0:
        display_shap_list.append(rest_neg_sum)
        display_features.append(f"其余负贡献合计\n(n={rest_neg_n}, SHAP={rest_neg_sum:.3f})")

    display_shap = np.asarray(display_shap_list, dtype=float)

    # 计算 base/final（output_space 决定语义）
    base_value = float(base_value)
    final_value = base_value + float(shap_sample.sum())
    pred_prob = float(y_proba[sample_idx])
    true_label = int(y_true[sample_idx])

    # 概率换算（仅在 margin / logodds 空间时有意义；probability 空间直接给数值）
    if output_space in ('margin', 'logodds'):
        base_prob = _sigmoid(base_value)
        final_prob = _sigmoid(final_value)
        base_repr = f"logit={base_value:.3f} (p={base_prob:.3f})"
        final_repr = f"logit={final_value:.3f} (p={final_prob:.3f})"
        link_mode = 'logit'
    else:  # probability
        base_prob = base_value
        final_prob = final_value
        base_repr = f"p={base_value:.3f}"
        final_repr = f"p={final_value:.3f}"
        link_mode = 'identity'

    base_value_plot = round(base_value, 3)
    display_shap_plot = np.round(display_shap.astype(float), 3)

    title_map = {
        'negative': f"{model_name} - Typical Negative Case Force Strip [{output_space}]",
        'positive': f"{model_name} - High-Risk Positive Case Force Strip [{output_space}]",
    }
    title_text = title_map.get(sample_type, f"{model_name} - Force Strip [{output_space}]")
    label_text = "True: Positive" if true_label == 1 else "True: Negative"

    # matplotlib 静态 Force Plot
    fig = plt.figure(figsize=(26, 12))
    try:
        shap.force_plot(
            base_value_plot,
            display_shap_plot,
            feature_names=display_features,
            matplotlib=True,
            show=False,
            contribution_threshold=0.05,
        )
        plt.subplots_adjust(top=0.80, bottom=0.18, left=0.02, right=0.98)
        fig.suptitle(title_text, fontsize=16, fontweight='bold', y=0.98)
        info_text = (
            f"{label_text} | Pred(p): {pred_prob:.3f}   [output space: {output_space}]\n"
            f"Base: {base_repr}\n"
            f"Final: {final_repr}\n"
            f"贡献图例: 红色(→|) 推高预测值;  蓝色(|←) 推低预测值;  条宽 = |SHAP 贡献|"
        )
        fig.text(0.99, 0.96,
                 info_text,
                 fontsize=11, fontweight='bold', va='top', ha='right',
                 bbox=dict(boxstyle='round,pad=0.4', facecolor='wheat', alpha=0.7,
                           edgecolor='gray', linewidth=1))
        out_path = os.path.join(output_dir,
                                f"{model_name}_force_strip_{sample_type}_sample{sample_idx}_{output_space}.JPG")
        plt.savefig(out_path, dpi=300, bbox_inches='tight', facecolor='white')
        plt.close()
        print(f"  ✓ Force Strip ({sample_type}, {output_space}) 已保存: {out_path}")
        # 同时生成 HTML 版本（可交互，旋转90度后截图可作纵向版）
        try:
            html_path = os.path.join(output_dir,
                                     f"{model_name}_force_strip_{sample_type}_sample{sample_idx}_{output_space}.html")
            force_obj = shap.force_plot(
                base_value_plot, display_shap_plot,
                feature_names=display_features, link=link_mode, show=False
            )
            shap.save_html(html_path, force_obj)
            print(f"  ✓ Force Strip ({sample_type}, {output_space}) HTML 已保存: {html_path}")
        except Exception as e_html:
            print(f"  ⚠ Force Strip ({sample_type}, {output_space}) HTML 生成失败: {e_html}")
        return True
    except Exception as e:
        plt.close()
        print(f"  ⚠ Force Strip ({sample_type}, {output_space}) matplotlib 版本失败: {e}")
        return False


# ==================== 模态（面/舌/脉）分组 ====================
# 严格前缀 → 模态 映射（与用户确认的中医特征模态一致）
# 顺序很重要：长前缀优先，避免 't' 误匹配到 TB/TC，'w' 误匹配 wholecolor
MODALITY_PREFIX_MAP = [
    ('面', ['color', 'lipcolor', 'wholecolor', 'GLCM']),
    ('舌', ['TB', 'TC', 'Per']),
    ('脉', ['h', 'w', 't']),
]


def assign_modality(feature_name):
    """严格前缀匹配：返回 '面' / '舌' / '脉'；不匹配则返回 None。
    匹配顺序：长前缀先（color/lipcolor/wholecolor/GLCM/TB/TC/Per），短前缀后（h/w/t）。
    """
    name = str(feature_name)
    for modality, prefixes in MODALITY_PREFIX_MAP:
        for p in prefixes:
            if name == p or name.startswith(p):
                return modality
    return None


def validate_modality_assignment(feature_names):
    """严格校验：所有特征必须匹配到某个模态。否则抛 ValueError 列出所有未匹配项。"""
    unmatched = []
    for fname in feature_names:
        if assign_modality(fname) is None:
            unmatched.append(str(fname))
    if unmatched:
        sample = unmatched[:20]
        raise ValueError(
            f"模态映射校验失败：{len(unmatched)} 个特征未匹配到任何模态（面/舌/脉）。\n"
            f"前 20 个未匹配特征: {sample}\n"
            f"已知前缀映射: {MODALITY_PREFIX_MAP}\n"
            f"请补充 MODALITY_PREFIX_MAP 或核对数据源。"
        )


def summarize_shap_by_modality(shap_vals, feature_names, output_dir, output_space='margin'):
    """按面/舌/脉三模态汇总 SHAP 贡献。
    输出：
    - shap_modality_summary.csv：每个模态的统计指标
    - shap_modality_bar.png：Mean|SHAP| 横向条形图
    - shap_modality_stacked_per_sample.png：每样本 SHAP 按模态堆叠（前 100 样本）
    """
    os.makedirs(output_dir, exist_ok=True)
    # 严格校验
    validate_modality_assignment(feature_names)

    shap_arr = np.asarray(shap_vals)
    if shap_arr.ndim == 1:
        shap_arr = shap_arr.reshape(1, -1)
    n_samples, n_features = shap_arr.shape
    if n_features != len(feature_names):
        raise ValueError(f"SHAP 列数({n_features}) ≠ 特征数({len(feature_names)})")

    # 模态 → 列索引
    modality_cols = {'面': [], '舌': [], '脉': []}
    for i, fname in enumerate(feature_names):
        m = assign_modality(fname)
        modality_cols[m].append(i)

    # 统计表
    rows = []
    for m in ['面', '舌', '脉']:
        cols = modality_cols[m]
        if not cols:
            continue
        sub = shap_arr[:, cols]  # (n_samples, n_modality_features)
        abs_sub = np.abs(sub)
        rows.append({
            'Modality': m,
            'Mean|SHAP|': float(abs_sub.mean()),
            'Sum|SHAP|': float(abs_sub.sum()),
            'Mean SHAP': float(sub.mean()),
            'Max|SHAP|': float(abs_sub.max()),
            'N_Features': len(cols),
        })
    df = pd.DataFrame(rows)
    csv_path = os.path.join(output_dir, 'shap_modality_summary.csv')
    df.to_csv(csv_path, index=False, encoding='utf-8-sig')
    print(f"  ✓ 模态汇总 CSV 已保存: {csv_path}")
    print(df.to_string(index=False))

    # 条形图：Mean|SHAP| 按模态
    fig, ax = plt.subplots(figsize=(10, 6))
    modality_order = df['Modality'].tolist()
    means = df['Mean|SHAP|'].tolist()
    colors = {'面': '#E74C3C', '舌': '#3498DB', '脉': '#2ECC71'}
    bar_colors = [colors.get(m, '#888888') for m in modality_order]
    bars = ax.barh(modality_order, means, color=bar_colors, edgecolor='black', linewidth=1)
    for bar, val in zip(bars, means):
        ax.text(val + max(means) * 0.01, bar.get_y() + bar.get_height() / 2,
                f'{val:.4f}', va='center', fontsize=11, fontweight='bold')
    ax.set_xlabel('Mean |SHAP|', fontsize=12)
    ax.set_title(f'SHAP Modality Summary [{output_space}]', fontsize=14, fontweight='bold')
    ax.invert_yaxis()
    ax.grid(axis='x', linestyle='--', alpha=0.5)
    plt.tight_layout()
    bar_path = os.path.join(output_dir, 'shap_modality_bar.png')
    plt.savefig(bar_path, dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"  ✓ 模态条形图已保存: {bar_path}")

    # 堆叠图：每样本 SHAP 按模态求和（前 100 样本）
    n_show = min(100, n_samples)
    modality_sums = np.zeros((n_show, 3))
    for j, m in enumerate(['面', '舌', '脉']):
        cols = modality_cols[m]
        if cols:
            modality_sums[:, j] = shap_arr[:n_show, cols].sum(axis=1)
    fig, ax = plt.subplots(figsize=(14, 6))
    bottom = np.zeros(n_show)
    x_idx = np.arange(n_show)
    for j, m in enumerate(['面', '舌', '脉']):
        ax.bar(x_idx, modality_sums[:, j], bottom=bottom,
               label=m, color=colors[m], edgecolor='black', linewidth=0.5)
        bottom += modality_sums[:, j]
    ax.axhline(0, color='black', linewidth=1, linestyle='-')
    ax.set_xlabel('Sample Index', fontsize=12)
    ax.set_ylabel('Summed SHAP (per modality)', fontsize=12)
    ax.set_title(f'Per-sample SHAP by Modality (first {n_show} samples) [{output_space}]',
                 fontsize=14, fontweight='bold')
    ax.legend(loc='best', fontsize=11)
    ax.grid(axis='y', linestyle='--', alpha=0.5)
    plt.tight_layout()
    stacked_path = os.path.join(output_dir, 'shap_modality_stacked_per_sample.png')
    plt.savefig(stacked_path, dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"  ✓ 模态堆叠图已保存: {stacked_path}")

    return df


def plot_shap_values(model, X_train, feature_names, model_name, output_dir="results", sample_size=200, y_train=None, n_background=50):
    """生成 SHAP 分析（单输出空间，普通版本）。
    - 树模型（XGBoost/LightGBM/RF/GBDT）：TreeExplainer (log-odds 空间)
    - 非树模型（SVM/KNN/MLP/ANN/LR）：KernelExplainer，输出空间 = predict_proba[:,1]（概率空间）
    - 背景：优先 shap.kmeans(n=n_background)，失败回退随机抽样
    - 产物保存到 shap_plots/，并做一份模态汇总
    """
    os.makedirs(output_dir, exist_ok=True)
    shap_dir = os.path.join(output_dir, "shap_plots")
    os.makedirs(shap_dir, exist_ok=True)
    try:
        # 模态校验（最先做，失败早退出，避免算完 SHAP 才发现问题）
        validate_modality_assignment(feature_names)

        # 提取preprocessor
        if hasattr(model, 'named_steps') and 'preprocessor' in model.named_steps:
            preproc = model.named_steps['preprocessor']
        else:
            preproc = Pipeline([('imputer', SimpleImputer(strategy='median')), ('scaler', StandardScaler())])
            preproc.fit(X_train)
        X_processed = preproc.transform(X_train)
        X_orig = X_train.values if hasattr(X_train, 'values') else X_train
        # 同步对齐 y_train（用于选择典型阴性/阳性病例做 Force Strip）
        y_arr = None
        if y_train is not None:
            y_arr = np.asarray(y_train).ravel()
            if len(y_arr) != X_orig.shape[0]:
                print(f"  ⚠ y_train 长度({len(y_arr)}) ≠ X_train 行数({X_orig.shape[0]})，Force Strip 将基于预测概率选择样本")
                y_arr = None
        if X_processed.shape[0] > sample_size:
            idx = RNG.choice(X_processed.shape[0], sample_size, replace=False)
            X_sample = X_processed[idx]
            X_orig_sample = X_orig[idx]
            y_sample = y_arr[idx] if y_arr is not None else None
        else:
            X_sample = X_processed
            X_orig_sample = X_orig
            y_sample = y_arr
        model_est = model.named_steps['model'] if 'model' in model.named_steps else model
        est_type_name = type(model_est).__name__

        is_tree = isinstance(model_est, (xgb.XGBClassifier, lgb.LGBMClassifier,
                                          RandomForestClassifier, GradientBoostingClassifier))

        # ============ 计算单一输出空间的 SHAP ============
        # shap_results: list of (ospace, shap_vals, exp_val)；树→logodds，非树→probability
        shap_results = []

        if is_tree:
            print(f"  [SHAP] {est_type_name}: 使用 TreeExplainer (log-odds)")
            explainer = shap.TreeExplainer(model_est)
            shap_vals_raw = explainer.shap_values(X_sample)
            if isinstance(shap_vals_raw, list):
                shap_vals = shap_vals_raw[1] if len(shap_vals_raw) == 2 else shap_vals_raw[0]
            else:
                shap_vals = shap_vals_raw
            if len(shap_vals.shape) == 1:
                shap_vals = shap_vals.reshape(-1, 1)
            exp_val = explainer.expected_value
            if isinstance(exp_val, (list, np.ndarray)):
                exp_val = float(np.asarray(exp_val).ravel()[-1])
            else:
                exp_val = float(exp_val)
            shap_results.append(('logodds', shap_vals, exp_val))
        else:
            # 非树模型：单一输出空间（概率空间，普通版本）
            if not hasattr(model_est, 'predict_proba'):
                raise RuntimeError(f"{est_type_name} 不支持 predict_proba，无法构造连续输出")
            ospace = 'probability'
            f_continuous = lambda X: np.asarray(model_est.predict_proba(X))[:, 1]

            # 背景数据：增大代表性（kmeans → 失败回退随机抽样）
            n_bg = min(n_background, X_sample.shape[0])
            background = None
            bg_mode = None
            try:
                background = shap.kmeans(X_sample, min(n_bg, X_sample.shape[0]))
                bg_mode = f"shap.kmeans(n={n_bg})"
            except Exception as e_bg:
                print(f"  [SHAP] shap.kmeans({n_bg}) 失败({e_bg})，回退到随机抽样背景")
                bg_idx = RNG.choice(X_sample.shape[0], min(n_bg, X_sample.shape[0]), replace=False)
                background = X_sample[bg_idx]
                bg_mode = f"random subset (n={n_bg})"
            print(f"  [SHAP] 背景数据: {bg_mode}")

            print(f"\n  [SHAP] === 输出空间: {ospace} ===")
            print(f"  [SHAP] 计算 SHAP 值中（样本数={X_sample.shape[0]}, "
                  f"特征数={X_sample.shape[1]}, 可能耗时 3-5 分钟）...")
            explainer = shap.KernelExplainer(f_continuous, background)
            shap_vals_raw = explainer.shap_values(X_sample)
            if isinstance(shap_vals_raw, list):
                shap_vals = shap_vals_raw[1] if len(shap_vals_raw) == 2 else shap_vals_raw[0]
            else:
                shap_vals = np.asarray(shap_vals_raw)
            if len(shap_vals.shape) == 1:
                shap_vals = shap_vals.reshape(-1, 1)
            exp_val = explainer.expected_value
            if isinstance(exp_val, (list, np.ndarray)):
                exp_val = float(np.asarray(exp_val).ravel()[-1])
            else:
                exp_val = float(exp_val)
            print(f"  [SHAP] [{ospace}] base value = {exp_val:.4f}, SHAP shape = {shap_vals.shape}")
            shap_results.append((ospace, shap_vals, exp_val))

        # ============ 渲染每个输出空间 ============
        for ospace, shap_vals, exp_val in shap_results:
            # 单输出空间：树模型放 shap_dir 根目录；非树模型放 shap_dir/{ospace} 子目录
            render_dir = os.path.join(shap_dir, ospace) if not is_tree else shap_dir
            os.makedirs(render_dir, exist_ok=True)
            print(f"\n  [SHAP] 渲染 [{ospace}] -> {render_dir}")

            # Beeswarm
            plt.figure(figsize=(14, 10))
            shap.summary_plot(shap_vals, X_sample, feature_names=feature_names, show=False)
            plt.title(f"{model_name} - SHAP Beeswarm [{ospace}]")
            plt.tight_layout()
            plt.savefig(os.path.join(render_dir, f"{model_name}_shap_summary_{ospace}.JPG"),
                        dpi=300, bbox_inches='tight')
            plt.close()

            # Bar plot
            plt.figure(figsize=(12, 8))
            shap.summary_plot(shap_vals, X_sample, feature_names=feature_names,
                              plot_type="bar", show=False)
            plt.title(f"{model_name} - SHAP Mean|Value| [{ospace}]")
            plt.tight_layout()
            plt.savefig(os.path.join(render_dir, f"{model_name}_shap_bar_plot_{ospace}.JPG"),
                        dpi=300, bbox_inches='tight')
            plt.close()

            # Violin plot
            plt.figure(figsize=(14, 10))
            shap.summary_plot(shap_vals, X_sample, feature_names=feature_names,
                              plot_type="violin", show=False)
            plt.title(f"{model_name} - SHAP Violin [{ospace}]")
            plt.tight_layout()
            plt.savefig(os.path.join(render_dir, f"{model_name}_shap_violin_summary_{ospace}.JPG"),
                        dpi=300, bbox_inches='tight')
            plt.close()

            # Force Strip: 典型阴性 + 高风险阳性
            y_pred_proba_force = (model_est.predict_proba(X_sample)[:, 1]
                                  if hasattr(model_est, 'predict_proba') else None)
            y_true_for_force = y_sample
            if y_pred_proba_force is None:
                print("  ⚠ 模型不支持 predict_proba，跳过 Force Strip")
            else:
                if y_true_for_force is None:
                    print("  ℹ 未传入 y_train，使用预测概率硬分类作为伪标签选择典型病例")
                    y_true_for_force = (y_pred_proba_force >= 0.5).astype(int)
                neg_idx = _select_force_case_index(y_true_for_force, y_pred_proba_force,
                                                    target_label=0, p_lo=0.05, p_hi=0.20)
                if neg_idx is not None:
                    plot_force_strip_sample(model_name, neg_idx, y_true_for_force, y_pred_proba_force,
                                            X_orig_sample, shap_vals, exp_val, feature_names,
                                            output_dir=render_dir, sample_type='negative',
                                            output_space=ospace, top_k=7)
                else:
                    print("  ⚠ 样本中无阴性病例，跳过典型阴性 Force Strip")
                pos_idx = _select_force_case_index(y_true_for_force, y_pred_proba_force,
                                                    target_label=1, p_lo=0.80, p_hi=0.95)
                if pos_idx is not None:
                    plot_force_strip_sample(model_name, pos_idx, y_true_for_force, y_pred_proba_force,
                                            X_orig_sample, shap_vals, exp_val, feature_names,
                                            output_dir=render_dir, sample_type='positive',
                                            output_space=ospace, top_k=7)
                else:
                    print("  ⚠ 样本中无阳性病例，跳过高风险阳性 Force Strip")

            # Heatmap
            plt.figure(figsize=(14, 10))
            shap.plots.heatmap(shap.Explanation(values=shap_vals, base_values=float(exp_val),
                                                data=X_orig_sample, feature_names=feature_names),
                               show=False, max_display=20)
            plt.title(f"{model_name} - SHAP Heatmap [{ospace}]")
            plt.tight_layout()
            plt.savefig(os.path.join(render_dir, f"{model_name}_shap_heatmap_{ospace}.JPG"),
                        dpi=300, bbox_inches='tight')
            plt.close()

            # Top 3 scatter plots
            mean_abs = np.mean(np.abs(shap_vals), axis=0)
            top3_idx = np.argsort(mean_abs)[::-1][:3]
            y_pred_proba = (model_est.predict_proba(X_sample)[:, 1]
                            if hasattr(model_est, 'predict_proba') else None)
            for idx_ft in top3_idx:
                ft_name = feature_names[idx_ft]
                ft_vals = X_orig_sample[:, idx_ft]
                shap_ft = shap_vals[:, idx_ft]
                plt.figure(figsize=(12, 8))
                if y_pred_proba is not None:
                    sc = plt.scatter(ft_vals, shap_ft, c=y_pred_proba, cmap='coolwarm',
                                     alpha=0.7, s=50, edgecolors='black')
                    plt.colorbar(sc, label='Predicted Probability')
                else:
                    plt.scatter(ft_vals, shap_ft, alpha=0.7, s=50, edgecolors='black')
                plt.axhline(0, color='black', linestyle='--', alpha=0.8, lw=2)
                plt.xlabel(f'{ft_name} (Value)')
                plt.ylabel(f'SHAP Value for {ft_name}')
                plt.title(f"{model_name} - SHAP Scatter: {ft_name} [{ospace}]")
                plt.grid(True, linestyle='--', alpha=0.5)
                try:
                    z = np.polyfit(ft_vals, shap_ft, 1)
                    p = np.poly1d(z)
                    plt.plot(np.sort(ft_vals), p(np.sort(ft_vals)), 'r-', lw=2, alpha=0.8,
                             label=f'Trend: slope={z[0]:.4f}')
                    plt.legend()
                except:
                    pass
                plt.tight_layout()
                # 特征名可能含 '/' 等文件名非法字符，替换后用于命名
                ft_name_safe = str(ft_name).replace('/', '_').replace('\\', '_').replace(':', '_')
                plt.savefig(os.path.join(render_dir,
                                         f"{model_name}_shap_scatter_{ft_name_safe}_{ospace}.JPG"),
                            dpi=300, bbox_inches='tight')
                plt.close()

            # Decision plot
            plt.figure(figsize=(14, 10))
            shap.decision_plot(exp_val, shap_vals, features=X_orig_sample,
                               feature_names=feature_names, show=False)
            plt.title(f"{model_name} - SHAP Decision Plot [{ospace}]")
            plt.tight_layout()
            plt.savefig(os.path.join(render_dir,
                                     f"{model_name}_shap_decision_plot_{ospace}.JPG"),
                        dpi=300, bbox_inches='tight')
            plt.close()

            # 模态汇总（按面/舌/脉三模态归并 SHAP 贡献）
            modality_dir = os.path.join(render_dir, "modality_summary")
            try:
                summarize_shap_by_modality(shap_vals, feature_names, modality_dir,
                                           output_space=ospace)
            except Exception as e_mod:
                print(f"  ⚠ 模态汇总失败 [{ospace}]: {e_mod}")

        print(f"\n✅ 所有SHAP可视化已保存至: {shap_dir}")
        # 返回本步已算出的 SHAP 值（抽样子集，默认 200 样本），供 main() 写入 checkpoint 复用，
        # 避免再在全训练集上二次重算（非树模型 KernelExplainer 全量需数小时）。
        shap_values_out = {ospace: {'values': sv, 'expected_value': ev}
                           for ospace, sv, ev in shap_results}
        return shap_values_out
    except Exception as e:
        print(f"❌ SHAP分析失败: {str(e)}")
        with open(os.path.join(output_dir, "shap_error_log.txt"), 'w') as f:
            f.write(str(e) + "\n" + traceback.format_exc())
        return None


def find_best_model_for_shap(best_models, cv_results, X_train, y_train, feature_names):
    """始终返回 CV AUC 最高的模型用于 SHAP 分析（与最终模型保持一致，避免方法学不匹配）。
    对于非树模型（SVM/KNN/MLP/ANN/LR），在 plot_shap_values 内部会改用 KernelExplainer。
    """
    sorted_cv = cv_results.sort_values('AUC', ascending=False).reset_index(drop=True)
    best_name = sorted_cv.iloc[0]['Model']
    best_auc = sorted_cv.iloc[0]['AUC']
    best_model = best_models[best_name]['pipeline']
    est_type = type(best_model.named_steps['model']).__name__
    print(f"\n[SHAP] 选择 CV 最佳模型: {best_name} ({est_type}, CV AUC={best_auc:.4f})")
    print(f"[SHAP] 解释器将与最终模型保持一致，避免方法学不匹配")
    return best_model, best_name


# 兼容旧调用名
find_suitable_model_for_shap = find_best_model_for_shap


# ==================== Checkpoint（SHAP-only 模式复用训练结果） ====================
def save_shap_checkpoint(best_models, cv_results, feature_names, output_dir,
                          shap_values_full=None):
    """保存训练好的最佳模型集合 + CV 结果 + 特征名，供 SHAP-only 模式复用。
    文件：{output_dir}/shap_checkpoint.pkl
    可选 shap_values_full: 抽样子集（默认 200 样本）SHAP 值（dict[ospace] = {'values', 'expected_value'}），
        来自 plot_shap_values 的返回值，供后续调用/复用。
    """
    os.makedirs(output_dir, exist_ok=True)
    checkpoint_path = os.path.join(output_dir, "shap_checkpoint.pkl")
    checkpoint = {
        'best_models': best_models,
        'cv_results': cv_results,
        'feature_names': list(feature_names),
        'random_seed': RANDOM_SEED,
    }
    if shap_values_full is not None:
        checkpoint['shap_values_full'] = shap_values_full
    joblib.dump(checkpoint, checkpoint_path)
    size_mb = os.path.getsize(checkpoint_path) / (1024 * 1024)
    print(f"\n[Checkpoint] 已保存训练结果 → {checkpoint_path} ({size_mb:.1f} MB)")
    if shap_values_full is not None:
        for ospace, info in shap_values_full.items():
            print(f"[Checkpoint]   SHAP/{ospace}: shape={info['values'].shape}, "
                  f"expected_value={info['expected_value']:.4f}")
    print(f"[Checkpoint] 下次可用 `python {__file__.split(os.sep)[-1]} --shap-only` 直接跑 SHAP")


def load_shap_checkpoint(output_dir):
    """加载 SHAP checkpoint"""
    checkpoint_path = os.path.join(output_dir, "shap_checkpoint.pkl")
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(
            f"未找到 checkpoint: {checkpoint_path}\n"
            f"请先完整运行 main() 一次以生成 checkpoint，再使用 --shap-only 模式"
        )
    checkpoint = joblib.load(checkpoint_path)
    print(f"[Checkpoint] 已加载 {checkpoint_path}")
    return checkpoint['best_models'], checkpoint['cv_results'], checkpoint['feature_names']


# ==================== best / best_tree 双模型识别与 web_app artifact 输出 ====================
TREE_MODELS = (xgb.XGBClassifier, lgb.LGBMClassifier,
               RandomForestClassifier, GradientBoostingClassifier)


def identify_best_and_best_tree(best_models, cv_results):
    """返回 (best_name, best_tree_name_or_None)。
    best = CV AUC 最高；best_tree = 当 best 非树时，取 CV AUC 最高的树模型；best 本身是树则返回 None。
    """
    sorted_cv = cv_results.sort_values('AUC', ascending=False).reset_index(drop=True)
    best_name = sorted_cv.iloc[0]['Model']
    best_est = best_models[best_name]['pipeline'].named_steps['model']
    if isinstance(best_est, TREE_MODELS):
        return best_name, None
    tree_rows = sorted_cv[sorted_cv['Model'].apply(
        lambda n: isinstance(best_models[n]['pipeline'].named_steps['model'], TREE_MODELS))]
    best_tree_name = tree_rows.iloc[0]['Model'] if len(tree_rows) else None
    return best_name, best_tree_name


def save_webapp_artifact(model, X_train, y_train, feature_names,
                          model_name, cv_auc, output_dir):
    """输出 web_app 兼容的 4 文件 artifact：
        best_model.joblib / calibrator.joblib / shap_background.npy / model_metadata.json
    与 web_app/model_manager.save_model_artifact 格式一致，供 web_app/saved_model/ 直接加载使用。
    """
    os.makedirs(output_dir, exist_ok=True)
    # 1) pipeline (preprocessor + model)
    joblib.dump(model, os.path.join(output_dir, 'best_model.joblib'))
    # 2) calibrator (在训练集 predict_proba 上拟合，复用主脚本现有 select_calibration_method)
    y_proba = model.predict_proba(X_train)[:, 1]
    y_arr = y_train.values if hasattr(y_train, 'values') else np.asarray(y_train)
    cal_method, _ = select_calibration_method(len(y_arr), int(np.sum(y_arr)))
    if cal_method == 'isotonic':
        calibrator = IsotonicRegression(out_of_bounds='clip', increasing=True)
    else:
        calibrator = LogisticRegression(C=1e10, solver='lbfgs', max_iter=1000,
                                         random_state=RANDOM_SEED)
    calibrator.fit(y_proba.reshape(-1, 1), y_arr)
    joblib.dump(calibrator, os.path.join(output_dir, 'calibrator.joblib'))
    # 3) SHAP background (k-means 替代随机抽样，与 plot_shap_values 一致)
    preproc = model.named_steps['preprocessor'] if hasattr(model, 'named_steps') else None
    if preproc is None:
        preproc = Pipeline([('imputer', SimpleImputer(strategy='median')),
                            ('scaler', StandardScaler())])
        preproc.fit(X_train)
    rng = np.random.RandomState(RANDOM_SEED)
    bg_idx = rng.choice(len(X_train), min(100, len(X_train)), replace=False)
    shap_bg = preproc.transform(X_train.iloc[bg_idx])
    np.save(os.path.join(output_dir, 'shap_background.npy'), shap_bg)
    # 4) metadata
    model_est = model.named_steps['model'] if hasattr(model, 'named_steps') else model
    feat_imp = None
    if hasattr(model_est, 'feature_importances_'):
        feat_imp = dict(zip(feature_names, model_est.feature_importances_.tolist()))
    elif hasattr(model_est, 'coef_'):
        coef = model_est.coef_
        imp = np.abs(coef).mean(axis=0) if len(coef.shape) > 1 else np.abs(coef)
        feat_imp = dict(zip(feature_names, imp.tolist()))
    metadata = {
        'model_name': model_name,
        'cv_auc': float(cv_auc),
        'feature_names': list(feature_names),
        'n_features': len(feature_names),
        'risk_thresholds': {'low': 0.40, 'high': 0.80},
        'calibration_method': cal_method,
        'training_samples': len(X_train),
        'random_seed': RANDOM_SEED,
        'feature_importance': feat_imp,
    }
    with open(os.path.join(output_dir, 'model_metadata.json'), 'w', encoding='utf-8') as f:
        json.dump(metadata, f, ensure_ascii=False, indent=2)
    print(f"  [WebApp Artifact] 已保存 → {os.path.abspath(output_dir)}")


def deploy_artifacts_to_webapp(prod_dir, output_dir):
    """把 production_artifact/{best,best_tree} 部署到 web_app/saved_models/{YYYYMMDD_HHMM}_{架构名}/。
    命名与 0504-9-LightGBM-NaN-Robust.py 归档脚本一致；web_app/app.py 的 scan_model_dirs 会自动发现为下拉可选项。
    时间戳优先取 OUTPUT_DIR 末尾的本次运行时间（精确到分），保证与 results 目录一致。
    返回部署成功的目录列表。"""
    script_dir = os.path.dirname(os.path.abspath(__file__))
    saved_models_dir = os.path.join(script_dir, 'web_app', 'saved_models')
    m = re.search(r'(\d{8}_\d{6})', os.path.basename(output_dir))
    deploy_ts = m.group(1)[:13] if m else datetime.now().strftime('%Y%m%d_%H%M')
    deployed = []
    for sub in ['best', 'best_tree']:
        src = os.path.join(prod_dir, sub)
        if not os.path.isdir(src):
            continue
        meta_path = os.path.join(src, 'model_metadata.json')
        if not os.path.exists(meta_path):
            continue
        with open(meta_path, 'r', encoding='utf-8') as f:
            model_name = json.load(f).get('model_name', sub)
        dst = os.path.join(saved_models_dir, f'{deploy_ts}_{model_name}')
        os.makedirs(dst, exist_ok=True)
        for fname in ['best_model.joblib', 'calibrator.joblib',
                      'shap_background.npy', 'model_metadata.json']:
            s = os.path.join(src, fname)
            if os.path.exists(s):
                shutil.copy2(s, os.path.join(dst, fname))
        deployed.append(dst)
        print(f"  [部署] {sub} → web_app/saved_models/{deploy_ts}_{model_name}/")
    if deployed:
        print(f"  [部署] 共 {len(deployed)} 个模型已写入 web_app/saved_models/，webapp 下拉框可直接选择")
    return deployed


def compute_shap_values_full(*args, **kwargs):
    """已废弃：原在全训练集（~1000+ 样本）上独立重算 SHAP 以写入 checkpoint，
    对非树模型（KernelExplainer）耗时可达数小时，且结果仅落盘未被下游读取。
    现改为复用 plot_shap_values 在 200 样本抽样上已算出的 SHAP 值（见 main()）。
    保留占位以防外部调用报 NameError。"""
    raise NotImplementedError(
        "compute_shap_values_full 已废弃；SHAP 值现由 plot_shap_values 返回并写入 checkpoint。"
    )


# ==================== 论文 Table 2/3 + 补充风险分层 CI 输出（与 0803-CI 脚本同口径） ====================
def _fold_mean_t_ci(fold_values, alpha=0.05):
    """折均值 t-based CI: mean ± t(1-α/2, n-1) × SE"""
    from scipy.stats import t as student_t
    arr = np.asarray(fold_values, dtype=float)
    n = len(arr)
    if n < 2:
        return float(arr.mean()) if n == 1 else float('nan'), float('nan'), float('nan')
    mean = float(arr.mean())
    se = float(arr.std(ddof=1) / np.sqrt(n))
    tval = student_t.ppf(1 - alpha / 2, df=n - 1)
    return mean, mean - tval * se, mean + tval * se


def _fmt_ci(point, lo, hi, decimals=3):
    if pd.isna(point):
        return "NA"
    if pd.isna(lo) or pd.isna(hi):
        return f"{point:.{decimals}f}"
    return f"{point:.{decimals}f} ({lo:.{decimals}f}-{hi:.{decimals}f})"


def _wilson_ci(k, n, alpha=0.05):
    """二项分布 Wilson score 区间。"""
    if n == 0:
        return float('nan'), float('nan'), float('nan')
    z = norm.ppf(1 - alpha / 2)
    p = k / n
    denom = 1 + z**2 / n
    center = (p + z**2 / (2 * n)) / denom
    half = z * np.sqrt(p * (1 - p) / n + z**2 / (4 * n**2)) / denom
    return p, max(0.0, center - half), min(1.0, center + half)


def compute_table2_dual_ci(best_models, X, y, output_dir, n_folds=10, n_bootstrap=1000):
    """对每个模型计算 CV 双 CI（折 t-based + OOF pooled），输出 Table2_CV_dual_CI.csv。

    折 t-CI: 直接从 grid_search.cv_results_[split{i}_test_{metric}][best_index] 取折值算
    OOF CI: 用 best params 重做一轮 10 折 CV 拿 OOF 预测，AUC=DeLong, 其他=bootstrap
    """
    from sklearn.base import clone as sklearn_clone
    from sklearn.model_selection import StratifiedKFold
    print(f"\n[Table 2 CI] 计算双 CI（折 t-CI + OOF CI），9 模型 × 10 折重做 OOF...")
    kfold = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=RANDOM_SEED)
    metric_list = ['AUC', 'AP', 'Accuracy', 'Precision', 'Sensitivity', 'Specificity', 'F1', 'Brier']
    rows = []
    for name, info in best_models.items():
        cv_results = info['cv_results']
        best_idx = info['best_index']
        # (a) 折 t-CI: 从 cv_results_ 提取每折值
        fold_metrics = {m: [] for m in metric_list}
        for m in metric_list:
            for fold_i in range(n_folds):
                col = f'split{fold_i}_test_{m}'
                if col in cv_results:
                    fold_metrics[m].append(cv_results[col][best_idx])
        # Brier 不在 grid_search 评分里，需要从 OOF 重算
        # (b) OOF CI: 用 best params 重做 10 折 CV
        best_pipe = info['pipeline']
        tuned_est = sklearn_clone(best_pipe.named_steps['model'])
        oof_true, oof_proba, oof_pred = [], [], []
        brier_folds = []
        for tr_idx, va_idx in kfold.split(X, y):
            fold_pre = Pipeline([('imputer', SimpleImputer(strategy='median')),
                                  ('scaler', StandardScaler())])
            fold_model = sklearn_clone(tuned_est)
            X_tr = X.iloc[tr_idx]; y_tr = y.iloc[tr_idx]
            X_va = X.iloc[va_idx]; y_va = y.iloc[va_idx]
            X_tr_pp = fold_pre.fit_transform(X_tr)
            X_va_pp = fold_pre.transform(X_va)
            fold_model.fit(X_tr_pp, y_tr)
            y_proba = fold_model.predict_proba(X_va_pp)[:, 1]
            y_pred = (y_proba >= 0.5).astype(int)
            brier_folds.append(brier_score_loss(y_va, y_proba))
            oof_true.extend(y_va.values if hasattr(y_va, 'values') else y_va)
            oof_proba.extend(y_proba)
            oof_pred.extend(y_pred)
        fold_metrics['Brier'] = brier_folds
        # 折 t-CI
        fold_ci = {m: _fold_mean_t_ci(fold_metrics[m]) for m in metric_list}
        # OOF pooled CI
        oof_true_arr = np.asarray(oof_true)
        oof_proba_arr = np.asarray(oof_proba)
        oof_pred_arr = np.asarray(oof_pred)
        ci_ext, _, _ = build_metrics_with_ci(oof_true_arr, oof_pred_arr, oof_proba_arr,
                                              n_bootstrap=n_bootstrap)
        # 组装行
        row = {'Model': name}
        for m in metric_list:
            mean, lo, hi = fold_ci[m]
            row[m] = round(mean, 4)
            row[f'{m}_fold_t_CI'] = _fmt_ci(mean, lo, hi)
            row[f'{m}_fold_t_Lower'] = lo
            row[f'{m}_fold_t_Upper'] = hi
            if m == 'AUC':
                row[f'{m}_oof_CI'] = _fmt_ci(ci_ext.get('AUC_Lower', float('nan')),
                                              ci_ext['AUC_Lower'], ci_ext['AUC_Upper'])
                # build_metrics_with_ci 没返回 AUC 点估计，用 oof 直接算
                auc_oof = roc_auc_score(oof_true_arr, oof_proba_arr)
                row[f'{m}_oof_CI'] = _fmt_ci(auc_oof, ci_ext['AUC_Lower'], ci_ext['AUC_Upper'])
                row[f'{m}_oof_Lower'] = ci_ext['AUC_Lower']
                row[f'{m}_oof_Upper'] = ci_ext['AUC_Upper']
            elif m == 'Brier':
                # Brier 点估计从 OOF
                brier_oof = brier_score_loss(oof_true_arr, oof_proba_arr)
                row[f'{m}_oof_CI'] = _fmt_ci(brier_oof, ci_ext.get('Brier_Lower', float('nan')),
                                              ci_ext.get('Brier_Upper', float('nan')))
                row[f'{m}_oof_Lower'] = ci_ext.get('Brier_Lower', float('nan'))
                row[f'{m}_oof_Upper'] = ci_ext.get('Brier_Upper', float('nan'))
            else:
                # 其他指标 Lower/Upper 在 ci_ext 里以 {m}_Lower / {m}_Upper 命名
                lo_key = f'{m}_Lower'; hi_key = f'{m}_Upper'
                point = float(np.mean(fold_metrics[m]))  # 近似点估计；实际 OOF 点估计从 oof 数组算
                # 重新算 OOF 点估计
                if m == 'AP': point_oof = average_precision_score(oof_true_arr, oof_proba_arr)
                elif m == 'Accuracy': point_oof = accuracy_score(oof_true_arr, oof_pred_arr)
                elif m == 'Precision': point_oof = precision_score(oof_true_arr, oof_pred_arr, zero_division=0)
                elif m == 'Sensitivity': point_oof = recall_score(oof_true_arr, oof_pred_arr, zero_division=0)
                elif m == 'Specificity': point_oof = recall_score(oof_true_arr, oof_pred_arr, pos_label=0, zero_division=0)
                elif m == 'F1': point_oof = f1_score(oof_true_arr, oof_pred_arr, zero_division=0)
                else: point_oof = float('nan')
                row[f'{m}_oof_CI'] = _fmt_ci(point_oof, ci_ext.get(lo_key, float('nan')),
                                              ci_ext.get(hi_key, float('nan')))
                row[f'{m}_oof_Lower'] = ci_ext.get(lo_key, float('nan'))
                row[f'{m}_oof_Upper'] = ci_ext.get(hi_key, float('nan'))
        rows.append(row)
        print(f"  {name}: AUC 折 t={row['AUC_fold_t_CI']}, OOF={row['AUC_oof_CI']}")
    df_out = pd.DataFrame(rows)
    out_dir = os.path.join(output_dir, "CI汇总")
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, "Table2_CV_dual_CI.csv")
    df_out.to_csv(out_path, index=False, encoding='utf-8-sig')
    print(f"  ✓ Saved: {out_path}")
    # 论文格式精简版
    pub_cols = ['Model'] + [f'{m}_fold_t_CI' for m in metric_list]
    df_pub = df_out[pub_cols].rename(columns={f'{m}_fold_t_CI': f'{m} (95% CI)' for m in metric_list})
    pub_path = os.path.join(out_dir, "Table2_CV_publication_format.csv")
    df_pub.to_csv(pub_path, index=False, encoding='utf-8-sig')
    print(f"  ✓ Saved: {pub_path}")
    return df_out


def compute_table3_svm_ci(best_models, X_test, y_test, X_ext, y_ext, output_dir, n_bootstrap=1000):
    """对 SVM 提取内部测试 + 外部验证的 8 指标 + 95% CI，输出 Table3 CSV。
    SVM 用 grid search 选出的最优超参（已验证为 C=1.0, gamma=scale, rbf, balanced，与论文锁定值一致）。
    """
    if 'SVM' not in best_models:
        print(f"  ⚠ best_models 里没 SVM，跳过 Table 3")
        return None
    svm_pipe = best_models['SVM']['pipeline']
    rows = []
    for label, X_eval, y_eval in [
        ('Internal test', X_test, y_test),
        ('External evaluation', X_ext, y_ext),
    ]:
        if X_eval is None or y_eval is None or len(X_eval) == 0:
            continue
        y_pred = svm_pipe.predict(X_eval)
        y_proba = svm_pipe.predict_proba(X_eval)[:, 1]
        # 8 指标点估计
        tn, fp, fn, tp = confusion_matrix(y_eval, y_pred).ravel()
        spec = tn / (tn + fp) if (tn + fp) > 0 else 0.0
        metrics = {
            'AUC': roc_auc_score(y_eval, y_proba),
            'AP': average_precision_score(y_eval, y_proba),
            'Accuracy': accuracy_score(y_eval, y_pred),
            'Precision': precision_score(y_eval, y_pred, zero_division=0),
            'Sensitivity': recall_score(y_eval, y_pred, zero_division=0),
            'Specificity': spec,
            'F1': f1_score(y_eval, y_pred, zero_division=0),
            'Brier': brier_score_loss(y_eval, y_proba),
        }
        # 95% CI
        ci_ext, ci_pub, _ = build_metrics_with_ci(y_eval, y_pred, y_proba, n_bootstrap=n_bootstrap)
        row = {'Set': label, 'Model': 'SVM', 'n': len(X_eval)}
        for m, v in metrics.items():
            row[m] = round(v, 4)
            if m == 'AUC':
                row[f'{m}_CI'] = _fmt_ci(v, ci_ext['AUC_Lower'], ci_ext['AUC_Upper'])
                row[f'{m}_Lower'] = ci_ext['AUC_Lower']
                row[f'{m}_Upper'] = ci_ext['AUC_Upper']
            else:
                row[f'{m}_CI'] = ci_pub.get(f'{m} (95% CI)', '')
                row[f'{m}_Lower'] = ci_ext.get(f'{m}_Lower', float('nan'))
                row[f'{m}_Upper'] = ci_ext.get(f'{m}_Upper', float('nan'))
        rows.append(row)
        print(f"  [Table 3] {label} (n={len(X_eval)}): AUC={row['AUC']:.3f} {row['AUC_CI']}")
    df_out = pd.DataFrame(rows)
    out_dir = os.path.join(output_dir, "CI汇总")
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, "Table3_SVM_internal_external_CI.csv")
    df_out.to_csv(out_path, index=False, encoding='utf-8-sig')
    print(f"  ✓ Saved: {out_path}")
    return df_out


def compute_supp_risk_strata_ci(best_model, X_test, y_test, X_ext, y_ext,
                                  low_thr=0.40, high_thr=0.80, output_dir=None):
    """对内部 + 外部验证集按 0.40/0.80 阈值做三层风险分层，输出 Wilson CI + High/Low ratio CI。

    输出 Supp_RiskStrata_Proportions_CI.csv + Supp_RiskStrata_HighToLow_Ratio_CI.csv
    """
    if output_dir is None:
        return None
    out_dir = os.path.join(output_dir, "CI汇总")
    os.makedirs(out_dir, exist_ok=True)
    rows_prop = []
    rows_ratio = []
    for label, X_eval, y_eval in [
        ('Internal test', X_test, y_test),
        ('External evaluation', X_ext, y_ext),
    ]:
        if X_eval is None or y_eval is None or len(X_eval) == 0:
            continue
        y_arr = np.asarray(y_eval).ravel()
        scores = best_model.predict_proba(X_eval)[:, 1]
        low_mask = scores < low_thr
        int_mask = (scores >= low_thr) & (scores < high_thr)
        high_mask = scores >= high_thr
        strata = {
            'Low': (int(low_mask.sum()), int(y_arr[low_mask].sum())),
            'Medium': (int(int_mask.sum()), int(y_arr[int_mask].sum())),
            'High': (int(high_mask.sum()), int(y_arr[high_mask].sum())),
        }
        ci_dict = {}
        for stratum in ['Low', 'Medium', 'High']:
            n, k = strata[stratum]
            p, lo, hi = _wilson_ci(k, n)
            ci_dict[stratum] = (p, lo, hi)
            rows_prop.append({
                'Dataset': label, 'Stratum': stratum, 'n': n, 'LC cases (k)': k,
                'Observed LC proportion': f"{p:.3f}" if not pd.isna(p) else "NA",
                'Observed LC proportion (95% CI)': f"{p:.3f} ({lo:.3f}-{hi:.3f})" if not pd.isna(p) else "NA",
                'n / LC cases': f"{k}/{n}",
            })
        # High/Low ratio + delta-method CI（log 空间）
        p_low, lo_low, hi_low = ci_dict['Low']
        p_high, lo_high, hi_high = ci_dict['High']
        if p_low > 0 and p_high > 0:
            ratio = p_high / p_low
            z = norm.ppf(0.975)
            var_high = max((p_high - lo_high)**2, (hi_high - p_high)**2) / (z**2)
            var_low = max((p_low - lo_low)**2, (hi_low - p_low)**2) / (z**2)
            se_log = np.sqrt(var_high / p_high**2 + var_low / p_low**2)
            log_r = np.log(ratio)
            r_lo, r_hi = np.exp(log_r - z * se_log), np.exp(log_r + z * se_log)
        else:
            ratio, r_lo, r_hi = float('nan'), float('nan'), float('nan')
        rows_ratio.append({
            'Dataset': label,
            'Low prop (95% CI)': f"{p_low:.3f} ({lo_low:.3f}-{hi_low:.3f})" if not pd.isna(p_low) else "NA",
            'High prop (95% CI)': f"{p_high:.3f} ({lo_high:.3f}-{hi_high:.3f})" if not pd.isna(p_high) else "NA",
            'Ratio': f"{ratio:.2f}" if not pd.isna(ratio) else "NA",
            'Approx 95% CI (delta method)': f"({r_lo:.2f}-{r_hi:.2f})" if not pd.isna(r_lo) else "NA",
        })
    df_prop = pd.DataFrame(rows_prop)
    df_ratio = pd.DataFrame(rows_ratio)
    out1 = os.path.join(out_dir, "Supp_RiskStrata_Proportions_CI.csv")
    out2 = os.path.join(out_dir, "Supp_RiskStrata_HighToLow_Ratio_CI.csv")
    df_prop.to_csv(out1, index=False, encoding='utf-8-sig')
    df_ratio.to_csv(out2, index=False, encoding='utf-8-sig')
    print(f"  ✓ Saved: {out1}")
    print(f"  ✓ Saved: {out2}")
    return df_prop, df_ratio


def main_shap_only(data_path, output_dir, target_column='target',
                   sample_size=200, n_background=50):
    """SHAP-only 模式：跳过 9 模型训练与评估，加载 checkpoint 后直接做 SHAP。
    前提：之前已完整运行过 main() 一次，生成了 shap_checkpoint.pkl。
    适用于反复调整 SHAP 可视化参数（force plot、beeswarm、模态汇总等）的场景。
    """
    print("="*60)
    print("SHAP-Only Mode - 跳过训练，仅运行 SHAP 分析")
    print(f"参数: sample_size={sample_size}, n_background={n_background}")
    print("="*60)

    # 加载已训练的模型
    print("\n[Step 1/3] 加载 checkpoint（训练好的 9 模型）...")
    best_models, cv_results, saved_feature_names = load_shap_checkpoint(output_dir)

    # 加载数据（仍需 X_train 用于 SHAP）
    print("\n[Step 2/3] 加载数据并复现训练/测试分割...")
    X, y, feature_names = load_and_preprocess_data(data_path, target_column)
    # 用 checkpoint 的特征顺序，确保列对齐
    if list(feature_names) != list(saved_feature_names):
        print(f"⚠ 数据列顺序与 checkpoint 不一致，按 checkpoint 顺序对齐")
        X = X[saved_feature_names]
        feature_names = saved_feature_names
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=RANDOM_SEED, stratify=y)

    print("\n[Step 3/3] 选择 CV 最佳模型并运行 SHAP...")
    shap_model, shap_name = find_best_model_for_shap(
        best_models, cv_results, X_train, y_train, feature_names)
    shap_model.fit(X_train, y_train)

    plot_shap_values(shap_model, X_train, feature_names, shap_name,
                     output_dir=os.path.join(output_dir, "model_interpretation"),
                     y_train=y_train,
                     sample_size=sample_size, n_background=n_background)
    print(f"\n✅ SHAP 分析完成。结果保存至: {os.path.abspath(os.path.join(output_dir, 'model_interpretation'))}")


# ==================== Main ====================
def main(data_path, external_test_path=None, output_dir="disease_diagnosis_results", target_column='target'):
    print("="*60)
    print("Enhanced Disease Diagnosis Prediction Model - v25 统一字体")
    print("[SHAP 单空间版] 始终对 CV 最佳模型做 SHAP 解释（避免 SVM 最佳却用 LightGBM 解释的方法学不匹配）")
    print("  • 树模型: TreeExplainer (log-odds 空间)")
    print("  • 非树模型（SVM/KNN/MLP/LR）: KernelExplainer，输出空间 = predict_proba[:,1]（概率空间）")
    print("  • SHAP 在 200 样本抽样子集上计算一次，结果同时用于绘图与写入 checkpoint（不再二次全量重算）")
    print("Risk thresholds: Low-risk ≤ 0.40, High-risk > 0.80")
    print("="*60)
    print("\n[Step 1/7] Loading data...")
    X, y, feature_names = load_and_preprocess_data(data_path, target_column)
    print("\n[Step 2/7] Splitting train/test...")
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=RANDOM_SEED, stratify=y)
    save_split_datasets(X_train, y_train, X_test, y_test, feature_names, target_column, output_dir)
    print("\n[Step 3/7] Loading model configs...")
    models = get_model_configurations()
    print(f"Total models: {len(models)}")
    print("\n[Step 4/7] Training models (10-fold CV)...")
    best_models, cv_results, _ = train_and_optimize_models(X_train, y_train, models, output_dir)
    cv_results.to_csv(os.path.join(output_dir, "cv_results_summary.csv"), index=False)
    # 识别 best + best_tree（best 非树时取 CV AUC 最高的树）
    best_name_id, best_tree_name = identify_best_and_best_tree(best_models, cv_results)
    if best_tree_name is None:
        print(f"\n[双模型] best={best_name_id} (是树模型，无需额外保存 best_tree)")
    else:
        best_tree_auc = cv_results[cv_results['Model'] == best_tree_name]['AUC'].values[0]
        print(f"\n[双模型] best={best_name_id}, best_tree={best_tree_name} "
              f"(CV AUC={best_tree_auc:.4f})")
    # 保存 checkpoint 供 SHAP-only 模式复用（避免下次调 SHAP 时重复训练 9 个模型）
    try:
        save_shap_checkpoint(best_models, cv_results, feature_names, output_dir)
    except Exception as e_ckpt:
        print(f"⚠ 保存 checkpoint 失败（不影响主流程）: {e_ckpt}")
    # ===== Table 2 双 CI（折 t-CI + OOF CI）整合到主脚本 =====
    try:
        compute_table2_dual_ci(best_models, X_train, y_train, output_dir)
    except Exception as e_t2:
        print(f"⚠ Table 2 CI 计算失败（不影响主流程）: {e_t2}")
        import traceback as _tb_t2
        _tb_t2.print_exc()
    best_ref = cv_results.sort_values('AUC', ascending=False).iloc[0]['Model']
    print(f"\nReference model for DeLong tests: {best_ref}")
    print("\n[Step 5/7] Evaluating all models on internal test set...")
    internal_metrics, _, _, _ = evaluate_all_models_on_test_set(best_models, X_test, y_test, best_ref, output_dir, phase="Test")
    print("\n[Step 6/7] Selecting and retraining best model...")
    final_model, best_name, _, _ = select_and_retrain_best_model(X_train, y_train, best_models, cv_results)
    test_metrics, y_proba_test = evaluate_model(final_model, X_test, y_test, best_name, output_dir, phase="Test")
    # Calibration
    y_proba_train = final_model.predict_proba(X_train)[:,1]
    plot_calibration_curve_safe(y_train, y_proba_train, y_test, y_proba_test, best_name, output_dir=os.path.join(output_dir, "test_plots"), calibration_method='auto')
    plot_decision_curve(y_test, y_proba_test, best_name, output_dir=os.path.join(output_dir, "test_plots"))
    plot_feature_importance(final_model, feature_names, best_name, output_dir=os.path.join(output_dir, "feature_analysis"))
    # SHAP
    shap_model, shap_name = find_suitable_model_for_shap(best_models, cv_results, X_train, y_train, feature_names)
    shap_model.fit(X_train, y_train)
    shap_values_cached = plot_shap_values(shap_model, X_train, feature_names, shap_name, output_dir=os.path.join(output_dir, "model_interpretation"), y_train=y_train, sample_size=200, n_background=50)
    # 复用上一步（200 样本抽样）已算出的 SHAP 值写入 checkpoint，供后续调用——不再在全训练集上二次重算。
    # 注：缓存的是抽样子集（默认 200 样本）的 SHAP，足以重绘所有图与做后续分析；非树模型若改存全量需数小时。
    if shap_values_cached:
        try:
            save_shap_checkpoint(best_models, cv_results, feature_names, output_dir,
                                  shap_values_full=shap_values_cached)
        except Exception as e_ckpt_shap:
            print(f"⚠ checkpoint 更新 SHAP 值失败（不影响主流程）: {e_ckpt_shap}")
    # ===== web_app 兼容 artifact：best 与 best_tree（best 非树时） =====
    prod_dir = os.path.join(output_dir, "production_artifact")
    best_auc_for_artifact = cv_results[cv_results['Model'] == best_name_id]['AUC'].values[0]
    save_webapp_artifact(final_model, X_train, y_train, feature_names,
                          best_name_id, best_auc_for_artifact,
                          os.path.join(prod_dir, "best"))
    if best_tree_name is not None:
        best_tree_pipe = best_models[best_tree_name]['pipeline']
        best_tree_pipe.fit(X_train, y_train)
        best_tree_auc_for_artifact = cv_results[cv_results['Model'] == best_tree_name]['AUC'].values[0]
        save_webapp_artifact(best_tree_pipe, X_train, y_train, feature_names,
                              best_tree_name, best_tree_auc_for_artifact,
                              os.path.join(prod_dir, "best_tree"))
    # ===== 部署到 web_app/saved_models/{YYYYMMDD_HHMM}_{架构名}/（与 0504-9 归档命名一致） =====
    # 主脚本直接部署，免去再跑 run_training.py 复制；scan_model_dirs 自动发现为下拉可选项
    try:
        deploy_artifacts_to_webapp(prod_dir, output_dir)
    except Exception as e_deploy:
        print(f"⚠ 部署到 web_app/saved_models/ 失败（不影响主流程）: {e_deploy}")
    # Risk stratification
    print("\n[Risk stratification] Computing thresholds on training set...")
    train_preds = pd.DataFrame({'True_Label': y_train.values, 'Predicted_Label': final_model.predict(X_train),
                                'Predicted_Probability': y_proba_train, 'Calibrated_Probability': y_proba_train})
    train_results, thresholds, _, _ = add_risk_stratification_fixed(train_preds, calibrated_prob_col='Calibrated_Probability', thresholds=None,
                                                                    y_true_for_validation=y_train.values, risk_labels=['low-risk','medium-risk','high-risk'],
                                                                    output_dir=output_dir, dataset_name="Training_Set")
    test_preds = pd.DataFrame({'True_Label': y_test.values, 'Predicted_Label': final_model.predict(X_test),
                               'Predicted_Probability': y_proba_test, 'Calibrated_Probability': y_proba_test})
    test_results, _, _, _ = add_risk_stratification_fixed(test_preds, calibrated_prob_col='Calibrated_Probability', thresholds=thresholds,
                                                          risk_labels=['low-risk','medium-risk','high-risk'], output_dir=output_dir, dataset_name="Test_Set")
    test_results.to_csv(os.path.join(output_dir, "test_set_predictions.csv"), index=False)
    plot_risk_stratification(test_results, calibrated_prob_col='Calibrated_Probability', risk_level_col='Risk_Level',
                             output_dir=os.path.join(output_dir, "risk_stratification_plots"), dataset_name="Test Set")

    # ==================== 外部验证（补全独立 external_test 文件夹） ====================
    if external_test_path:
        print("\n[Step 7/7] External validation...")
        X_ext, y_ext = load_external_test_data(external_test_path, feature_names, target_column)
        # 首先对所有模型进行外部验证（生成 external_test_all_models）
        external_metrics_df = evaluate_all_models_on_external_test_fixed(best_models, X_ext, y_ext, feature_names, best_ref, thresholds, output_dir)
        # 然后对最佳模型单独进行外部验证（生成 external_test 文件夹）
        best_external_model_name = external_metrics_df.nlargest(1, 'AUC').iloc[0]['Model']
        best_external_model = best_models[best_external_model_name]['pipeline']
        _ = evaluate_external_test_fixed(best_external_model, X_ext, y_ext, best_external_model_name, feature_names, thresholds, output_dir)
    else:
        print("\n[Step 7/7] No external test set provided. Skipping external validation.")
        X_ext, y_ext = None, None

    # ===== Table 3 SVM 内部+外部 CI + Supp 风险分层 CI 整合到主脚本 =====
    try:
        compute_table3_svm_ci(best_models, X_test, y_test, X_ext, y_ext, output_dir)
    except Exception as e_t3:
        print(f"⚠ Table 3 SVM CI 计算失败（不影响主流程）: {e_t3}")
        import traceback as _tb_t3
        _tb_t3.print_exc()
    try:
        compute_supp_risk_strata_ci(final_model, X_test, y_test, X_ext, y_ext,
                                     low_thr=thresholds.get('low_threshold', 0.40),
                                     high_thr=thresholds.get('high_threshold', 0.80),
                                     output_dir=output_dir)
    except Exception as e_supp:
        print(f"⚠ Supp 风险分层 CI 计算失败（不影响主流程）: {e_supp}")
        import traceback as _tb_supp
        _tb_supp.print_exc()

    # ==================== 最终总结打印 ====================
    print("\n" + "="*60)
    print("SUMMARY OF RESULTS (v25 - Updated Clinical Driven Risk Stratification: 0.40/0.80)")
    print("="*60)
    print(f"\nCV-SELECTED REFERENCE MODEL (used for DeLong tests):")
    print(f"  Model: {best_ref}")
    best_cv_auc = cv_results[cv_results['Model']==best_ref]['AUC'].values[0]
    print(f"  CV AUC: {best_cv_auc:.4f}")
    print(f"\nRisk Stratification Thresholds (computed on TRAINING set only - CLINICAL DRIVEN METHOD v25):")
    print(f"  Low-risk: ≤ {thresholds.get('low_threshold', 0.40):.4f}")
    print(f"  High-risk: > {thresholds.get('high_threshold', 0.80):.4f}")
    print(f"  Method: {thresholds.get('method', 'clinical_driven')}")
    print(f"\nInternal Test Set Results (DeLong vs Reference Model, Bonferroni corrected):")
    internal_best = internal_metrics.nlargest(1, 'AUC').iloc[0]
    print(f"  Best performing model on internal test: {internal_best['Model']} (AUC={internal_best['AUC']:.4f})")
    delong_path = os.path.join(output_dir, "test_all_models", f"delong_test_results_Test_with_reference_{best_ref}.csv")
    if os.path.exists(delong_path):
        delong_df = pd.read_csv(delong_path)
        sig_col = 'Significant_Corrected' if 'Significant_Corrected' in delong_df.columns else 'Significant_Raw'
        sig_pairs = delong_df[delong_df[sig_col] == 'Yes']
        if len(sig_pairs) > 0:
            print(f"\n  Significant differences found in {len(sig_pairs)} model pairs (vs reference, Bonferroni):")
            for _, row in sig_pairs.iterrows():
                p_col = 'P_Value_Corrected' if 'P_Value_Corrected' in row else 'P_Value_Raw'
                sig_col_val = row.get('Significance_Level_Corrected', row.get('Significance_Level_Raw', 'ns'))
                print(f"    {row['Compared_Model']}: p={row[p_col]:.6f} {sig_col_val}")
        else:
            print("\n  No significant differences found between any model and reference (Bonferroni corrected, p > 0.05)")
    if external_test_path:
        external_metrics_path = os.path.join(output_dir, "external_test_all_models", "all_models_metrics_summary.csv")
        if os.path.exists(external_metrics_path):
            external_metrics_df = pd.read_csv(external_metrics_path)
            external_best = external_metrics_df.nlargest(1, 'AUC').iloc[0]
            print(f"\nExternal Test Set Results (DeLong vs Reference Model, Bonferroni corrected):")
            print(f"  Best performing model on external test: {external_best['Model']} (AUC={external_best['AUC']:.4f})")
            external_delong_path = os.path.join(output_dir, "external_test_all_models", f"delong_test_results_External_Test_with_reference_{best_ref}.csv")
            if os.path.exists(external_delong_path):
                external_delong_df = pd.read_csv(external_delong_path)
                sig_col_ext = 'Significant_Corrected' if 'Significant_Corrected' in external_delong_df.columns else 'Significant_Raw'
                sig_pairs_ext = external_delong_df[external_delong_df[sig_col_ext] == 'Yes']
                if len(sig_pairs_ext) > 0:
                    print(f"\n  Significant differences found in {len(sig_pairs_ext)} model pairs (vs reference, Bonferroni):")
                    for _, row in sig_pairs_ext.iterrows():
                        p_col = 'P_Value_Corrected' if 'P_Value_Corrected' in row else 'P_Value_Raw'
                        sig_col_val = row.get('Significance_Level_Corrected', row.get('Significance_Level_Raw', 'ns'))
                        print(f"    {row['Compared_Model']}: p={row[p_col]:.6f} {sig_col_val}")
                else:
                    print("\n  No significant differences found between any model and reference (Bonferroni corrected, p > 0.05)")
            if internal_best['Model'] == external_best['Model']:
                print(f"\n✓ Same best model ({internal_best['Model']}) performs well on both internal and external test sets")
            else:
                print(f"\n⚠ Different best models: Internal={internal_best['Model']}, External={external_best['Model']}")
    print(f"\nTotal models evaluated: {len(internal_metrics)}")
    print(f"Pairwise DeLong test results with Bonferroni correction saved in output directories")
    print(f"Risk stratification: CLINICAL DRIVEN METHOD - thresholds computed ONLY on training set - NO DATA LEAKAGE")
    print(f"Updated thresholds: Low-risk ≤ 0.40, High-risk > 0.80")
    print(f"All plots font unified: Times New Roman, title size={TITLE_FONT_SIZE}, label size={AXIS_LABEL_FONT_SIZE}, tick size={TICK_FONT_SIZE}")
    print(f"\nAll results saved to: {os.path.abspath(output_dir)}")


if __name__ == "__main__":
    # Repo-relative defaults via config.py — no hardcoded absolute paths.
    import sys as _sys
    from pathlib import Path as _Path
    _sys.path.insert(0, str(_Path(__file__).resolve().parent))
    from config import FULL_DEV_CSV, FULL_EXTERNAL_CSV, RESULTS_DIR

    parser = argparse.ArgumentParser(
        description="Multimodal lung-cancer vs benign-nodule pipeline "
                    "(paper Tables 2-3 + SHAP + risk strata). "
                    "疾病诊断预测 + SHAP 解释（最佳模型）")
    parser.add_argument('--shap-only', action='store_true',
                        help='跳过 9 模型训练与评估，直接加载 shap_checkpoint.pkl 做 SHAP（需先完整运行过一次）')
    parser.add_argument('--sample-size', type=int, default=200,
                        help='SHAP 采样样本数（默认 200，越小越快，蜂群图越稀疏）')
    parser.add_argument('--n-background', type=int, default=50,
                        help='KernelExplainer 背景数据大小（默认 50，越大代表性越好但越慢）')
    parser.add_argument('--data', default=str(FULL_DEV_CSV),
                        help='Dev cohort CSV (full matrix). Default: repo full dev matrix.')
    parser.add_argument('--external', default=str(FULL_EXTERNAL_CSV),
                        help='External cohort CSV (full matrix). Default: repo full external matrix.')
    parser.add_argument('--out', default=None,
                        help='Output directory. Default: results/run_<timestamp>')
    args = parser.parse_args()

    _run_ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    OUTPUT_DIR = args.out or str(RESULTS_DIR / f"run_{_run_ts}")
    _Path(OUTPUT_DIR).mkdir(parents=True, exist_ok=True)

    if args.shap_only:
        main_shap_only(args.data, OUTPUT_DIR, target_column='target',
                       sample_size=args.sample_size, n_background=args.n_background)
    else:
        main(args.data, args.external, OUTPUT_DIR, target_column='target')