# -*- coding: utf-8 -*-
"""
Feature configuration for the Lung Cancer Early Screening AI Warning System.
Defines the 121 feature names (in exact training order), feature groupings,
and risk stratification thresholds.
"""

TARGET_COLUMN = 'target'

RISK_THRESHOLDS = {'low': 0.40, 'high': 0.80}

RANDOM_SEED = 42

# 121 features in exact training order (from CSV header, excluding 'target')
FEATURE_NAMES = [
    'TB-MEAN', 'Per-all', 'TB-Cr', 'TB-R', 'TC-Cr', 'TB-I', 'TC-L', 'TC-Y',
    'TB-b', 'TC-R', 'TB-CON', 'TB-B', 'TC-CON', 'TC-Cb', 'TC-a', 'TB-G',
    'TB-Y', 'TC-H', 'TC-G', 'TB-H', 'TB-a', 'Per-part', 'TB-S', 'TC-ENT',
    'TC-S', 'TC-ASM', 'TB-ASM', 'TC-MEAN', 'h4/h1', 'h4', 'h5', 'h1/t1',
    'h3/h1', 't4', 'w2/t', 't1/t', 't5', 'w1/t', 'lipcolor-R', 'GLCM-idm_2',
    'GLCM-idm_3', 'GLCM-asm_3', 'GLCM-con_2', 'GLCM-con_1', 'GLCM-con_0',
    'wholecolor-H', 'color-Cb-7', 'wholecolor-b', 'color-a-0', 'wholecolor-Cb',
    'color-Cb-1', 'color-Cr-6', 'lipcolor-Cb', 'color-b-4', 'wholecolor-S',
    'color-H-6', 'color-a-3', 'color-R-2', 'color-Cb-3', 'color-L-4',
    'color-S-6', 'color-L-0', 'color-R-6', 'color-b-6', 'color-B-1',
    'color-R-1', 'lipcolor-Y', 'color-B-2', 'color-Cb-2', 'lipcolor-S',
    'color-R-0', 'color-Cb-0', 'color-S-2', 'color-G-1', 'color-a-1',
    'color-Cr-1', 'color-V-1', 'color-Cb-5', 'color-G-0', 'wholecolor-R',
    'color-V-6', 'wholecolor-V', 'color-Y-6', 'lipcolor-V', 'color-V-7',
    'color-V-4', 'color-S-3', 'color-G-5', 'wholecolor-Y', 'lipcolor-a',
    'color-R-3', 'color-R-7', 'color-Cr-4', 'color-b-3', 'color-G-2',
    'color-G-3', 'color-Cr-5', 'color-R-4', 'color-V-5', 'color-V-3',
    'color-S-4', 'lipcolor-L', 'color-H-2', 'color-S-5', 'color-b-5',
    'color-L-7', 'wholecolor-B', 'color-Cr-7', 'color-L-2', 'wholecolor-L',
    'color-H-5', 'color-H-1', 'color-Cr-0', 'wholecolor-a', 'wholecolor-Cr',
    'lipcolor-H', 'GLCM-ent_2', 'GLCM-asm_0', 'GLCM-idm_0', 'GLCM-con_3',
    'GLCM-idm_1',
]

N_FEATURES = len(FEATURE_NAMES)  # 121

# Feature groupings for display and analysis
FEATURE_GROUPS = {
    'tongue_body': [f for f in FEATURE_NAMES if f.startswith('TB-')],
    'tongue_coating': [f for f in FEATURE_NAMES if f.startswith('TC-')],
    'pulse': [f for f in FEATURE_NAMES if any(f.startswith(p) for p in ('h', 't', 'w', 'Per-'))],
    'lip_color': [f for f in FEATURE_NAMES if f.startswith('lipcolor-')],
    'glcm_texture': [f for f in FEATURE_NAMES if f.startswith('GLCM-')],
    'face_color': [f for f in FEATURE_NAMES if f.startswith(('color-', 'wholecolor-'))],
}

# Display names for UI (English)
GROUP_DISPLAY_NAMES = {
    'tongue_body': 'Tongue Body',
    'tongue_coating': 'Tongue Coating',
    'pulse': 'Pulse Waveform',
    'lip_color': 'Lip Color',
    'glcm_texture': 'Facial Texture',
    'face_color': 'Facial Complexion',
}

# Face color split for the manual-entry dialog: 64 features are too many to show
# in a single table, so the Facial Complexion tab uses two accordions.
FACE_COLOR_SUBGROUPS = {
    'whole': [f for f in FEATURE_GROUPS['face_color'] if f.startswith('wholecolor-')],
    'region': [f for f in FEATURE_GROUPS['face_color'] if f.startswith('color-')],
}

# ===== User-curated feature organization (per 特征.xlsx) =====
# Three top-level dimensions (舌/面/脉), each with subgroups.
# Note: this covers 98 of the 121 model features. The remaining 23 (mostly RGB
# channels: TB-R/B/G, TC-R/G, lipcolor-R, wholecolor-R/B, color-R-*/G-*/B-*)
# are listed in EXTRA_MODEL_FEATURES below and shown in a separate "Other" tab.
# Excel and CSV both use slashes for pulse ratios (h3/h1, h4/h1, etc.).
FEATURE_DIMENSIONS = [
    {
        'key': 'tongue',
        'display': 'Tongue',
        'display_cn': '舌',
        'subgroups': [
            {
                'key': 'tongue_body',
                'display': 'Tongue Body',
                'display_cn': '舌质',
                'features': ['TB-Y', 'TB-Cr', 'TB-a', 'TB-b', 'TB-H', 'TB-I',
                             'TB-S', 'TB-MEAN', 'TB-CON', 'TB-ASM'],
            },
            {
                'key': 'tongue_coating',
                'display': 'Tongue Coating',
                'display_cn': '舌苔',
                'features': ['TC-Y', 'TC-Cb', 'TC-Cr', 'TC-L', 'TC-a', 'TC-H',
                             'TC-S', 'TC-MEAN', 'TC-CON', 'TC-ENT', 'TC-ASM'],
            },
            {
                'key': 'coating_index',
                'display': 'Coating Index',
                'display_cn': '舌苔指数',
                'features': ['Per-all', 'Per-part'],
            },
        ],
    },
    {
        'key': 'face',
        'display': 'Face',
        'display_cn': '面',
        'subgroups': [
            {
                'key': 'lip_color',
                'display': 'Lip Color',
                'display_cn': '唇色',
                'features': ['lipcolor-Y', 'lipcolor-Cb', 'lipcolor-L',
                             'lipcolor-a', 'lipcolor-H', 'lipcolor-S', 'lipcolor-V'],
            },
            {
                'key': 'face_whole',
                'display': 'Face Whole Color',
                'display_cn': '面色整体',
                'features': ['wholecolor-Y', 'wholecolor-Cb', 'wholecolor-Cr',
                             'wholecolor-L', 'wholecolor-a', 'wholecolor-b',
                             'wholecolor-H', 'wholecolor-S', 'wholecolor-V'],
            },
            {
                'key': 'face_region_0',
                'display': 'Face Region 0',
                'display_cn': '面色分区0',
                'features': ['color-Cb-0', 'color-Cr-0', 'color-L-0', 'color-a-0'],
            },
            {
                'key': 'face_region_1',
                'display': 'Face Region 1',
                'display_cn': '面色分区1',
                'features': ['color-Cb-1', 'color-Cr-1', 'color-a-1',
                             'color-H-1', 'color-V-1'],
            },
            {
                'key': 'face_region_2',
                'display': 'Face Region 2',
                'display_cn': '面色分区2',
                'features': ['color-Cb-2', 'color-L-2', 'color-H-2', 'color-S-2'],
            },
            {
                'key': 'face_region_3',
                'display': 'Face Region 3',
                'display_cn': '面色分区3',
                'features': ['color-Cb-3', 'color-a-3', 'color-b-3',
                             'color-S-3', 'color-V-3'],
            },
            {
                'key': 'face_region_4',
                'display': 'Face Region 4',
                'display_cn': '面色分区4',
                'features': ['color-Cr-4', 'color-L-4', 'color-b-4',
                             'color-S-4', 'color-V-4'],
            },
            {
                'key': 'face_region_5',
                'display': 'Face Region 5',
                'display_cn': '面色分区5',
                'features': ['color-Cb-5', 'color-Cr-5', 'color-b-5',
                             'color-H-5', 'color-S-5', 'color-V-5'],
            },
            {
                'key': 'face_region_6',
                'display': 'Face Region 6',
                'display_cn': '面色分区6',
                'features': ['color-Cr-6', 'color-b-6', 'color-H-6',
                             'color-S-6', 'color-V-6'],
            },
            {
                'key': 'face_region_7',
                'display': 'Face Region 7',
                'display_cn': '面色分区7',
                'features': ['color-Cb-7', 'color-Cr-7', 'color-L-7', 'color-V-7'],
            },
            {
                'key': 'glcm_texture',
                'display': 'Facial Texture (GLCM)',
                'display_cn': '纹理（GLCM）',
                'features': ['GLCM-con_0', 'GLCM-con_1', 'GLCM-con_2', 'GLCM-con_3',
                             'GLCM-ent_2', 'GLCM-asm_0', 'GLCM-asm_3',
                             'GLCM-idm_0', 'GLCM-idm_1', 'GLCM-idm_2', 'GLCM-idm_3'],
            },
        ],
    },
    {
        'key': 'pulse',
        'display': 'Pulse',
        'display_cn': '脉',
        'subgroups': [
            {
                'key': 'pulse_wave',
                'display': 'Pulse Wave',
                'display_cn': '脉搏波',
                'features': ['h4', 'h5', 'h3/h1', 'h4/h1', 'h1/t1',
                             't4', 't5', 't1/t', 'w1/t', 'w2/t'],
            },
        ],
    },
]

# All features covered by FEATURE_DIMENSIONS (98 of 121)
DIMENSION_FEATURES = [f for dim in FEATURE_DIMENSIONS
                      for sg in dim['subgroups'] for f in sg['features']]

# Features in the model but NOT in the user's Excel table (23 RGB-channel features).
# These are auto-imputed (training mean) when the user doesn't enter them.
EXTRA_MODEL_FEATURES = [f for f in FEATURE_NAMES if f not in DIMENSION_FEATURES]
