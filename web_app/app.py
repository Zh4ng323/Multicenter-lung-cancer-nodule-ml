# -*- coding: utf-8 -*-
"""
Lung Cancer Early Screening Multi-modal TCM Four-Diagnosis AI Warning System
"""

import os
import re
import sys
import traceback
import json
import base64

# Compatible base dir for both source run and PyInstaller bundle
# BASE_DIR = bundled read-only resources (model, assets) -> _MEIPASS when frozen
# APP_DIR  = writable location for prediction outputs -> user's Documents folder
BASE_DIR = getattr(sys, '_MEIPASS', os.path.dirname(os.path.abspath(__file__)))
try:
    APP_DIR = os.path.join(os.path.expanduser('~'), 'Documents', 'LungCancerAI_Results')
except Exception:
    APP_DIR = os.path.expanduser('~')

# PWA 资源目录（manifest / service worker / 图标）。
# 只有源码目录包含 pwa/ 时才启用 PWA 注入与新启动方式；
# 桌面 PyInstaller 打包（_MEIPASS 中无 pwa/）自动走原有逻辑，零影响。
PWA_DIR = os.path.join(BASE_DIR, "pwa")
PWA_AVAILABLE = os.path.isfile(os.path.join(PWA_DIR, "manifest.json"))

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

import gradio as gr

if not getattr(sys, 'frozen', False):
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from predictor import DiagnosisPredictor
from risk_stratifier import assign_risk_level, get_risk_html, RISK_THRESHOLDS
from shap_explainer import (explain_single_patient, generate_waterfall_plot,
                            generate_feature_importance_bar,
                            generate_decision_explanation)
from feature_config import (
    FEATURE_NAMES, FEATURE_DIMENSIONS, DIMENSION_FEATURES,
    EXTRA_MODEL_FEATURES, TARGET_COLUMN,
)
from health_guidance import get_guidance_html

plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['font.size'] = 10
plt.rcParams['axes.titlesize'] = 11
plt.rcParams['axes.labelsize'] = 10
plt.rcParams['xtick.labelsize'] = 9
plt.rcParams['ytick.labelsize'] = 9

# ==================== Theme System ====================
THEME_NAME = os.environ.get("THEME", "professional")

_SHARED_CSS = """
/* ===== App shell ===== */
.gradio-container {
    max-width: 1280px !important;
    margin: 8px auto 16px auto !important;
    font-family: system-ui, -apple-system, "Segoe UI", Roboto, "Helvetica Neue", Arial, sans-serif !important;
    font-size: 14px !important;
    border: 1px solid #c5cdd6 !important;
    border-radius: 14px !important;
    box-shadow: 0 6px 24px rgba(15, 40, 80, 0.12) !important;
    overflow: hidden !important;
    background: #f5f7fa !important;
    padding: 0 !important;
}
footer { display: none !important; }

/* Header — force pure white text everywhere */
#app-header {
    background: linear-gradient(135deg, #0d47a1 0%, #1565c0 55%, #1a73e8 100%);
    color: #ffffff !important;
    padding: 11px 18px 10px 18px;
    display: flex;
    align-items: center;
    justify-content: space-between;
    gap: 12px;
    border-bottom: 1px solid rgba(0,0,0,0.08);
    border-radius: 14px 14px 0 0;
}
#app-header, #app-header * {
    color: #ffffff !important;
}
#app-header .brand { display: flex; align-items: center; gap: 11px; }
#app-header .brand-logo {
    width: 44px; height: 44px; border-radius: 10px;
    object-fit: contain;
    background: rgba(255,255,255,0.95);
    border: 1px solid rgba(255,255,255,0.45);
    padding: 3px;
    flex-shrink: 0;
    box-shadow: 0 1px 4px rgba(0,0,0,0.18);
}
#app-header .brand-text h1,
#app-header h1 {
    margin: 0; font-size: 19px; font-weight: 700;
    color: #ffffff !important;
    letter-spacing: 0.2px; line-height: 1.25;
}
#app-header .brand-text .sub {
    margin: 2px 0 0 0; font-size: 11.5px; opacity: 0.95; font-weight: 400;
    color: #ffffff !important;
}
#app-header .meta {
    text-align: right; font-size: 11.5px; opacity: 0.95; line-height: 1.4;
    white-space: nowrap; color: #ffffff !important;
}
#app-header .meta .badge {
    display: inline-block; background: rgba(255,255,255,0.18);
    border: 1px solid rgba(255,255,255,0.35); border-radius: 10px;
    padding: 1px 8px; font-size: 10.5px; margin-left: 4px; font-weight: 600;
    color: #ffffff !important;
}

/* Footer */
#app-footer {
    background: #e8edf3; border-top: 1px solid #d0d7de;
    padding: 6px 16px; font-size: 11.5px; color: #5a6573;
    display: flex; justify-content: space-between; align-items: center;
}
#app-footer .ver { color: #1a73e8; font-weight: 700; }

.gradio-container > .main,
.gradio-container .contain,
.gradio-container > main {
    padding: 8px 12px 4px 12px !important;
}

/* Labels / inputs */
label, .gr-input-label, .gr-box label span,
.block > label > span, .label-wrap span {
    font-size: 13px !important; font-weight: 600 !important;
}
input[type="text"], input[type="number"], textarea,
.gr-text-input input, .gr-input {
    font-size: 13px !important;
}
button, .gr-button, button.lg, button.sm {
    font-size: 13px !important; font-weight: 600 !important;
}

/* Checkbox group */
.gr-checkboxgroup .wrap { flex-wrap: nowrap !important; gap: 2px !important; }
.gr-checkboxgroup label { padding: 1px 3px !important; font-size: 11px !important; line-height: 1.2 !important; }
.gr-checkboxgroup label span { font-size: 11px !important; }
.gr-checkboxgroup input[type="checkbox"] { width: 12px !important; height: 12px !important; }

/* Fixed-height panels */
#overview-img img { object-fit: contain !important; }
#overview-img > div { min-height: 220px !important; }

/* CSV upload — compact two-line drop text, keep icon + clear (X) */
#csv-upload {
    min-height: 78px !important;
    max-height: 96px !important;
}
/* UploadText .wrap default min-height is 240px; force compact two-line layout */
#csv-upload .wrap {
    min-height: 56px !important;
    height: auto !important;
    padding-top: 4px !important;
    padding-bottom: 4px !important;
    gap: 0 !important;
    line-height: 1.25 !important;
}
/* hide big upload icon to save vertical space (two-line text only) */
#csv-upload .icon-wrap {
    display: none !important;
}
#csv-upload .or {
    display: inline !important;
    font-size: 11px !important;
    margin: 0 4px !important;
}
#csv-upload .wrap,
#csv-upload .wrap * {
    font-size: 12px !important;
}
/* file name after upload */
#csv-upload .file-name,
#csv-upload .filename,
#csv-upload table {
    font-size: 12.5px !important;
}
/* clear / remove (X) — ModifyUpload IconButton, top-right absolute */
#csv-upload button,
#csv-upload button[aria-label],
#csv-upload .label-clear-button {
    display: inline-flex !important;
    visibility: visible !important;
    opacity: 1 !important;
    pointer-events: auto !important;
    z-index: 20 !important;
}
#csv-upload button svg {
    display: inline-block !important;
    visibility: visible !important;
    width: 14px !important;
    height: 14px !important;
    opacity: 1 !important;
}

/* Decision table: previous 220/242 +5% */
#decision-box { min-height: 231px; max-height: 254px; overflow-y: auto; }

/* Strip Gradio HTML-component wrapper so the inner styled div doesn't
   double-border / double-background against its parent block.
   This removes the "two stacked layers" visual on risk/prob/guidance/decision. */
#risk-html, #prob-html, #guidance-html, #decision-html {
    border: none !important;
    background: transparent !important;
    padding: 0 !important;
    margin: 0 !important;
    box-shadow: none !important;
}
#risk-html > .wrap,
#prob-html > .wrap,
#guidance-html > .wrap,
#decision-html > .wrap,
#risk-html .gr-html,
#prob-html .gr-html,
#guidance-html .gr-html,
#decision-html .gr-html {
    border: none !important;
    background: transparent !important;
    padding: 0 !important;
    margin: 0 !important;
    box-shadow: none !important;
}
/* Gradio 4 wraps gr.HTML in <form class="gr-block"> — neutralize it too */
form#risk-html.gr-block,
form#prob-html.gr-block,
form#guidance-html.gr-block,
form#decision-html.gr-block,
#risk-html.gr-block,
#prob-html.gr-block,
#guidance-html.gr-block,
#decision-html.gr-block {
    border: none !important;
    background: transparent !important;
    padding: 0 !important;
    margin: 0 !important;
    box-shadow: none !important;
}

/* SHAP panels — label merged into gr.Image; no placeholder images */
#shap-plots {
    margin-top: 4px !important;
    gap: 8px !important;
    align-items: stretch !important;
}
#shap-plots .block {
    margin-top: 0 !important;
    padding-top: 0 !important;
}
#shap-plots label,
#shap-plots .label-wrap span,
#shap-plots .gr-box label span {
    font-size: 13px !important;
    font-weight: 700 !important;
    color: #0d47a1 !important;
}
/* offset generated figure from the top edge of its box (below label) */
#plot-waterfall,
#plot-importance {
    padding-top: 10px !important;
}
#plot-waterfall .image-container,
#plot-importance .image-container,
#plot-waterfall > .wrap,
#plot-importance > .wrap {
    padding-top: 8px !important;
    margin-top: 4px !important;
}
#plot-waterfall img,
#plot-importance img,
#shap-plots img {
    object-fit: contain !important;
    width: 100% !important;
    margin-top: 8px !important;
    background: #ffffff !important;
    border-radius: 4px;
}
/* hide empty-state giant icon if Image has no value */
#plot-waterfall .empty,
#plot-importance .empty,
#shap-plots .empty {
    min-height: 120px !important;
    max-height: 160px !important;
}
#plot-waterfall .empty svg,
#plot-importance .empty svg {
    width: 20px !important;
    height: 20px !important;
    opacity: 0.35 !important;
}

/* Model row: dropdown + Switch same total height (label + input) */
#model-row {
    align-items: stretch !important;
    gap: 8px !important;
}
#model-row > div {
    display: flex !important;
    align-items: stretch !important;
}
/* Switch fills full height of the model block (~label + dropdown) */
#btn-switch,
#model-row #btn-switch,
#model-row button {
    min-width: 72px !important;
    height: 100% !important;
    min-height: 68px !important;
    padding: 0 12px !important;
    font-size: 13px !important;
    font-weight: 600 !important;
    align-self: stretch !important;
    display: inline-flex !important;
    align-items: center !important;
    justify-content: center !important;
}

/* Action row: Reset | Predict (big center) | Save */
#action-row {
    align-items: stretch !important;
    gap: 6px !important;
}
#action-row button {
    height: 38px !important;
}
#btn-predict {
    font-size: 14px !important;
    font-weight: 700 !important;
}
#btn-reset, #btn-save {
    font-size: 12px !important;
    min-width: 60px !important;
}

.gap { gap: 6px !important; }

/* ===== Manual entry input row + dialog ===== */
/* CSV upload + Manual Entry + View/Edit in one row */
#input-row { align-items: stretch !important; gap: 6px !important; }
#input-row > div { display: flex !important; }
#btn-manual, #btn-view {
    min-width: 72px !important;
    height: 78px !important;
    font-size: 12px !important;
    font-weight: 600 !important;
    white-space: pre-line !important;
    line-height: 1.15 !important;
    padding: 4px 6px !important;
    align-self: stretch !important;
    display: inline-flex !important;
    align-items: center !important;
    justify-content: center !important;
}
/* Dialog sizing — fallback Column acts as a fixed-position overlay */
#manual-dialog {
    position: fixed !important;
    top: 50% !important;
    left: 50% !important;
    transform: translate(-50%, -50%) !important;
    z-index: 9999 !important;
    background: #ffffff !important;
    border: 2px solid #1a73e8 !important;
    border-radius: 10px !important;
    padding: 14px 18px !important;
    max-width: 900px !important;
    width: 92vw !important;
    max-height: 88vh !important;
    overflow-y: auto !important;
    box-shadow: 0 12px 40px rgba(15, 40, 80, 0.35) !important;
}
#manual-dialog-actions { gap: 8px !important; }
#manual-dialog-actions button { flex: 1; }
#btn-manual-cancel, #btn-manual-clear {
    font-size: 13px !important;
}
#btn-manual-save {
    font-size: 13px !important;
    font-weight: 700 !important;
}
/* Compact scrollable dataframe tables in the dialog */
#manual-tabs .gr-dataframe { max-height: 360px; overflow: auto; }
#manual-tabs .gr-dataframe table { font-size: 12px !important; }
#manual-tabs .gr-dataframe th,
#manual-tabs .gr-dataframe td { padding: 3px 6px !important; }
#manual-tabs .gr-dataframe td:first-child { color: #5a6470 !important; }

/* =========================================================================
   Responsive (mobile) overrides — applied only at narrow widths.
   Desktop layout is untouched; these all live inside @media.
   ========================================================================= */
/* Shared mobile breakpoint — wizard step visibility (default: show all).
   Body gets data-wizard-step="1|2|3" set by JS on mobile only.
   On desktop JS never activates it, so everything stays visible. */
@media (max-width: 768px) {
    /* Shell: go full-width, drop the heavy card chrome */
    .gradio-container {
        max-width: 100% !important;
        margin: 0 !important;
        border: none !important;
        border-radius: 0 !important;
        box-shadow: none !important;
    }
    .gradio-container > .main,
    .gradio-container .contain,
    .gradio-container > main {
        padding: 8px 8px 4px 8px !important;
    }

    /* Header: allow wrapping, shrink type */
    #app-header {
        padding: 9px 12px !important;
        flex-wrap: wrap !important;
        gap: 6px !important;
        border-radius: 0 !important;
    }
    #app-header .brand-logo {
        width: 36px !important; height: 36px !important;
    }
    #app-header .brand-text h1,
    #app-header h1 {
        font-size: 16px !important;
    }
    #app-header .brand-text .sub {
        font-size: 10.5px !important;
    }
    /* meta: stop forcing one line; right-align remains */
    #app-header .meta {
        white-space: normal !important;
        font-size: 10.5px !important;
        text-align: right !important;
    }

    /* Footer: wrap */
    #app-footer {
        padding: 6px 12px !important;
        flex-wrap: wrap !important;
        gap: 4px !important;
        font-size: 10.5px !important;
    }

    /* Main two-column layout → stack vertically.
       Gradio 4 Rows are flex; flip direction on the top Row. */
    .gradio-container > .main > .wrap > .form,
    .gradio-container main > div > .gap {
        flex-direction: column !important;
    }
    /* Force any Row that is meant to be a two-column split to stack. */
    .gradio-container .form > .wrap,
    .gradio-container .gap > .wrap {
        flex-wrap: wrap !important;
    }

    /* CheckboxGroup (Steps): keep the three options on ONE line on phones.
       In Gradio 4.44 a CheckboxGroup renders as <fieldset class="block ...">
       with an inner <div class="wrap">, NOT as .gr-checkboxgroup, so target
       the actual fieldset+wrap structure. */
    fieldset .wrap {
        flex-wrap: nowrap !important;
        gap: 4px !important;
        overflow-x: auto !important;
    }
    fieldset .wrap > label {
        padding: 1px 4px !important;
        font-size: 10.5px !important;
        line-height: 1.15 !important;
        white-space: nowrap !important;
    }
    fieldset .wrap > label span { font-size: 10.5px !important; }

    /* CSV upload + Manual + View row: stack so the upload gets full width */
    #input-row > div {
        flex-direction: column !important;
    }
    #btn-manual, #btn-view {
        height: auto !important;
        min-height: 44px !important;
        padding: 8px 10px !important;
    }
    /* Model row: dropdown + Switch — allow wrap, switch becomes normal height */
    #model-row > div {
        flex-wrap: wrap !important;
    }
    #model-row #btn-switch,
    #model-row button {
        min-height: 44px !important;
        padding: 8px 12px !important;
    }

    /* Action row: keep horizontal but allow wrap; bigger touch targets */
    #action-row {
        flex-wrap: wrap !important;
    }
    #action-row button {
        height: 44px !important;
        min-width: 72px !important;
    }

    /* Images: shrink fixed heights so they don't dominate a phone screen */
    #overview-img > div {
        min-height: 140px !important;
    }
    #overview-img {
        height: auto !important;
    }
    #plot-waterfall,
    #plot-importance {
        height: auto !important;
    }
    #plot-waterfall .empty,
    #plot-importance .empty,
    #shap-plots .empty {
        min-height: 80px !important;
        max-height: 120px !important;
    }

    /* SHAP plots: stack the two charts vertically instead of side by side */
    #shap-plots {
        flex-direction: column !important;
    }

    /* Decision / overview container heights: relax fixed px so content flows */
    #decision-box {
        min-height: auto !important;
        max-height: none !important;
    }

    /* Manual-entry dialog → full-screen sheet on mobile, sitting ABOVE the
       bottom wizard nav. The middle (tabs) scrolls; the action row
       (Cancel/Clear/Save) is pinned at the bottom of the sheet so it is
       always reachable even when many Accordion groups are expanded.

       IMPORTANT: do NOT set `display` here. Gradio toggles the dialog's
       visibility via inline style (display:none / display:flex) from the
       Cancel/Save handlers; a `display:flex !important` here would override
       that and leave the dialog permanently open. We only set
       flex-direction so that WHEN Gradio shows it (inline display:flex) the
       three-section column layout applies. */
    #manual-dialog {
        top: 0 !important;
        left: 0 !important;
        transform: none !important;
        width: 100vw !important;
        max-width: 100vw !important;
        /* Leave room for the fixed bottom wizard nav bar (~61px + safe area).
           Pin BOTH top and bottom edges so the sheet can never extend under
           the nav, even on engines where `100vh` is wrong/stale while the
           mobile keyboard is open. `max-height` stays as a fallback for
           browsers that ignore `bottom` on fixed elements. */
        bottom: calc(72px + env(safe-area-inset-bottom, 0px)) !important;
        max-height: calc(100vh - 72px - env(safe-area-inset-bottom, 0px)) !important;
        border-radius: 0 !important;
        border: none !important;
        padding: 10px 12px 8px 12px !important;
        flex-direction: column !important;
        touch-action: manipulation !important;
        overscroll-behavior: contain !important;
    }
    /* When the manual-entry sheet is open, it is a full-screen modal: hide
       the bottom wizard nav (the step bar belongs to the page behind it) and
       let the sheet fill the screen minus the safe area. The JS toggles
       .nav-hidden / .dialog-open while the dialog is visible.
       NOTE: Gradio scopes injected CSS by prepending a container prefix to
       every selector, so `body[...]`-based rules would silently stop
       matching; a class on the element itself survives the scoping and also
       wins the specificity fight against the default `#wizard-nav` rule. */
    #wizard-nav.nav-hidden {
        display: none !important;
    }
    #manual-dialog.dialog-open {
        bottom: calc(8px + env(safe-area-inset-bottom, 0px)) !important;
        max-height: calc(100vh - 8px - env(safe-area-inset-bottom, 0px)) !important;
    }
    /* Make the tabs area the scrollable region of the dialog. The dialog is a
       flex column (max-height leaves room for the bottom wizard nav); the tabs
       take the flexible middle and scroll. */
    #manual-dialog > div:first-child { flex: 0 0 auto !important; }
    #manual-tabs {
        flex: 1 1 auto !important;
        min-height: 0 !important;
        overflow-y: auto !important;
        -webkit-overflow-scrolling: touch;
        overscroll-behavior: contain;
    }
    /* Action row: a normal-flow sibling at the bottom of the dialog (NOT
       position:fixed, which breaks Gradio's click handlers). The dialog is a
       flex column with max-height; the tabs area scrolls (flex:1) while this
       row is flex:0 (never shrinks) so it always sits at the bottom of the
       sheet, fully visible above the wizard nav. Three buttons forced to ONE
       row with equal widths. */
    #manual-dialog-actions {
        flex: 0 0 auto !important;
        display: flex !important;
        flex-direction: row !important;
        flex-wrap: nowrap !important;
        gap: 6px !important;
        padding: 8px 0 4px 0 !important;
        background: #ffffff !important;
        border-top: 1px solid #d5dbe3 !important;
        z-index: 5 !important;
    }
    /* Equal-width buttons on one row; compact but tappable. */
    #manual-dialog-actions button {
        flex: 1 1 0 !important;
        min-width: 0 !important;
        width: auto !important;
        height: 42px !important;
        padding: 0 6px !important;
        font-size: 13px !important;
        white-space: nowrap !important;
        overflow: hidden !important;
        text-overflow: ellipsis !important;
        touch-action: manipulation !important;
    }
    /* Accordion headers + tab buttons: kill the 300ms tap delay / double-tap
       zoom so taps register immediately (important when the keyboard is
       closing and the layout is still shifting). */
    #manual-tabs button,
    #manual-tabs .tab-nav button,
    #manual-tabs [role="button"] {
        touch-action: manipulation !important;
    }
    /* 键盘呼出时：操作行（Cancel/Clear/Save）让位给输入区，键盘会盖住它；
       键盘收起后自动恢复，并且始终不会被底部流程栏遮挡。 */
    #manual-dialog.kb-open #manual-dialog-actions {
        display: none !important;
    }
    /* Bigger accordion toggle hit-area. */
    #manual-tabs button[aria-expanded] {
        min-height: 44px !important;
    }
    /* The manual-entry tables are Gradio virtual tables. The FIRST column
       (Feature name, e.g. TB-Y) is display-only, but Gradio renders each cell
       as <span role="button">, so swiping over it on a phone triggers an
       unwanted edit and pops the keyboard. Make the whole first column
       non-interactive (pointer-events:none) so only the Value column is
       editable. (Note: Gradio gives Feature cells tabindex="-1" and Value
       cells tabindex="0", so do NOT filter on tabindex.) */
    #manual-tabs tbody td:first-child,
    #manual-tabs tbody td:first-child span,
    #manual-tabs tbody td:first-child span[role="button"],
    #manual-tabs tbody td:first-child .svelte-z9gpua {
        pointer-events: none !important;
        cursor: default !important;
    }
    /* Disable the header sort buttons too on mobile — they are not needed for
       data entry and are easy to hit accidentally while scrolling. */
    #manual-tabs thead th span[role="button"],
    #manual-tabs thead th .sort-button,
    #manual-tabs thead th .svelte-z9gpua {
        pointer-events: none !important;
    }

    /* ===== Wizard step visibility (mobile only) =====
       The actual show/hide is done in JS by toggling inline style.display
       on each tagged element (inline styles always win over Gradio's
       #id CSS, so this is robust). Here we only style the mobile nav. */

    /* ===== Mobile wizard nav bar (fixed bottom) ===== */
    #wizard-nav {
        position: fixed !important;
        left: 0 !important;
        right: 0 !important;
        bottom: 0 !important;
        z-index: 10000 !important;
        display: flex !important;
        align-items: center !important;
        justify-content: space-between !important;
        gap: 8px !important;
        padding: 8px 12px calc(8px + env(safe-area-inset-bottom, 0px)) !important;
        background: #ffffff !important;
        border-top: 1px solid #d5dbe3 !important;
        box-shadow: 0 -2px 10px rgba(15, 40, 80, 0.10) !important;
    }
    #wizard-nav .wizard-dots {
        display: flex !important; align-items: center !important; gap: 6px !important;
        flex: 1 !important; justify-content: center !important;
    }
    #wizard-nav .wizard-dot {
        width: 26px !important; height: 26px !important; border-radius: 50% !important;
        border: 2px solid #c5cdd6 !important; background: #fff !important;
        color: #8a94a0 !important; font-size: 12px !important; font-weight: 700 !important;
        display: inline-flex !important; align-items: center !important; justify-content: center !important;
        cursor: pointer !important;
    }
    #wizard-nav .wizard-dot.done {
        background: #1a73e8 !important; border-color: #1a73e8 !important; color: #fff !important;
    }
    #wizard-nav .wizard-dot.active {
        background: #0d47a1 !important; border-color: #0d47a1 !important; color: #fff !important;
        box-shadow: 0 0 0 3px rgba(26,115,232,0.20) !important;
    }
    #wizard-nav button {
        min-width: 96px !important; height: 40px !important;
        border-radius: 6px !important; font-size: 13px !important; font-weight: 700 !important;
        border: none !important;
    }
    #wizard-nav #wizard-back {
        background: #eef1f6 !important; color: #2c3a4a !important;
    }
    #wizard-nav #wizard-forward {
        background: linear-gradient(135deg, #1a73e8, #1565c0) !important; color: #fff !important;
    }
    #wizard-nav #wizard-forward:disabled,
    #wizard-nav #wizard-back:disabled {
        opacity: 0.4 !important;
    }

    /* Leave bottom room for the fixed wizard nav bar (nav ~56px + padding +
       safe-area). Apply to the container AND its inner scroll region so the
       last content row is never hidden behind the nav. */
    .gradio-container {
        padding-bottom: calc(72px + env(safe-area-inset-bottom, 0px)) !important;
    }
    .gradio-container > .main,
    .gradio-container .contain,
    .gradio-container > main {
        padding-bottom: calc(76px + env(safe-area-inset-bottom, 0px)) !important;
    }
}

/* Desktop (default): never show the mobile nav. Scoped to min-width:769px
   so it cannot override the mobile @media nav rule (equal specificity,
   later source order would otherwise win). Step show/hide is JS-driven and
   only runs on mobile, so desktop content is naturally untouched. */
@media (min-width: 769px) {
    #wizard-nav { display: none !important; }
}
"""

THEMES = {
    "professional": _SHARED_CSS + """
    .gradio-container { background: #f0f3f7 !important; }
    .block, .gr-panel, .form, .panel {
        background: #ffffff !important;
        border: 1px solid #d5dbe3 !important;
        border-radius: 6px !important;
        box-shadow: 0 1px 2px rgba(0,0,0,0.04) !important;
    }
    .gr-input, .gr-text-input input, input[type="text"] {
        border: 1px solid #c5cdd6 !important; border-radius: 5px !important;
        background: #ffffff !important; padding: 5px 8px !important;
    }
    .gr-input:focus, .gr-text-input input:focus, input[type="text"]:focus {
        border-color: #1a73e8 !important;
        box-shadow: 0 0 0 2px rgba(26,115,232,0.15) !important;
        outline: none !important;
    }
    .gr-dropdown {
        border: 1px solid #c5cdd6 !important; border-radius: 5px !important;
        background: #ffffff !important;
    }
    label, .gr-input-label, .gr-box label span { color: #2c3a4a !important; }
    button.primary, .primary-btn, .btn-primary {
        background: linear-gradient(135deg, #1a73e8, #1565c0) !important;
        border: none !important; border-radius: 5px !important;
        box-shadow: 0 1px 3px rgba(26,115,232,0.28) !important;
        color: #fff !important;
    }
    button.primary:hover {
        background: linear-gradient(135deg, #1565c0, #0d47a1) !important;
    }
    button.secondary, .secondary-btn, .btn-secondary {
        background: #ffffff !important; border: 1px solid #c5cdd6 !important;
        border-radius: 5px !important; color: #2c3a4a !important;
    }
    button.secondary:hover { background: #f0f3f7 !important; }
    .gr-checkboxgroup {
        border: 1px solid #c5cdd6 !important; border-radius: 5px !important;
        background: #ffffff !important; padding: 3px 6px !important;
    }
    #overview-img {
        border: 1px solid #c5cdd6 !important; border-radius: 6px !important;
        background: #ffffff !important;
    }
    #csv-upload {
        border: 1.5px dashed #b8c0cc !important; border-radius: 6px !important;
        background: #ffffff !important;
    }
    #csv-upload:hover { border-color: #1a73e8 !important; background: #f0f5ff !important; }
    #shap-plots label, #shap-plots .gr-box label span { color: #0d47a1 !important; }
    .gr-number input {
        border: 1px solid #c5cdd6 !important; border-radius: 5px !important;
        background: #ffffff !important;
    }
    """,

    "minimal": _SHARED_CSS + """
    .gradio-container { background: #f5f5f5 !important; }
    #app-header {
        background: linear-gradient(135deg, #f97316, #ea580c) !important;
    }
    #app-footer .ver { color: #ea580c; }
    button.primary {
        background: linear-gradient(135deg, #f97316, #ea580c) !important;
        color: #fff !important; border: none !important;
    }
    """,
}


def _load_logo_data_uri():
    """Encode logo.png as a base64 data URI for inline use in gradio HTML.

    Returns None if the file is missing — the img tag then hides itself.
    """
    path = os.path.join(BASE_DIR, "logo.png")
    if not os.path.exists(path):
        return None
    import base64
    try:
        with open(path, 'rb') as f:
            data = base64.b64encode(f.read()).decode('ascii')
        return f"data:image/png;base64,{data}"
    except Exception:
        return None


_LOGO_DATA_URI = _load_logo_data_uri()

CUSTOM_CSS = THEMES.get(THEME_NAME, THEMES["professional"])


# ---------- helpers ----------

def _model_display_label(model_name, auc=None):
    """SVM · AUC=0.950  (architecture + AUC only, no date)."""
    if auc is not None:
        try:
            return f"{model_name} · AUC={float(auc):.3f}"
        except Exception:
            pass
    return str(model_name)


def scan_model_dirs(base=None):
    """
    Return list of dicts: {path, label, model_name, auc, dir_name}
    label shown in dropdown = architecture only, e.g. SVM / LightGBM.
    If two folders share the same model_name, disambiguate with date suffix.
    """
    if base is None:
        base = os.path.join(BASE_DIR, "saved_models")
    legacy = os.path.join(BASE_DIR, "saved_model")
    raw = []
    if os.path.isdir(base):
        for name in sorted(os.listdir(base), reverse=True):
            sub = os.path.join(base, name)
            if os.path.isdir(sub) and os.path.exists(os.path.join(sub, "best_model.joblib")):
                model_name, auc = "?", None
                meta = os.path.join(sub, "model_metadata.json")
                if os.path.exists(meta):
                    try:
                        with open(meta) as f:
                            m = json.load(f)
                        model_name = m.get('model_name', '?')
                        auc = m.get('cv_auc', None)
                    except Exception:
                        pass
                raw.append({
                    'path': sub,
                    'model_name': model_name,
                    'auc': auc,
                    'dir_name': name,
                })
    # disambiguate duplicate architecture names
    counts = {}
    for r in raw:
        counts[r['model_name']] = counts.get(r['model_name'], 0) + 1
    dirs = []
    for r in raw:
        if counts[r['model_name']] > 1:
            # keep short date if present so user can tell them apart
            m = re.match(r'^(\d{8})', r['dir_name'])
            date = m.group(1) if m else r['dir_name'][:8]
            label = f"{r['model_name']} ({date})"
        else:
            label = r['model_name']
        dirs.append({
            'path': r['path'],
            'label': label,
            'model_name': r['model_name'],
            'auc': r['auc'],
            'dir_name': r['dir_name'],
        })
    if os.path.isdir(legacy) and os.path.exists(os.path.join(legacy, "best_model.joblib")):
        dirs.append({
            'path': legacy,
            'label': 'legacy',
            'model_name': 'legacy',
            'auc': None,
            'dir_name': 'saved_model',
        })
    return dirs


def _find_model(selected_label):
    for item in scan_model_dirs():
        if item['label'] == selected_label or item['dir_name'] == selected_label \
                or item['model_name'] == selected_label:
            return item
    if selected_label and os.path.isdir(selected_label):
        return {'path': selected_label, 'label': selected_label,
                'model_name': '?', 'auc': None, 'dir_name': selected_label}
    return None


def _find_lightgbm_model():
    """Find the LightGBM model in saved_models (case-insensitive). Returns item or None."""
    for item in scan_model_dirs():
        name = (item.get('model_name') or '').lower()
        if 'lightgbm' in name or 'lgbm' in name or 'lgb' in name:
            return item
    return None


def _auto_load_latest(predictor):
    choices = scan_model_dirs()
    if choices:
        try:
            predictor.model_dir = choices[0]['path']
            predictor.load()
            info = predictor.get_model_info()
            return choices[0], info
        except Exception:
            pass
    return None, None


def _build_save_name(name, pid):
    parts = [p.strip() for p in [name, pid] if p and p.strip()]
    return "_".join(parts) if parts else "result"


def _empty_plot_placeholder(caption="Awaiting SHAP analysis..."):
    """Clean placeholder image for empty SHAP panels (no Gradio giant Empty icon)."""
    from PIL import Image as PILImage, ImageDraw
    w, h = 520, 240
    img = PILImage.new('RGB', (w, h), (250, 251, 252))
    draw = ImageDraw.Draw(img)
    draw.rectangle([0, 0, w - 1, h - 1], outline=(208, 215, 222), width=1)
    # subtle dashed feel via dots
    for x in range(12, w - 12, 10):
        draw.point((x, 12), fill=(220, 224, 230))
        draw.point((x, h - 13), fill=(220, 224, 230))
    try:
        # approximate center text without depending on a system font path
        tw = len(caption) * 6
        draw.text(((w - tw) // 2, h // 2 - 8), caption, fill=(154, 163, 173))
    except Exception:
        pass
    return img


_EMPTY_WATERFALL = None
_EMPTY_IMPORTANCE = None


def _get_empty_waterfall():
    global _EMPTY_WATERFALL
    if _EMPTY_WATERFALL is None:
        _EMPTY_WATERFALL = _empty_plot_placeholder("SHAP Waterfall — run Predict")
    return _EMPTY_WATERFALL


def _get_empty_importance():
    global _EMPTY_IMPORTANCE
    if _EMPTY_IMPORTANCE is None:
        _EMPTY_IMPORTANCE = _empty_plot_placeholder("Feature Importance — run Predict")
    return _EMPTY_IMPORTANCE


def _fig_to_pil(fig):
    """Convert matplotlib figure → PIL Image (for gr.Image display)."""
    if fig is None:
        return None
    try:
        from io import BytesIO
        from PIL import Image as PILImage
        buf = BytesIO()
        fig.savefig(buf, dpi=140, bbox_inches='tight', format='png',
                    facecolor='white', edgecolor='none')
        buf.seek(0)
        img = PILImage.open(buf).convert('RGB')
        # detach from buffer
        out = img.copy()
        img.close()
        return out
    except Exception:
        traceback.print_exc()
        return None


def _fig_to_png_bytes(fig):
    """Serialize a matplotlib figure to PNG bytes without closing caller's ref."""
    if fig is None:
        return None
    try:
        from io import BytesIO
        buf = BytesIO()
        fig.savefig(buf, dpi=150, bbox_inches='tight', format='png')
        return buf.getvalue()
    except Exception:
        return None


def _write_png_bytes(data, path):
    if data:
        try:
            with open(path, 'wb') as f:
                f.write(data)
        except Exception:
            pass


def _b64_to_bytes(data):
    """把 JSON-safe 的 base64 字符串还原成 PNG bytes（配合 state 序列化）。"""
    if not data:
        return None
    try:
        return base64.b64decode(data)
    except Exception:
        return None


def _pick_save_dir_and_name(default_name):
    """Native folder dialog (local desktop); returns (dir, folder_name) or (None, None)."""
    try:
        import tkinter as tk
        from tkinter import filedialog, simpledialog
        root = tk.Tk()
        root.withdraw()
        root.attributes('-topmost', True)
        folder = filedialog.askdirectory(
            title="Select save directory",
            initialdir=APP_DIR if os.path.isdir(APP_DIR) else os.path.expanduser('~'),
        )
        if not folder:
            root.destroy()
            return None, None
        name = simpledialog.askstring(
            "Save Name",
            "Folder / file name:",
            initialvalue=default_name,
            parent=root,
        )
        root.destroy()
        if not name or not name.strip():
            return None, None
        return folder, name.strip()
    except Exception as e:
        print(f"Native dialog unavailable: {e}")
        return None, None


def create_demo():

    predictor = DiagnosisPredictor()
    latest_item, latest_info = _auto_load_latest(predictor)

    HEAD_OVERRIDE = """
    <meta name="viewport" content="width=device-width, initial-scale=1, viewport-fit=cover" />
    <meta name="mobile-web-app-capable" content="yes" />
    <meta name="apple-mobile-web-app-capable" content="yes" />
    <meta name="apple-mobile-web-app-status-bar-style" content="black-translucent" />
    <meta name="apple-mobile-web-app-title" content="LungCancerAI" />
    <meta name="theme-color" content="#0d47a1" />
    <script>
    Object.defineProperty(navigator, 'language', { get: () => 'en' });
    Object.defineProperty(navigator, 'languages', { get: () => ['en', 'en-US'] });
    </script>
    <script>
    /* Mobile wizard step controller.
       - Only activates on narrow screens (<=768px). On desktop it's a no-op,
         so the desktop one-screen layout is untouched.
       - Show/hide is done by toggling inline style.display on each tagged
         element. Inline styles always beat Gradio's #id CSS, which made a
         pure-CSS approach unreliable.
       - No Gradio state is involved, so switching steps never loses the
         user's inputs or prediction results.

       Step -> elements (by id, or by class for the name/ID row):
         Step 1 (input):  input-row, model-row, action-row, name/ID row(.wizard-step-1),
                          Steps checkbox(.wizard-step-1)
         Step 2 (result): risk-html, prob-html, guidance-html, overview-img, decision-html
         Step 3 (SHAP):   shap-plots
       The two top columns also tagged wizard-col-left / wizard-col-right are
       hidden wholesale when their column has no active content for the step.
    */
    (function () {
        var MOBILE_MQ = window.matchMedia('(max-width: 768px)');
        var navReady = false;
        var STEP_IDS = {
            1: ['input-row', 'model-row', 'action-row'],
            2: ['risk-html', 'prob-html', 'guidance-html', 'overview-img', 'decision-html'],
            3: ['shap-plots']
        };
        // The name/ID row and the Steps CheckboxGroup have no elem_id; they
        // carry the class wizard-step-1 and belong to step 1.
        var STEP1_CLASS = 'wizard-step-1';

        function isMobile() { return MOBILE_MQ.matches; }
        function currentStep() {
            var b = document.body.getAttribute('data-wizard-step');
            return b ? parseInt(b, 10) : 0;
        }

        function elById(id) { return document.getElementById(id); }
        function step1ClassEls() { return Array.prototype.slice.call(document.getElementsByClassName(STEP1_CLASS)); }
        function leftCol() { return document.querySelector('.wizard-col-left'); }
        function rightCol() { return document.querySelector('.wizard-col-right'); }

        // Remember each element's original display so we can restore it on
        // desktop / when shown. Inline display:'' falls back to stylesheet.
        function showEl(el) { if (el) el.style.display = ''; }
        function hideEl(el) { if (el) el.style.display = 'none'; }

        function applyStep(n) {
            // Hide everything wizard-tagged, then show only the current step.
            var allIds = STEP_IDS[1].concat(STEP_IDS[2], STEP_IDS[3]);
            for (var i = 0; i < allIds.length; i++) hideEl(elById(allIds[i]));
            step1ClassEls().forEach(hideEl);
            // Columns: default to visible; specific steps hide one wholesale.
            showEl(leftCol()); showEl(rightCol());

            var ids = STEP_IDS[n] || [];
            for (var j = 0; j < ids.length; j++) showEl(elById(ids[j]));
            if (n === 1) step1ClassEls().forEach(showEl);

            // Step 1 has no result content -> hide right column.
            if (n === 1) hideEl(rightCol());
            // Step 3 has no left-column content -> hide left column.
            if (n === 3) hideEl(leftCol());
        }

        function setStep(n) {
            n = parseInt(n, 10) || 1;
            if (n < 1) n = 1;
            if (n > 3) n = 3;
            document.body.setAttribute('data-wizard-step', String(n));
            if (isMobile()) {
                applyStep(n);
                updateNav(n);
                window.scrollTo({ top: 0, behavior: 'smooth' });
            }
        }

        function updateNav(n) {
            var nav = document.getElementById('wizard-nav');
            if (!nav) return;
            nav.style.display = isMobile() ? 'flex' : 'none';
            var dots = nav.querySelectorAll('.wizard-dot');
            dots.forEach(function (d) {
                var s = parseInt(d.getAttribute('data-step'), 10);
                d.classList.toggle('active', s === n);
                d.classList.toggle('done', s < n);
            });
            var back = nav.querySelector('#wizard-back');
            var fwd  = nav.querySelector('#wizard-forward');
            if (back) back.disabled = (n <= 1);
            if (fwd)  fwd.disabled  = (n >= 3);
            if (fwd) {
                fwd.textContent = (n === 1) ? 'Predict \u2192 Result'
                               : (n === 2) ? 'View SHAP \u2192'
                               : 'Back to Result \u2190';
            }
        }

        // Exposed for Gradio js= callbacks (Predict button -> step 2) and nav.
        window.wizardGo = function (n) { setStep(n); return n; };

        function restoreAll() {
            // Desktop / deactivated: clear every inline display we set.
            var allIds = STEP_IDS[1].concat(STEP_IDS[2], STEP_IDS[3]);
            for (var i = 0; i < allIds.length; i++) showEl(elById(allIds[i]));
            step1ClassEls().forEach(showEl);
            showEl(leftCol()); showEl(rightCol());
        }

        function refreshActivation() {
            if (isMobile()) {
                if (!currentStep()) {
                    document.body.setAttribute('data-wizard-step', '1');
                    applyStep(1);
                    updateNav(1);
                } else {
                    applyStep(currentStep());
                    updateNav(currentStep());
                }
            } else {
                restoreAll();
                updateNav(0);
            }
        }

        function bindNav() {
            if (navReady) return;
            var nav = document.getElementById('wizard-nav');
            if (!nav) return;
            nav.querySelector('#wizard-back').addEventListener('click', function () {
                setStep(Math.max(1, currentStep() - 1));
            });
            nav.querySelector('#wizard-forward').addEventListener('click', function () {
                // On step 1 the real Predict button (with its js= callback) is
                // the primary path that also runs the backend prediction. This
                // nav forward button still advances visually for convenience.
                setStep(Math.min(3, currentStep() + 1));
            });
            nav.querySelectorAll('.wizard-dot').forEach(function (d) {
                d.addEventListener('click', function () {
                    setStep(parseInt(d.getAttribute('data-step'), 10));
                });
            });
            navReady = true;
        }

        function init() {
            bindNav();
            refreshActivation();
        }

        if (document.readyState === 'loading') {
            document.addEventListener('DOMContentLoaded', init);
        } else {
            init();
        }
        // Gradio re-renders parts of the DOM on interaction; rebind/reapply
        // lazily until the nav is bound and the current step is reflected.
        var pollCount = 0;
        var boundPoll = setInterval(function () {
            pollCount++;
            if (!navReady) bindNav();
            if (isMobile() && currentStep()) applyStep(currentStep());
            if (navReady && pollCount > 3) clearInterval(boundPoll);
        }, 1000);
        setTimeout(function () { clearInterval(boundPoll); }, 15000);

        try {
            MOBILE_MQ.addEventListener('change', refreshActivation);
        } catch (e) {
            MOBILE_MQ.addListener(refreshActivation);
        }
        window.addEventListener('resize', refreshActivation);
    })();
    </script>
    <script>
    /* Manual-entry dialog: mobile keyboard/tap hardening.
       Two real-phone problems are handled here:

       1) After typing a value, the mobile keyboard stays open. The first tap
          outside the input (e.g. the accordion header arrow) closes the
          keyboard, the visual viewport expands and the sheet shifts, so the
          browser cancels the click and the tap "does nothing".
       2) While the keyboard is open the bottom part of the fixed sheet
          (Save row) is covered; users must close the keyboard first.

       Fix strategy (mobile widths only, desktop untouched):
       - On pointerdown outside an editable field: blur the focused input
         immediately so the keyboard starts closing BEFORE the tap is
         processed, then remember the intended target.
       - The native click wins when it lands on the intended target. If it is
         eaten by the layout shift, or lands on a NEIGHBOURING element that
         moved under the finger (e.g. accordion A collapses and B slides up),
         the click is intercepted and re-dispatched to the original target,
         so one tap never opens a different accordion. A short coordinate
         guard then swallows any leftover late click.
       - Lock body scrolling while the sheet is open (some WebViews scroll the
         whole page when the keyboard opens, which moves the fixed sheet).
       - While the sheet is open, add .nav-hidden / .dialog-open so CSS hides
         the bottom wizard nav and lets the sheet fill the screen (no overlap
         with the Save row at all). */
    (function () {
        var MOBILE = window.matchMedia('(max-width: 768px)');
        if (!MOBILE.matches) return;

        var dialog = null;
        var pending = null;       // { target, expires }
        var bodyLocked = false;
        var started = false;
        var lateGuard = null;     // { x, y, seq, until } - 补发后迟到的原生 click 压制
        var gestureSeq = 0;       // 每次按下 +1，用于区分“同一次手势的迟到 click”
                                  // 和“用户下一次全新的点击”
        var kbBase = null;        // 键盘收起时的视口高度基准，用于识别键盘呼出
        var HAS_POINTER = !!(window.PointerEvent);

        function isEditable(el) {
            return !!(el && (el.tagName === 'INPUT' || el.tagName === 'TEXTAREA' ||
                             el.tagName === 'SELECT' ||
                             (el.isContentEditable)));
        }

        function setOpen(open) {
            var on = !!open;
            if (on !== bodyLocked) {
                bodyLocked = on;
                document.body.setAttribute('data-manual-open', on ? '1' : '0');
                // Class-based CSS hooks (Gradio scopes injected CSS, so a
                // body[data-...] selector would not survive the rewrite).
                var nav = document.getElementById('wizard-nav');
                if (nav) nav.classList.toggle('nav-hidden', on);
                dialog.classList.toggle('dialog-open', on);
                // Lock body scroll so the keyboard/page cannot shift the
                // fixed sheet on WebViews where fixed positioning breaks.
                document.body.style.overflow = on ? 'hidden' : '';
                if (on) {
                    document.body.style.touchAction = 'none';
                    window.scrollTo(0, 0);
                } else {
                    document.body.style.touchAction = '';
                }
            }
        }

        function dialogVisible() {
            // Gradio may re-render and replace the Column element on server
            // round-trips; re-resolve it so the handlers keep working.
            var el = document.getElementById('manual-dialog');
            if (el && el !== dialog) dialog = el;
            return !!(el && getComputedStyle(el).display !== 'none');
        }

        function refreshOpen() {
            setOpen(dialogVisible());
            kbCheck();
        }

        // 检测手机键盘是否呼出：
        // - overlay 模式（多数手机浏览器）：可视视口明显小于布局视口；
        // - adjustResize 模式的 WebView：布局视口相对基准明显缩小。
        // 检测到键盘呼出后，给弹窗加 .kb-open，CSS 会把操作行隐藏，
        // 把高度让给输入表格；键盘收起后自动恢复。
        function kbCheck() {
            if (!dialog) return;
            var vv = window.visualViewport;
            var nowH = window.innerHeight;
            var kbOpen = (vv && vv.height + 60 < nowH) ||
                         (kbBase && nowH + 80 < kbBase);
            dialog.classList.toggle('kb-open', kbOpen && dialogVisible());
            if (!kbOpen && dialogVisible()) {
                kbBase = nowH;
            }
        }

        // 从 pointer / touch 事件中取坐标（兼容不支持 PointerEvent 的 WebView）
        function evPoint(e, kind) {
            if (HAS_POINTER) return { x: e.clientX, y: e.clientY };
            var t = kind === 'move' ? e.touches[0] : e.changedTouches[0];
            return t ? { x: t.clientX, y: t.clientY } : { x: 0, y: 0 };
        }

        function closestStable(el) {
            return el.closest('button, [role="button"], td, [role="gridcell"], label') || el;
        }

        // 是“控件”还是“表格单元格”？单元格交给 Gradio 自己的编辑流程，
        // 我们的补点/拦截逻辑不碰它，避免破坏单元格点击编辑。
        function isControl(el) {
            return !!(el && (el.tagName === 'BUTTON' || el.tagName === 'LABEL' ||
                             (el.getAttribute && el.getAttribute('role') === 'button')));
        }

        function init() {
            if (started) return;
            dialog = document.getElementById('manual-dialog');
            if (!dialog) return;
            started = true;

            // Gradio shows/hides the Column by toggling the `hide` class.
            var mo = new MutationObserver(refreshOpen);
            mo.observe(dialog, { attributes: true, attributeFilter: ['class', 'style'] });
            refreshOpen();
            kbCheck();

            // 键盘呼出/收起时视口尺寸会变化，实时刷新 .kb-open 状态
            if (window.visualViewport) {
                window.visualViewport.addEventListener('resize', kbCheck);
                window.visualViewport.addEventListener('scroll', kbCheck);
            }
            window.addEventListener('resize', kbCheck);

            var DOWN = HAS_POINTER ? 'pointerdown' : 'touchstart';
            var MOVE = HAS_POINTER ? 'pointermove' : 'touchmove';
            var UP = HAS_POINTER ? 'pointerup' : 'touchend';
            var CANCEL = HAS_POINTER ? 'pointercancel' : 'touchcancel';

            function samePoint(a, b, tol) {
                return Math.abs(a.x - b.x) <= tol && Math.abs(a.y - b.y) <= tol;
            }

            // 统一补发：原生 click 没到（被键盘/布局位移吃掉）或落错目标时，
            // 在延迟后把 click 派发给最初按下的目标。
            function scheduleRedispatch(ms) {
                setTimeout(function () {
                    var q = pending;
                    pending = null;
                    if (q && Date.now() <= q.expires && q.target &&
                        q.target.isConnected && dialogVisible()) {
                        q.target.dispatchEvent(new MouseEvent('click', {
                            bubbles: true,
                            cancelable: true,
                            view: window
                        }));
                        // 补发后，迟到的原生 click 仍可能到达（键盘/视口动画
                        // 把它推迟了），而且布局位移后可能落在别的元素上；
                        // 短时间内把按下位置附近的 click 全部拦下，避免
                        // “点一个收起、相邻的另一个却打开”。
                        lateGuard = { x: q.x, y: q.y, seq: gestureSeq, until: Date.now() + 1500 };
                    }
                }, ms);
            }

            document.addEventListener(DOWN, function (e) {
                gestureSeq++;
                if (!dialogVisible() || !dialog.contains(e.target)) return;
                // 点击输入框/输入控件本身：不干预。
                if (isEditable(e.target)) return;
                // 值单元格（可编辑格子）完全交给 Gradio 自己的
                // touchstart -> 编辑 -> focus 流程处理：我们既不 blur 也不
                // 补点，避免程序性 blur 和键盘收起动画打架，导致
                // “键盘闪一下就消失”。
                if (e.target.closest && e.target.closest('td[tabindex="0"]')) return;
                // 手风琴头 / 选项卡 / 操作按钮等控件：按下瞬间先收起键盘
                // （在布局位移之前），再走补点逻辑。
                var ae = document.activeElement;
                if (ae && dialog.contains(ae) && isEditable(ae) && ae !== e.target) {
                    try { ae.blur(); } catch (err) {}
                }
                var stable = closestStable(e.target);
                if (!isControl(stable)) return;
                var p = evPoint(e, 'down');
                pending = {
                    target: stable,
                    x: p.x,
                    y: p.y,
                    moved: false,
                    control: true,
                    expires: Date.now() + 900
                };
            }, true);

            // 滑动/拖动时取消补点，避免滚动弹窗误触发控件
            document.addEventListener(MOVE, function (e) {
                if (!pending) return;
                var p = evPoint(e, 'move');
                pending.moved = true;
                if (Math.abs(p.x - pending.x) > 10 || Math.abs(p.y - pending.y) > 10) {
                    pending = null;
                }
            }, true);

            document.addEventListener('click', function (e) {
                // 1) 补发之后迟到的原生 click：仍在按下点附近就拦下
                if (lateGuard) {
                    var g = lateGuard;
                    if (g.seq === gestureSeq && Date.now() <= g.until &&
                        samePoint({ x: e.clientX, y: e.clientY }, g, 60)) {
                        e.stopImmediatePropagation();
                        e.preventDefault();
                        lateGuard = null;
                        return;
                    }
                    if (Date.now() > g.until) lateGuard = null;
                }
                if (pending && Date.now() <= pending.expires) {
                    var t = pending.target;
                    // 2) 原生 click 落在预期目标上 -> 正常执行，取消补发
                    if (e.target === t || (t.contains && t.contains(e.target))) {
                        pending = null;
                        return;
                    }
                    // 3) 布局位移让 click 落到了别的元素上，但仍在按下位置
                    //    附近（例如 A 收起后相邻的 B 移到手指下方）：
                    //    拦下它，稍后补发到原目标，避免误开相邻维度。
                    if (pending.control &&
                        samePoint({ x: e.clientX, y: e.clientY }, pending, 60)) {
                        e.stopImmediatePropagation();
                        e.preventDefault();
                        return;
                    }
                    // 无关的点击（按下后又点了别处）：放弃补发
                    pending = null;
                }
            }, true);

            document.addEventListener(UP, function () {
                if (!pending) return;
                scheduleRedispatch(180);
            }, true);

            // 取消手势：如果只是视图层被系统打断（例如键盘收起触发 viewport
            // 变化导致 touchcancel），仍按“点击”补发；若是拖动产生的取消则跳过。
            document.addEventListener(CANCEL, function () {
                if (!pending) return;
                if (pending.moved) { pending = null; return; }
                scheduleRedispatch(120);
            }, true);

            // 长按/右键菜单：取消所有待办，避免误触发
            document.addEventListener('contextmenu', function () { pending = null; }, true);
        }

        // This script runs from <head>, before Gradio has mounted the
        // components, so `#manual-dialog` does not exist yet. Defer until the
        // DOM is ready and retry until the dialog appears (same pattern as
        // the wizard controller above).
        if (document.readyState === 'loading') {
            document.addEventListener('DOMContentLoaded', init);
        } else {
            init();
        }
        var pollCount = 0;
        var poll = setInterval(function () {
            pollCount++;
            init();
            if (started) refreshOpen();
            if (started && pollCount > 3) clearInterval(poll);
        }, 1000);
        setTimeout(function () { clearInterval(poll); }, 15000);
    })();
    </script>
    """

    # ===== PWA 注入（仅当以 PWA 模式启动时）=====
    # manifest 让安卓/iOS 出现“添加到主屏幕”；apple-touch-icon 供 iOS
    # 独立图标使用；favicon 直接指向 pwa 图标。SW 注册在 load 之后，
    # 桌面 exe / share 隧道 / 未部署 pwa/ 时这里整个块为空，行为与原来
    # 完全一致（__main__ 的 PWA 分支会先设置 PWA_SERVE=1）。
    if PWA_AVAILABLE and os.environ.get("PWA_SERVE") == "1":
        HEAD_OVERRIDE += """
    <link rel="manifest" href="/manifest.json" />
    <link rel="icon" type="image/png" href="/pwa/icons/favicon-32.png" />
    <link rel="apple-touch-icon" href="/pwa/icons/apple-touch-icon-180.png" />
    <script>
    if ('serviceWorker' in navigator) {
        window.addEventListener('load', function () {
            navigator.serviceWorker.register('/sw.js').catch(function (err) {
                console.warn('SW register failed:', err);
            });
        });
    }
    </script>
    """

    with gr.Blocks(css=CUSTOM_CSS, title="Lung Cancer AI Warning System",
                    head=HEAD_OVERRIDE) as demo:

        # last prediction payload for deferred Save
        result_state = gr.State(None)

        # ===== Header =====
        logo_src = _LOGO_DATA_URI or ""
        logo_html = (
            f'<img class="brand-logo" src="{logo_src}" alt="Logo"'
            f' onerror="this.style.display=\'none\'"/>'
            if logo_src else ''
        )
        gr.HTML(f"""
        <div id="app-header">
          <div class="brand">
            {logo_html}
            <div class="brand-text">
              <h1>Lung Cancer Early Screening AI Warning System</h1>
              <div class="sub">Multi-modal TCM Four-Diagnosis · Risk Stratification · SHAP Explainability</div>
            </div>
          </div>
          <div class="meta">
            Clinical Decision Support
            <span class="badge">v1.0</span><br/>
            Reference Only
          </div>
        </div>
        """)

        with gr.Row():
            # ===== LEFT COLUMN =====
            with gr.Column(scale=1, min_width=300, elem_classes="wizard-col-left"):

                # Name + ID only (Age removed)
                with gr.Row(elem_classes="wizard-step-1"):
                    patient_name = gr.Textbox(label="Name", placeholder="e.g. Zhang", scale=1)
                    patient_id = gr.Textbox(label="ID", placeholder="LCS-001", scale=1)

                with gr.Row(elem_id="input-row", elem_classes="wizard-step-1"):
                    csv_upload = gr.File(label="Upload CSV", file_types=[".csv"],
                                         file_count="single", height=78,
                                         elem_id="csv-upload", scale=3,
                                         min_width=120)
                    manual_entry_btn = gr.Button(
                        "Manual\nEntry", variant="secondary", size="sm",
                        scale=1, elem_id="btn-manual", min_width=72,
                    )
                    view_edit_btn = gr.Button(
                        "View /\nEdit", variant="secondary", size="sm",
                        scale=1, elem_id="btn-view", min_width=72,
                    )

                manual_state = gr.State(value={})

                model_items = scan_model_dirs()
                choice_labels = [c['label'] for c in model_items]
                if latest_info and latest_item:
                    auc = latest_info.get('cv_auc', latest_item.get('auc'))
                    model_label = "Model: " + _model_display_label(
                        latest_item['model_name'], auc)
                    default_model = latest_item['label']
                else:
                    model_label = "Model: none"
                    default_model = None
                with gr.Row(elem_id="model-row", elem_classes="wizard-step-1"):
                    model_dropdown = gr.Dropdown(
                        choices=choice_labels,
                        label=model_label,
                        value=default_model,
                        allow_custom_value=True,
                        scale=6,
                        min_width=160,
                        container=True,
                    )
                    load_model_btn = gr.Button(
                        "Switch", variant="secondary", scale=1, size="sm",
                        min_width=64, elem_id="btn-switch",
                    )

                steps = gr.CheckboxGroup(
                    choices=["Predict", "SHAP Analysis", "SHAP Plot"],
                    value=["Predict", "SHAP Analysis", "SHAP Plot"],
                    label="Steps",
                    elem_classes="wizard-step-1",
                )

                # Reset | Predict (center, larger) | Save
                with gr.Row(elem_id="action-row", elem_classes="wizard-step-1"):
                    reset_btn = gr.Button("Reset", variant="secondary",
                                          scale=1, elem_id="btn-reset", min_width=60)
                    predict_btn = gr.Button("Predict", variant="primary",
                                            scale=3, elem_id="btn-predict")
                    save_btn = gr.Button("Save", variant="secondary",
                                         scale=1, elem_id="btn-save", min_width=60)

                risk_html = gr.HTML(
                    '<div style="background:#e8edf3; padding:10px 14px; border-radius:5px; '
                    'text-align:center; color:#8a94a0; font-size:15px; border:1px solid #d5dbe3;">'
                    'Awaiting prediction...</div>',
                    elem_id="risk-html",
                    elem_classes="wizard-step-2",
                )
                prob_display = gr.HTML(
                    '<div style="display:flex; flex-wrap:wrap; justify-content:center; gap:18px; font-size:14px; '
                    'color:#8a94a0; padding:5px 0; border:1px solid #d5dbe3; border-radius:5px; '
                    'margin:2px 0; background:#fff;">'
                    '<span><b>Raw Prob:</b> —</span>'
                    '<span><b>Cal Prob:</b> —</span>'
                    '</div>',
                    elem_id="prob-html",
                    elem_classes="wizard-step-2",
                )

                health_guidance = gr.HTML(
                    '<div style="border:1px solid #d5dbe3; border-radius:6px; padding:10px; '
                    'color:#8a94a0; font-size:14px; text-align:center; background:#fafbfc;">'
                    'Health guidance will appear after prediction...</div>',
                    elem_id="guidance-html",
                    elem_classes="wizard-step-2",
                )

            # ===== RIGHT COLUMN =====
            with gr.Column(scale=2, min_width=580, elem_classes="wizard-col-right"):

                gr.Image(
                    value=os.path.join(BASE_DIR, "assets", "overview.png"),
                    elem_id="overview-img",
                    show_label=False, interactive=False, height=220,
                    show_download_button=False, show_share_button=False,
                    elem_classes="wizard-step-2",
                )

                decision_explanation = gr.HTML(
                    '<div id="decision-box" style="border:1px solid #d5dbe3; border-radius:6px; '
                    'padding:8px; color:#8a94a0; font-size:14px; text-align:center; '
                    'min-height:231px; display:flex; align-items:center; justify-content:center; '
                    'background:#fafbfc;">'
                    'Decision explanation will appear here...</div>',
                    elem_id="decision-html",
                    elem_classes="wizard-step-2",
                )

                # No placeholder images — empty until Predict generates figures.
                # Labels merged into Image box; CSS adds top offset for figure.
                with gr.Row(elem_id="shap-plots", elem_classes="wizard-step-3"):
                    shap_waterfall = gr.Image(
                        value=None,
                        label="SHAP Waterfall",
                        show_label=True, interactive=False, height=260,
                        elem_id="plot-waterfall",
                        show_download_button=False, show_share_button=False,
                        container=True,
                    )
                    shap_importance = gr.Image(
                        value=None,
                        label="Feature Importance",
                        show_label=True, interactive=False, height=260,
                        elem_id="plot-importance",
                        show_download_button=False, show_share_button=False,
                        container=True,
                    )

        # ===== Footer =====
        gr.HTML("""
        <div id="app-footer">
          <span>© Lung Cancer Early Screening AI · For research &amp; screening reference only · Not a clinical diagnosis</span>
          <span class="ver">CDS Platform · v1.0</span>
        </div>
        """)

        # ===== Mobile wizard nav (fixed bottom bar) =====
        # Hidden on desktop via CSS (#wizard-nav { display:none }); JS shows it
        # only on mobile. The Back/Forward buttons drive step switching; on
        # step 1 the real Predict button (with its js= callback) is the primary
        # way to run prediction AND advance to step 2.
        gr.HTML("""
        <nav id="wizard-nav" aria-label="Step navigation">
          <button id="wizard-back" type="button" aria-label="Previous step">&#8592; Back</button>
          <div class="wizard-dots">
            <span class="wizard-dot" data-step="1" title="Input">1</span>
            <span class="wizard-dot" data-step="2" title="Risk Result">2</span>
            <span class="wizard-dot" data-step="3" title="SHAP">3</span>
          </div>
          <button id="wizard-forward" type="button">Predict &#8594; Result</button>
        </nav>
        """)

        # ===== Manual Entry Dialog (fallback: visible-toggle Column) =====
        # Per 特征.xlsx: 3 outer dimensions (舌/面/脉) + 1 "Other" tab for the 23
        # model features not in the user's table. Each subgroup uses an Accordion.
        with gr.Column(visible=False, elem_id="manual-dialog") as manual_dialog:
            gr.HTML(
                '<div style="display:flex; justify-content:space-between; '
                'align-items:center; margin-bottom:6px;">'
                '<b style="font-size:15px; color:#1a73e8;">Manual Feature Entry</b>'
                '<span style="font-size:11.5px; color:#8a94a0;">'
                'Empty cells are imputed with training mean · '
                f'{len(DIMENSION_FEATURES)} of {len(FEATURE_NAMES)} features listed</span>'
                '</div>'
            )
            subgroup_tables = {}
            with gr.Tabs(elem_id="manual-tabs"):
                for dim in FEATURE_DIMENSIONS:
                    tab_label = f"{dim['display']} ({dim['display_cn']})"
                    with gr.Tab(tab_label):
                        for sg_idx, sg in enumerate(dim['subgroups']):
                            feats = sg['features']
                            label = (f"{sg['display']} ({sg['display_cn']}) · "
                                     f"{len(feats)} features")
                            with gr.Accordion(label, open=(sg_idx == 0)):
                                subgroup_tables[sg['key']] = gr.Dataframe(
                                    value=pd.DataFrame({"Feature": feats,
                                                        "Value": [None] * len(feats)}),
                                    headers=["Feature", "Value"],
                                    datatype=["str", "number"],
                                    col_count=(2, "fixed"),
                                    row_count=(len(feats), "fixed"),
                                    interactive=True, wrap=True,
                                    elem_id=f"tbl-{sg['key']}",
                                )
                # "Other" tab: 23 model features not in user's Excel
                with gr.Tab(f"Other ({len(EXTRA_MODEL_FEATURES)} extra)"):
                    gr.Markdown(
                        f"_These {len(EXTRA_MODEL_FEATURES)} RGB-channel features are "
                        "used by the model but not listed in your dimension table. "
                        "Leave blank to auto-impute with training mean._"
                    )
                    with gr.Accordion(
                        f"Extra Model Features · {len(EXTRA_MODEL_FEATURES)} features",
                        open=False,
                    ):
                        other_table = gr.Dataframe(
                            value=pd.DataFrame({"Feature": EXTRA_MODEL_FEATURES,
                                                "Value": [None] * len(EXTRA_MODEL_FEATURES)}),
                            headers=["Feature", "Value"],
                            datatype=["str", "number"],
                            col_count=(2, "fixed"),
                            row_count=(len(EXTRA_MODEL_FEATURES), "fixed"),
                            interactive=True, wrap=True, elem_id="tbl-other",
                        )
            with gr.Row(elem_id="manual-dialog-actions"):
                cancel_btn = gr.Button("Cancel", variant="secondary",
                                       elem_id="btn-manual-cancel")
                clear_btn = gr.Button("Clear All", variant="secondary",
                                      elem_id="btn-manual-clear")
                save_btn_dialog = gr.Button("Save", variant="primary",
                                            elem_id="btn-manual-save")

        # ===== EVENT HANDLERS =====

        def handle_load_model(selected_label):
            item = _find_model(selected_label)
            if item is None:
                return gr.update(label=f"Model: not found")
            try:
                predictor.model_dir = item['path']
                predictor.load()
                info = predictor.get_model_info()
                auc = info.get('cv_auc', item.get('auc'))
                return gr.update(
                    label="Model: " + _model_display_label(item['model_name'], auc)
                )
            except Exception as e:
                return gr.update(label=f"Model: load failed — {e}")

        def handle_predict(file, selected_steps, patient_name_val, id_val,
                           manual_data=None):
            # 防御：勾选框在某些情况下可能为 None，避免 “NoneType is not iterable”
            selected_steps = selected_steps or []
            _PROB_PLACEHOLDER = (
                '<div style="display:flex; flex-wrap:wrap; justify-content:center; gap:18px; font-size:14px; '
                'color:#8a94a0; padding:5px 0; border:1px solid #d5dbe3; border-radius:5px; '
                'margin:2px 0; background:#fff;">'
                '<span><b>Raw Prob:</b> —</span>'
                '<span><b>Cal Prob:</b> —</span>'
                '</div>'
            )
            empty_state = None

            if not predictor.is_loaded:
                return (
                    '<div style="background:#e74c3c; color:white; padding:8px; '
                    'border-radius:5px; text-align:center; font-size:14px;">Please load model first</div>',
                    _PROB_PLACEHOLDER, "", None, None, "", empty_state,
                )
            if file is None and not manual_data:
                return (
                    '<div style="background:#f39c12; color:white; padding:8px; '
                    'border-radius:5px; text-align:center; font-size:14px;">Please upload CSV or enter manual data</div>',
                    _PROB_PLACEHOLDER, "", None, None, "", empty_state,
                )
            try:
                # Manual mode: force-load LightGBM, skip imputation
                use_no_impute = bool(manual_data) and file is None
                if use_no_impute:
                    lgb_item = _find_lightgbm_model()
                    if lgb_item is None:
                        return (
                            '<div style="background:#e74c3c; color:white; padding:8px; '
                            'border-radius:5px; text-align:center; font-size:14px;">'
                            'Manual entry requires LightGBM model but none was found in saved_models.</div>',
                            _PROB_PLACEHOLDER, "", None, None, "", empty_state,
                        )
                    if (predictor.model_dir != lgb_item['path']) or not predictor.is_loaded:
                        predictor.model_dir = lgb_item['path']
                        predictor.load()

                if file is not None:
                    df = pd.read_csv(file.name)
                    df_features = df.drop(columns=[TARGET_COLUMN]) if TARGET_COLUMN in df.columns else df
                else:
                    row = {f: manual_data.get(f) for f in FEATURE_NAMES}
                    df_features = pd.DataFrame([row])

                if use_no_impute:
                    predictions, risks = predictor.predict_batch_no_impute(df_features)
                else:
                    predictions, risks = predictor.predict_batch(df_features)
                first_result = risks[0]
                risk_display = get_risk_html(first_result)

                waterfall_fig = None
                importance_fig = None
                decision_html = ""
                shap_needed = "SHAP Analysis" in selected_steps or "SHAP Plot" in selected_steps
                if shap_needed:
                    try:
                        first_patient = df_features.iloc[0].to_dict()
                        explanation = explain_single_patient(
                            predictor.pipeline, first_patient,
                            predictor.shap_background,
                            skip_preprocess=use_no_impute,
                        )
                        label = patient_name_val or "Patient #1"
                        if "SHAP Plot" in selected_steps:
                            waterfall_fig = generate_waterfall_plot(explanation, label)
                            importance_fig = generate_feature_importance_bar(explanation)
                        if "SHAP Analysis" in selected_steps:
                            decision_html = generate_decision_explanation(explanation, label)
                    except Exception as e:
                        print(f"SHAP analysis failed: {e}")
                        traceback.print_exc()

                for col in ['Raw_Probability', 'Calibrated_Probability']:
                    if col in predictions.columns:
                        predictions[col] = predictions[col].round(3)
                result_df = pd.concat([df_features.reset_index(drop=True),
                                       predictions.reset_index(drop=True)], axis=1)
                default_name = _build_save_name(patient_name_val, id_val)
                # state 会被 Gradio 队列 JSON 序列化：DataFrame/bytes 直接放进去
                # 会导致结果发不回前端、Save 也用不了。这里转成 JSON 安全格式。
                state = {
                    'result_df': result_df.to_json(orient='split') if result_df is not None else None,
                    'waterfall_png': base64.b64encode(_fig_to_png_bytes(waterfall_fig)).decode('ascii') if waterfall_fig is not None else None,
                    'importance_png': base64.b64encode(_fig_to_png_bytes(importance_fig)).decode('ascii') if importance_fig is not None else None,
                    'default_name': default_name,
                }

                guidance_html = get_guidance_html(first_result['level'])

                raw_val = round(float(predictions['Raw_Probability'].iloc[0]), 3)
                cal_val = round(float(predictions['Calibrated_Probability'].iloc[0]), 3)
                prob_html = (
                    f'<div style="display:flex; flex-wrap:wrap; justify-content:center; gap:18px; font-size:14px; '
                    f'padding:5px 0; border:1px solid #d5dbe3; border-radius:5px; margin:2px 0; background:#fff;">'
                    f'<span><b>Raw Prob:</b> {raw_val:.3f}</span>'
                    f'<span><b>Cal Prob:</b> {cal_val:.3f}</span>'
                    f'</div>'
                )

                # Convert figures → PIL; None keeps panel empty (no placeholder image)
                wf_img = _fig_to_pil(waterfall_fig)
                im_img = _fig_to_pil(importance_fig)

                try:
                    if waterfall_fig is not None:
                        plt.close(waterfall_fig)
                    if importance_fig is not None:
                        plt.close(importance_fig)
                except Exception:
                    pass

                return (
                    risk_display,
                    prob_html,
                    guidance_html,
                    wf_img,
                    im_img,
                    decision_html,
                    state,
                )
            except Exception as e:
                traceback.print_exc()
                return (
                    f'<div style="background:#e74c3c; color:white; padding:8px; '
                    f'border-radius:5px; font-size:14px;">Error: {e}</div>',
                    _PROB_PLACEHOLDER, "", None, None, "", empty_state,
                )

        def handle_save(state):
            if not state:
                return
            default_name = state.get('default_name') or 'result'
            folder, name = _pick_save_dir_and_name(default_name)
            if not folder or not name:
                return
            try:
                save_dir = os.path.join(folder, name)
                os.makedirs(save_dir, exist_ok=True)
                result_df = state.get('result_df')
                if result_df:
                    pd.read_json(result_df, orient='split').to_csv(
                        os.path.join(save_dir, 'predictions.csv'),
                        index=False, encoding='utf-8-sig')
                _write_png_bytes(_b64_to_bytes(state.get('waterfall_png')),
                                 os.path.join(save_dir, 'shap_waterfall.png'))
                _write_png_bytes(_b64_to_bytes(state.get('importance_png')),
                                 os.path.join(save_dir, 'shap_importance.png'))
            except Exception as e:
                traceback.print_exc()
                print(f"Save failed: {e}")

        def handle_reset():
            return (
                '<div style="background:#e8edf3; padding:10px 14px; border-radius:5px; '
                'text-align:center; color:#8a94a0; font-size:15px; border:1px solid #d5dbe3;">'
                'Awaiting prediction...</div>',
                '<div style="display:flex; flex-wrap:wrap; justify-content:center; gap:18px; font-size:14px; '
                'color:#8a94a0; padding:5px 0; border:1px solid #d5dbe3; border-radius:5px; '
                'margin:2px 0; background:#fff;">'
                '<span><b>Raw Prob:</b> —</span>'
                '<span><b>Cal Prob:</b> —</span>'
                '</div>',
                '<div style="border:1px solid #d5dbe3; border-radius:6px; padding:10px; '
                'color:#8a94a0; font-size:14px; text-align:center; background:#fafbfc;">'
                'Health guidance will appear after prediction...</div>',
                None, None,
                '<div id="decision-box" style="border:1px solid #d5dbe3; border-radius:6px; '
                'padding:8px; color:#8a94a0; font-size:14px; text-align:center; '
                'min-height:231px; display:flex; align-items:center; justify-content:center; '
                'background:#fafbfc;">'
                'Decision explanation will appear here...</div>',
                None,
            )

        # ===== Manual Data Entry Helpers =====

        def _empty_table(feats):
            return pd.DataFrame({"Feature": feats, "Value": [None] * len(feats)})

        def _seed_table(state_dict, feats):
            return pd.DataFrame({
                "Feature": feats,
                "Value": [state_dict.get(f) for f in feats],
            })

        def _table_to_dict(df, feats):
            out = {}
            if df is None or len(df) == 0:
                return out
            for i, f in enumerate(feats):
                if i >= len(df):
                    break
                v = df.iloc[i].get("Value")
                if v is None:
                    continue
                try:
                    fv = float(v)
                except (TypeError, ValueError):
                    continue
                if pd.isna(fv):
                    continue
                out[f] = fv
            return out

        # Ordered list of (subgroup_key, features) matching the UI declaration order
        _SUBGROUP_SPECS = (
            [(sg['key'], sg['features'])
             for dim in FEATURE_DIMENSIONS for sg in dim['subgroups']]
            + [('other', EXTRA_MODEL_FEATURES)]
        )
        # Ordered list of subgroup keys — same order as tables declared in the UI
        _SUBGROUP_KEYS = [k for k, _ in _SUBGROUP_SPECS]

        def open_manual_dialog(state_dict, _specs=_SUBGROUP_SPECS):
            state_dict = state_dict or {}
            outs = [_seed_table(state_dict, feats) for _, feats in _specs]
            outs.append(gr.update(visible=True))
            return tuple(outs)

        def save_manual_dialog(state_dict, *tables, _specs=_SUBGROUP_SPECS):
            new_state = dict(state_dict or {})
            for (_, feats), df in zip(_specs, tables):
                new_state.update(_table_to_dict(df, feats))
            # When manual data is non-empty, auto-switch the model dropdown to
            # LightGBM — manual mode bypasses imputation and requires a model
            # that handles NaN natively.
            if new_state:
                lgb_item = _find_lightgbm_model()
                if lgb_item is not None:
                    auc = lgb_item.get('auc')
                    lgb_label = "Model: " + _model_display_label(lgb_item['model_name'], auc)
                    dropdown_update = gr.update(value=lgb_item['label'], label=lgb_label)
                else:
                    dropdown_update = gr.update()
            else:
                dropdown_update = gr.update()
            return new_state, gr.update(visible=False), dropdown_update

        def clear_manual_tables(_specs=_SUBGROUP_SPECS):
            return tuple(_empty_table(feats) for _, feats in _specs)

        def close_manual_dialog():
            return gr.update(visible=False)

        def reset_manual_state():
            return {}

        load_model_btn.click(
            fn=handle_load_model,
            inputs=[model_dropdown],
            outputs=[model_dropdown],
        )

        # Manual entry dialog: both buttons open it seeded from manual_state.
        # Tables list = 15 subgroups + 1 "other" table, matching UI declaration order.
        _all_tables = [subgroup_tables[k] for k in _SUBGROUP_KEYS
                       if k != 'other'] + [other_table]
        _open_outputs = _all_tables + [manual_dialog]
        manual_entry_btn.click(
            fn=open_manual_dialog, inputs=[manual_state], outputs=_open_outputs,
        )
        view_edit_btn.click(
            fn=open_manual_dialog, inputs=[manual_state], outputs=_open_outputs,
        )
        save_btn_dialog.click(
            fn=save_manual_dialog,
            inputs=[manual_state] + _all_tables,
            outputs=[manual_state, manual_dialog, model_dropdown],
        )
        cancel_btn.click(fn=close_manual_dialog, outputs=[manual_dialog])
        clear_btn.click(fn=clear_manual_tables, outputs=_all_tables)

        predict_btn.click(
            fn=handle_predict,
            inputs=[csv_upload, steps, patient_name, patient_id, manual_state],
            outputs=[risk_html, prob_display,
                     health_guidance,
                     shap_waterfall, shap_importance, decision_explanation,
                     result_state],
            # 关键：Gradio 会把 js() 的返回值当作后端输入。前端把输入参数
            # 展开成多个实参传入（(...args) 收集为数组），必须把前 5 个
            # inputs 原样返回，否则上传/手动录入数据会被清空。
            js="(...args) => { if (window.wizardGo) { window.wizardGo(2); } return args.slice(0, 5); }",
        )

        save_btn.click(
            fn=handle_save,
            inputs=[result_state],
            outputs=[],
        )

        reset_btn.click(
            fn=handle_reset,
            inputs=[],
            outputs=[risk_html, prob_display,
                     health_guidance,
                     shap_waterfall, shap_importance, decision_explanation,
                     result_state],
        )
        reset_btn.click(fn=reset_manual_state, outputs=[manual_state])

        return demo


if __name__ == "__main__":
    from auth_manager import auth_handler

    server_name = os.environ.get("SERVER_NAME", "0.0.0.0")
    server_port = int(os.environ.get("SERVER_PORT", "7860"))
    auth_enabled = os.environ.get("AUTH_ENABLED", "false").lower() == "true"
    share_enabled = os.environ.get("SHARE", "false").lower() == "true"
    print(f"Starting Lung Cancer AI Warning System...")
    print(f"  Server: {server_name}:{server_port}")
    print(f"  Auth: {'enabled' if auth_enabled else 'disabled'}")
    print(f"  Share: {'enabled' if share_enabled else 'disabled'}")

    # ===== 启动方式 =====
    # PWA 模式：pwa/ 存在且非 share 时，用 FastAPI 挂载 Gradio，并额外
    # 暴露 manifest / service worker / 图标三个静态路由，供手机“添加到
    # 主屏幕”和离线壳使用。静态路由直接挂在 FastAPI 上，不受登录保护
    # 影响（manifest/sw/图标对浏览器必须可匿名访问）。
    # 其余情况（桌面 exe、share 隧道、无 pwa/ 目录）完全保持原启动方式。
    pwa_mode = PWA_AVAILABLE and not share_enabled
    if pwa_mode:
        # 先告知 create_demo 注入 PWA 头部（manifest / SW 注册），
        # 再创建界面；桌面/source 直接运行时不设此变量，头部保持原样。
        os.environ["PWA_SERVE"] = "1"

    demo = create_demo()

    if pwa_mode:
        import uvicorn
        from fastapi import FastAPI
        from fastapi.responses import FileResponse, Response
        from fastapi.staticfiles import StaticFiles

        api = FastAPI(title="Lung Cancer AI Warning System")

        manifest_path = os.path.join(PWA_DIR, "manifest.json")
        sw_path = os.path.join(PWA_DIR, "sw.js")
        icons_dir = os.path.join(PWA_DIR, "icons")

        @api.get("/manifest.json")
        def _manifest():
            return FileResponse(
                manifest_path,
                media_type="application/manifest+json",
                headers={"Cache-Control": "no-cache"},
            )

        @api.get("/sw.js")
        def _sw():
            # sw.js 禁止缓存：保证浏览器能及时拿到新版本
            with open(sw_path, "rb") as f:
                body = f.read()
            return Response(
                content=body,
                media_type="application/javascript",
                headers={
                    "Cache-Control": "no-cache",
                    "Service-Worker-Allowed": "/",
                },
            )

        # /pwa/icons/... 静态资源（只暴露图标目录，不暴露 pwa/ 下的脚本）
        api.mount(
            "/pwa/icons",
            StaticFiles(directory=icons_dir),
            name="pwa-icons",
        )

        gr.mount_gradio_app(
            api,
            demo,
            path="/",
            auth=auth_handler if auth_enabled else None,
            auth_message="Please enter credentials",
            favicon_path=os.path.join(icons_dir, "favicon-32.png"),
        )
        print("  PWA mode: enabled (manifest / sw / icons served)")
        uvicorn.run(api, host=server_name, port=server_port, log_level="info")
    else:
        demo.launch(
            server_name=server_name,
            server_port=server_port,
            auth=auth_handler if auth_enabled else None,
            auth_message="Please enter credentials",
            share=share_enabled,
            favicon_path=os.path.join(PWA_DIR, "icons", "favicon-32.png")
            if PWA_AVAILABLE
            else None,
        )
