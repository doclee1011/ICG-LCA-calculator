# -*- coding: utf-8 -*-
"""
LCA Dependency Risk Predictor — Streamlit Web Application (Formal English Version)
Run: streamlit run app.py

Model: Logistic Regression Pipeline (with StandardScaler)
Core Features: Ctvalue, diameter, Aeterial_cl, LCA_dis, CEA (Multiverse LASSO intersection, 100% stability)
Risk Stratification: Tertile-based (Low < 0.344; Medium 0.344–0.679; High > 0.679)
Data Source: ICG fluorescence perfusion study in laparoscopic rectal cancer surgery (n = 376)

Corresponding manuscript: Preoperative prediction of left colic artery dependency
based on ICG fluorescence perfusion imaging in rectal cancer surgery (under review).
This tool is provided for research and educational purposes only.
"""
import os
import numpy as np
import pandas as pd
import joblib
import streamlit as st

# ==============================================================================
# Page Configuration
# ==============================================================================
st.set_page_config(
    page_title="LCA Dependency Risk Predictor",
    page_icon="🩺",
    layout="centered",
    initial_sidebar_state="expanded",
)

# ==============================================================================
# Load Model
# ==============================================================================
WORK_DIR = os.path.dirname(os.path.abspath(__file__))
model_path = os.path.join(WORK_DIR, 'best_prediction_model.pkl')
scaler_path = os.path.join(WORK_DIR, 'feature_scaler.pkl')

# Core features (Multiverse LASSO intersection, 100% stability)
CORE_VARS = ['Ctvalue', 'diameter', 'Aeterial_cl', 'LCA_dis', 'CEA']

# Risk stratification cutoffs (tertile-based; validated by sensitivity analyses across
# four cutoff-determination methods; consistent with the manuscript)
RISK_LOW_CUTOFF = 0.344
RISK_HIGH_CUTOFF = 0.679

# Clinical reference cutoff for LDR (high LCA dependency)
LDR_CLINICAL_CUTOFF = 0.30

try:
    best_model = joblib.load(model_path)
    MODEL_LOADED = True
    MODEL_TYPE = type(best_model).__name__
except Exception as e:
    MODEL_LOADED = False
    MODEL_TYPE = "Unknown"
    st.error(f"Model loading failed: {e}")
    st.info("Please run prediction_model_complete_analysis.py first to generate the model files.")

# ==============================================================================
# Variable Labels and Options
# ==============================================================================
VAR_INFO = {
    'Ctvalue': {
        'label': 'CT Attenuation Ratio (LCA/IMA)',
        'help': 'CT attenuation ratio of the left colic artery (LCA) to the inferior mesenteric artery (IMA). '
                'Risk factor: > 0.52 is associated with high LCA dependency.',
        'options': {0: '≤ 0.52 (low risk)', 1: '> 0.52 (high risk)'},
        'direction': 'Risk Factor'
    },
    'diameter': {
        'label': 'Short-axis Diameter Ratio (LCA/IMA)',
        'help': 'Short-axis diameter ratio of the LCA to the IMA (SA ratio). '
                'Risk factor: > 0.63 is associated with high LCA dependency.',
        'options': {0: '≤ 0.63 (low risk)', 1: '> 0.63 (high risk)'},
        'direction': 'Risk Factor'
    },
    'Aeterial_cl': {
        'label': 'LCA Origin Type',
        'help': 'Anatomical origin pattern of the LCA from the IMA. '
                'Type 2 (trifurcation) is a risk factor for high LCA dependency.',
        'options': {0: 'Type 1 (common trunk; low risk)', 1: 'Type 2 (trifurcation; high risk)'},
        'direction': 'Risk Factor'
    },
    'LCA_dis': {
        'label': 'LCA Origin Distance',
        'help': 'Distance of the LCA origin from the IMA origin. Protective factor: '
                'a shorter distance (≤ 3.5 cm) is associated with high LCA dependency.',
        'options': {0: '≤ 3.5 cm (high-risk value)', 1: '> 3.5 cm (low-risk value)'},
        'direction': 'Protective Factor'
    },
    'CEA': {
        'label': 'Serum CEA Level',
        'help': 'Serum carcinoembryonic antigen (CEA) level. Protective factor: '
                'a lower level (≤ 5 ng/mL) is associated with high LCA dependency.',
        'options': {0: '≤ 5 ng/mL (high-risk value)', 1: '> 5 ng/mL (low-risk value)'},
        'direction': 'Protective Factor'
    },
}

# ==============================================================================
# Title and Study Background
# ==============================================================================
st.title("🩺 LCA Dependency Risk Predictor")
st.markdown(
    "Preoperative prediction of left colic artery (LCA) dependency in laparoscopic "
    "rectal cancer surgery based on ICG fluorescence perfusion imaging"
)
st.markdown("---")

with st.expander("📖 Study Background & Methodology", expanded=False):
    st.markdown(
        """
**Background**: During laparoscopic rectal cancer surgery, indocyanine green (ICG)
fluorescence imaging assesses rectal wall perfusion before and after temporary clamping
of the left colic artery (LCA). The LCA dependency ratio (LDR) is defined as
LDR = (slope-pre − slope-post) / slope-pre, where higher values indicate greater perfusion
loss after LCA clamping, i.e., higher LCA dependency.

**Prediction Target**: Probability of LDR ≥ 0.30 (high LCA dependency, for which
LCA preservation is recommended).

**Core Features** (identified by multiverse LASSO intersection, 100% stability,
validated across four cutoff-determination methods):
- CT attenuation ratio (LCA/IMA) — OR = 3.80, P < 0.0001 (strongest predictor)
- Short-axis diameter ratio (LCA/IMA) — OR = 2.44, P = 0.0003
- LCA origin type (trifurcation) — OR = 2.88, P = 0.011
- LCA origin distance — OR = 0.58, P = 0.028 (protective)
- Serum CEA level — OR = 0.62, P = 0.058 (borderline, protective)

**Model Performance** (independent test set, n = 76):
- AUC = 0.793, Accuracy = 77.6%, Sensitivity = 82.9%, Specificity = 71.4%
- Brier score = 0.179; Hosmer–Lemeshow P = 0.720 (good calibration)

**Risk Stratification** (tertile-based; validated by sensitivity analyses across
four cutoff-determination methods):
- Low risk: probability < 0.344 (observed high-dependency rate 18.9%)
- Medium risk: probability 0.344–0.679 (observed rate 63.8%)
- High risk: probability > 0.679 (observed rate 80.3%)
"""
    )

if MODEL_LOADED:
    st.success(f"✅ Model loaded: {MODEL_TYPE} (Logistic Regression Pipeline)")
else:
    st.warning("⚠️ Model not loaded — interface demonstration only")

st.markdown("---")

# ==============================================================================
# Input Section
# ==============================================================================
st.subheader("📋 Patient Clinical Parameters")

input_values = {}
col1, col2 = st.columns(2)

for i, var in enumerate(CORE_VARS):
    info = VAR_INFO.get(var, {'label': var, 'help': '', 'options': {0: '0', 1: '1'}})
    options = info['options']
    display_options = {v: k for k, v in options.items()}

    with col1 if i % 2 == 0 else col2:
        direction_icon = "⚠️" if info.get('direction') == 'Risk Factor' else "🛡️"
        choice = st.selectbox(
            f"**{direction_icon} {info['label']}**",
            options=list(display_options.keys()),
            help=info.get('help', ''),
            key=f"input_{var}",
        )
        input_values[var] = display_options[choice]

st.markdown("---")

# ==============================================================================
# Predict Button
# ==============================================================================
col1, col2, col3 = st.columns([1, 2, 1])
with col2:
    predict_btn = st.button("🔍 Predict", type="primary", use_container_width=True)

if predict_btn:
    if MODEL_LOADED:
        # Build input vector (Pipeline handles the scaler automatically)
        X_input = pd.DataFrame([input_values])[CORE_VARS].values.astype(float)
        prob = best_model.predict_proba(X_input)[0, 1]
        pred_class = 1 if prob >= 0.5 else 0

        # Risk stratification
        if prob >= RISK_HIGH_CUTOFF:
            risk_level = "🔴 High Risk"
            risk_color = "red"
        elif prob >= RISK_LOW_CUTOFF:
            risk_level = "🟡 Medium Risk"
            risk_color = "orange"
        else:
            risk_level = "🟢 Low Risk"
            risk_color = "green"

        st.markdown("---")
        st.subheader("📊 Prediction Results")

        # Probability display
        col_a, col_b, col_c = st.columns(3)
        with col_a:
            st.metric("High LCA Dependency Probability", f"{prob * 100:.1f}%")
        with col_b:
            st.metric("Predicted Class",
                      f"High dependency (LDR ≥ {LDR_CLINICAL_CUTOFF:.0%})" if pred_class == 1
                      else f"Low dependency (LDR < {LDR_CLINICAL_CUTOFF:.0%})")
        with col_c:
            st.metric("Risk Level", risk_level)

        # Probability progress bar
        st.progress(float(prob))

        # Risk stratification explanation
        st.info(
            f"""
**Risk Stratification Criteria** (tertile-based; validated by sensitivity analyses across
four cutoff-determination methods):
- 🟢 Low risk: probability < {RISK_LOW_CUTOFF * 100:.1f}% (observed high-dependency rate 18.9%)
- 🟡 Medium risk: probability {RISK_LOW_CUTOFF * 100:.1f}%–{RISK_HIGH_CUTOFF * 100:.1f}% (observed rate 63.8%)
- 🔴 High risk: probability > {RISK_HIGH_CUTOFF * 100:.1f}% (observed rate 80.3%)
"""
        )

        # Clinical recommendations
        st.markdown("### 💡 Clinical Recommendations")
        if prob >= RISK_HIGH_CUTOFF:
            st.error(
                f"""
**High LCA Dependency (probability ≥ {RISK_HIGH_CUTOFF * 100:.0f}%)**: Rectal perfusion in this
patient is highly dependent on the left colic artery. **LCA preservation (low ligation) is
recommended** to secure adequate anastomotic blood supply and reduce the risk of anastomotic
leakage. Intraoperative ICG fluorescence angiography should still be used for real-time
confirmation.
"""
            )
        elif prob >= RISK_LOW_CUTOFF:
            st.warning(
                f"""
**Moderate LCA Dependency (probability {RISK_LOW_CUTOFF * 100:.0f}%–{RISK_HIGH_CUTOFF * 100:.0f}%)**:
This patient shows moderate dependency on the LCA. Intraoperative ICG fluorescence assessment
is recommended to guide the decision on LCA preservation. If high ligation is considered,
adequate marginal artery perfusion should be confirmed.
"""
            )
        else:
            st.success(
                f"""
**Low LCA Dependency (probability < {RISK_LOW_CUTOFF * 100:.0f}%)**: Rectal perfusion in this
patient shows low dependency on the LCA. The marginal artery generally provides adequate
collateral compensation, and high ligation (sacrificing the LCA) is relatively safe with
respect to blood supply. Intraoperative ICG fluorescence confirmation is still recommended.
"""
            )

        # Input echo
        with st.expander("📝 Input Parameters & Feature Directions"):
            df_display = pd.DataFrame(
                [
                    {
                        'Feature': VAR_INFO.get(v, {'label': v})['label'],
                        'Value': VAR_INFO.get(v, {'options': {0: '0', 1: '1'}})['options'].get(
                            input_values[v], str(input_values[v])),
                        'Direction': VAR_INFO.get(v, {'direction': 'Unknown'})['direction'],
                        'Encoded': input_values[v],
                    }
                    for v in CORE_VARS
                ]
            )
            st.dataframe(df_display, use_container_width=True, hide_index=True)

        # Nomogram scoring reference
        with st.expander("📐 Nomogram Scoring Reference (Logistic Regression)"):
            st.markdown(
                """
**Points assigned to the high-risk value of each feature** (max coefficient normalized to 100 points):
- CT attenuation ratio (LCA/IMA), > 0.52: 100.0 points
- Short-axis diameter ratio (LCA/IMA), > 0.63: 70.5 points
- LCA origin type (trifurcation): 55.7 points
- LCA origin distance (≤ 3.5 cm; protective): 35.8 points
- Serum CEA (≤ 5 ng/mL; protective): 32.7 points

**Total score range**: 0–294.7 points
**Usage**: sum the points across features to obtain the total score, which maps to the
predicted probability of high LCA dependency.
"""
            )

    else:
        st.error("Model not loaded — cannot perform prediction. Please run "
                 "prediction_model_complete_analysis.py first.")

# ==============================================================================
# Footer
# ==============================================================================
st.markdown("---")
st.caption(
    """
**Model Description**: This predictor is based on core features identified by multiverse
LASSO intersection and uses logistic regression (test-set AUC = 0.793), selected for its
interpretability, calibration, and clinical utility. Risk-stratification cutoffs were
validated by sensitivity analyses across four cutoff-determination methods (tertiles,
Youden's index, 80% sensitivity/specificity, and quartiles), supporting the robustness
of the conclusions.

**Disclaimer**: Predictions are provided for clinical reference and research purposes only.
They do not replace professional medical judgment or real-time intraoperative ICG
fluorescence assessment. Decisions regarding vascular ligation must always be made by the
operating surgeon based on the full clinical context.

**Data Source**: ICG fluorescence perfusion study in laparoscopic rectal cancer surgery
(n = 376; training set n = 300, test set n = 76)
**Corresponding Manuscript**: Preoperative prediction of left colic artery dependency based
on ICG fluorescence perfusion imaging in rectal cancer surgery (under review)
**Version**: 1.0 (October 2026)
"""
)
