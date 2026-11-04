# -*- coding: utf-8 -*-
"""
LCA Dependency Risk Predictor — Streamlit Web Application (Formal English Version)
Run: streamlit run app.py

Model: Logistic Regression Pipeline (with StandardScaler)
Core Features: Ctvalue, diameter, Aeterial_cl, LCA_dis, CEA (Multiverse LASSO intersection, 100% stability)
Risk Stratification: Tertile-based (Low < 0.344; Medium 0.344–0.679; High > 0.679)
Data Source: ICG fluorescence perfusion study in laparoscopic rectal cancer surgery (n = 376)

Input design (revised per corresponding author): clinicians enter the directly measured
values from preoperative CTA (LCA and IMA CT attenuation in HU, LCA and IMA vessel
diameter in mm, LCA origin distance in cm, and serum CEA in ng/mL); the application
computes the ratios and applies the training-set cutoffs internally.
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

# Training-set cutoffs used to dichotomize the continuous measurements
CT_RATIO_CUTOFF = 0.52      # Ctvalue = 1 if LCA/IMA CT-attenuation ratio > 0.52
DIAM_RATIO_CUTOFF = 0.63    # diameter = 1 if LCA/IMA diameter ratio > 0.63
DISTANCE_CUTOFF = 3.5       # LCA_dis = 1 if origin distance > 3.5 cm (protective)
CEA_CUTOFF = 5.0            # CEA = 1 if serum CEA > 5 ng/mL (protective)

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
# Page Title and Study Background
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
- LCA / IMA CT attenuation ratio — OR = 3.80, P < 0.0001 (strongest predictor)
- LCA / IMA vessel diameter ratio — OR = 2.44, P = 0.0003
- LCA origin type (trifurcation vs. common trunk) — OR = 2.88, P = 0.011
- LCA origin distance from IMA root — OR = 0.58, P = 0.028 (protective)
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
# Input Section — directly measured CTA / laboratory values; ratios computed internally
# ==============================================================================
st.subheader("📋 Patient Clinical Parameters")
st.caption(
    "Enter the values as measured on preoperative arterial-phase CTA and preoperative "
    "fasting blood tests. The application computes the LCA/IMA ratios internally."
)

# --- 1. CT attenuation (HU) ---------------------------------------------------
st.markdown("**1. CT attenuation on arterial-phase CTA (Hounsfield units, HU)**")
col_ct1, col_ct2 = st.columns(2)
with col_ct1:
    lca_ct = st.number_input(
        "LCA CT attenuation (HU)", min_value=0.0, max_value=1500.0,
        value=200.0, step=1.0,
        help="Mean CT attenuation within a circular ROI placed in the proximal LCA lumen.",
    )
with col_ct2:
    ima_ct = st.number_input(
        "IMA CT attenuation (HU)", min_value=0.0, max_value=1500.0,
        value=350.0, step=1.0,
        help="Mean CT attenuation within a circular ROI placed in the proximal IMA lumen.",
    )
ct_ratio = lca_ct / ima_ct if ima_ct > 0 else 0.0
st.write(f"Computed LCA/IMA CT attenuation ratio: **{ct_ratio:.2f}** "
         f"(training-set cutoff: {CT_RATIO_CUTOFF:.2f})")

# --- 2. Vessel diameter (mm) -------------------------------------------------
st.markdown("**2. Vessel diameter on CTA (mm, measured perpendicular to vessel centerline)**")
col_d1, col_d2 = st.columns(2)
with col_d1:
    lca_diam = st.number_input(
        "LCA diameter (mm)", min_value=0.0, max_value=20.0,
        value=3.0, step=0.1,
        help="Maximum diameter of the LCA measured on a plane perpendicular to its centerline.",
    )
with col_d2:
    ima_diam = st.number_input(
        "IMA diameter (mm)", min_value=0.0, max_value=30.0,
        value=6.0, step=0.1,
        help="Maximum diameter of the IMA measured on a plane perpendicular to its centerline.",
    )
diam_ratio = lca_diam / ima_diam if ima_diam > 0 else 0.0
st.write(f"Computed LCA/IMA diameter ratio: **{diam_ratio:.2f}** "
         f"(training-set cutoff: {DIAM_RATIO_CUTOFF:.2f})")

# --- 3. LCA origin type ------------------------------------------------------
st.markdown("**3. LCA origin anatomy**")
lca_type_label = st.radio(
    "LCA origin type",
    options=["Type 1 — common trunk of LCA and sigmoid artery (low risk)",
             "Type 2 — LCA arising directly from IMA as a trifurcation (high risk)"],
    help="Type 1: LCA shares a common trunk with the sigmoid artery; "
         "Type 2: LCA arises directly from the IMA without a common trunk.",
)
aeterial_cl = 1 if lca_type_label.startswith("Type 2") else 0

# --- 4. LCA origin distance --------------------------------------------------
st.markdown("**4. LCA origin distance**")
lca_dis = st.number_input(
    "Distance from IMA root to LCA origin (cm)",
    min_value=0.0, max_value=15.0, value=3.0, step=0.1,
    help="Curved length along the IMA centerline from the aortic origin to the LCA takeoff. "
         "Measured intraoperatively with a vessel loop and ruler (see Supplementary Figure).",
)

# --- 5. Serum CEA ------------------------------------------------------------
st.markdown("**5. Preoperative serum tumor marker**")
cea = st.number_input(
    "Serum CEA (ng/mL)", min_value=0.0, max_value=500.0,
    value=3.0, step=0.1,
    help="Carcinoembryonic antigen from fasting venous blood within 1 week before surgery.",
)

st.markdown("---")

# ==============================================================================
# Dichotomize inputs to match training-set feature encoding
# ==============================================================================
Ctvalue = 1 if ct_ratio > CT_RATIO_CUTOFF else 0
diameter = 1 if diam_ratio > DIAM_RATIO_CUTOFF else 0
LCA_dis_code = 1 if lca_dis > DISTANCE_CUTOFF else 0   # >3.5 cm = 1 = low risk
CEA_code = 1 if cea > CEA_CUTOFF else 0               # >5 ng/mL = 1 = low risk

input_values = {
    'Ctvalue': Ctvalue,
    'diameter': diameter,
    'Aeterial_cl': aeterial_cl,
    'LCA_dis': LCA_dis_code,
    'CEA': CEA_code,
}

# ==============================================================================
# Predict Button
# ==============================================================================
col1, col2, col3 = st.columns([1, 2, 1])
with col2:
    predict_btn = st.button("🔍 Predict", type="primary", use_container_width=True)

if predict_btn:
    if MODEL_LOADED:
        X_input = pd.DataFrame([input_values])[CORE_VARS].values.astype(float)
        prob = best_model.predict_proba(X_input)[0, 1]
        pred_class = 1 if prob >= 0.5 else 0

        if prob >= RISK_HIGH_CUTOFF:
            risk_level = "🔴 High Risk"
        elif prob >= RISK_LOW_CUTOFF:
            risk_level = "🟡 Medium Risk"
        else:
            risk_level = "🟢 Low Risk"

        st.markdown("---")
        st.subheader("📊 Prediction Results")

        col_a, col_b, col_c = st.columns(3)
        with col_a:
            st.metric("High LCA Dependency Probability", f"{prob * 100:.1f}%")
        with col_b:
            st.metric("Predicted Class",
                      f"High dependency (LDR ≥ {LDR_CLINICAL_CUTOFF:.0%})" if pred_class == 1
                      else f"Low dependency (LDR < {LDR_CLINICAL_CUTOFF:.0%})")
        with col_c:
            st.metric("Risk Level", risk_level)

        st.progress(float(prob))

        st.info(
            f"""
**Risk Stratification Criteria** (tertile-based; validated by sensitivity analyses across
four cutoff-determination methods):
- 🟢 Low risk: probability < {RISK_LOW_CUTOFF * 100:.1f}% (observed high-dependency rate 18.9%)
- 🟡 Medium risk: probability {RISK_LOW_CUTOFF * 100:.1f}%–{RISK_HIGH_CUTOFF * 100:.1f}% (observed rate 63.8%)
- 🔴 High risk: probability > {RISK_HIGH_CUTOFF * 100:.1f}% (observed rate 80.3%)
"""
        )

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

        with st.expander("📝 Entered Values and Encoded Features"):
            df_display = pd.DataFrame(
                [
                    {"Feature": "LCA/IMA CT attenuation ratio",
                     "Entered values": f"LCA {lca_ct:.0f} HU / IMA {ima_ct:.0f} HU",
                     "Computed ratio": f"{ct_ratio:.2f}",
                     "Encoded": Ctvalue,
                     "Direction": "Risk factor" if Ctvalue == 1 else "Low risk"},
                    {"Feature": "LCA/IMA diameter ratio",
                     "Entered values": f"LCA {lca_diam:.1f} mm / IMA {ima_diam:.1f} mm",
                     "Computed ratio": f"{diam_ratio:.2f}",
                     "Encoded": diameter,
                     "Direction": "Risk factor" if diameter == 1 else "Low risk"},
                    {"Feature": "LCA origin type",
                     "Entered values": lca_type_label.split(" — ")[0],
                     "Computed ratio": "—",
                     "Encoded": aeterial_cl,
                     "Direction": "Risk factor" if aeterial_cl == 1 else "Low risk"},
                    {"Feature": "LCA origin distance",
                     "Entered values": f"{lca_dis:.1f} cm",
                     "Computed ratio": "—",
                     "Encoded": LCA_dis_code,
                     "Direction": "Protective" if LCA_dis_code == 1 else "Risk factor"},
                    {"Feature": "Serum CEA",
                     "Entered values": f"{cea:.1f} ng/mL",
                     "Computed ratio": "—",
                     "Encoded": CEA_code,
                     "Direction": "Protective" if CEA_code == 1 else "Risk factor"},
                ]
            )
            st.dataframe(df_display, use_container_width=True, hide_index=True)

        with st.expander("📐 Nomogram Scoring Reference (Logistic Regression)"):
            st.markdown(
                """
**Points assigned to the high-risk value of each feature** (max coefficient normalized to 100 points):
- LCA/IMA CT attenuation ratio > 0.52: 100.0 points
- LCA/IMA diameter ratio > 0.63: 70.5 points
- LCA origin type (trifurcation): 55.7 points
- LCA origin distance ≤ 3.5 cm (protective direction): 35.8 points
- Serum CEA ≤ 5 ng/mL (protective direction): 32.7 points

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
**Version**: 1.1 (November 2026) — direct CTA value input
"""
)
