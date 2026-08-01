import os
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import streamlit as st

# 1. Page Configuration
st.set_page_config(
    page_title="LCA-Predict: Decision Support System",
    page_icon="🩺",
    layout="wide",
    initial_sidebar_state="expanded"
)

# 2. Modern, Clean Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 24px;
        font-weight: 700;
        color: #1A365D;
        padding-bottom: 8px;
        border-bottom: 3px solid #2B6CB0;
        margin-bottom: 15px;
    }
    .engine-card {
        background-color: #FFFFFF;
        border: 2px solid #E2E8F0;
        border-radius: 10px;
        padding: 16px;
        box-shadow: 0 2px 4px rgba(0,0,0,0.05);
        position: relative;
    }
    .engine-card-selected {
        background-color: #F7FAFC;
        border: 2.5px solid #2B6CB0;
        border-radius: 10px;
        padding: 16px;
        box-shadow: 0 4px 10px rgba(43,108,176,0.15);
        position: relative;
    }
    .selection-badge-active {
        background-color: #2B6CB0;
        color: #FFFFFF;
        font-size: 11px;
        font-weight: 700;
        padding: 3px 10px;
        border-radius: 12px;
        display: inline-block;
        margin-bottom: 8px;
        letter-spacing: 0.5px;
    }
    .selection-badge-inactive {
        background-color: #EDF2F7;
        color: #A0AEC0;
        font-size: 11px;
        font-weight: 600;
        padding: 3px 10px;
        border-radius: 12px;
        display: inline-block;
        margin-bottom: 8px;
    }
    .status-badge-high {
        background-color: #FED7D7;
        color: #9B2C2C;
        font-weight: 700;
        padding: 4px 12px;
        border-radius: 12px;
        font-size: 13px;
        display: inline-block;
    }
    .status-badge-low {
        background-color: #C6F6D5;
        color: #22543D;
        font-weight: 700;
        padding: 4px 12px;
        border-radius: 12px;
        font-size: 13px;
        display: inline-block;
    }
    .arbitration-banner {
        background-color: #EBF8FF;
        border-left: 5px solid #3182CE;
        border-radius: 6px;
        padding: 12px 16px;
        margin-top: 15px;
        margin-bottom: 15px;
    }
    .high-risk-card {
        background-color: #FFF5F5;
        border-left: 6px solid #E53E3E;
        padding: 18px;
        border-radius: 6px;
        margin-top: 10px;
        margin-bottom: 15px;
    }
    .low-risk-card {
        background-color: #F0FFF4;
        border-left: 6px solid #38A169;
        padding: 18px;
        border-radius: 6px;
        margin-top: 10px;
        margin-bottom: 15px;
    }
</style>
""", unsafe_allow_html=True)


# 3. Load Pre-trained Dual-Engine Assets
@st.cache_resource
def load_assets():
    asset_dir = 'model_assets'
    scaler = joblib.load(os.path.join(asset_dir, 'scaler.joblib'))
    model_lr = joblib.load(os.path.join(asset_dir, 'lr_model.joblib'))
    model_svm = joblib.load(os.path.join(asset_dir, 'svm_model.joblib'))

    preop_vars = [
        'gender', 'age', 'BMI', 'Aeterial_cl', 'LCA_dis', 'tumor_dia',
        'Ctvalue', 'diameter', 'tumor_dis', 'CEA', 'CA199', 'protein_lev',
        'T_stage', 'N_stage', 'M_stage'
    ]
    return scaler, model_lr, model_svm, preop_vars


try:
    scaler, model_lr, model_svm, preop_vars = load_assets()
except Exception as e:
    st.error(f"Failed to load model assets from 'model_assets/'. Please run export_models.py first. Error: {e}")
    st.stop()

# Title Header
st.markdown('<div class="main-header">🩺 LCA-Predict: Decision Support System for Left Colic Artery Preservation</div>', unsafe_allow_html=True)
st.caption("Individualized Risk Assessment for Left Colic Artery (LCA) Perfusion Dependency | Laparoscopic Rectal Cancer Surgery")

# Sidebar
st.sidebar.header("📋 Patient Input Parameters")

with st.sidebar.expander("🧬 1. Vascular Geometry Features", expanded=True):
    ctvalue_input = st.selectbox("LCA/IMA CT Ratio", ["≤ 0.52 (Low Risk)", "> 0.52 (High Risk [OR=3.91])"], index=0)
    ctvalue_val = 0 if "≤" in ctvalue_input else 1

    diameter_input = st.selectbox("LCA/IMA Diameter Ratio", ["≤ 0.63 (Slender)", "> 0.63 (Thick [OR=2.72])"], index=0)
    diameter_val = 0 if "≤" in diameter_input else 1

    arterial_input = st.selectbox("LCA Branch Typology", ["Shared / Co-origin", "Type-2 Trifurcation [OR=3.91]"], index=0)
    arterial_val = 0 if "Shared" in arterial_input else 1

    lca_dis_input = st.selectbox("LCA Origin Distance", ["≤ 3.5 cm (Proximal [OR=1.80])", "> 3.5 cm (Distal)"], index=0)
    lca_dis_val = 0 if "≤" in lca_dis_input else 1

with st.sidebar.expander("📊 2. Patient Demographics & Lab Data", expanded=False):
    gender_input = st.selectbox("Sex", ["Male", "Female"], index=0)
    gender_val = 0 if gender_input == "Male" else 1

    age_input = st.selectbox("Age Group", ["≤ 60 years", "> 60 years"], index=0)
    age_val = 0 if "≤" in age_input else 1

    bmi_input = st.selectbox("BMI", ["≤ 24 kg/m²", "> 24 kg/m²"], index=0)
    bmi_val = 0 if "≤" in bmi_input else 1

    protein_input = st.selectbox("Serum Albumin Level", ["< 40 g/L", "≥ 40 g/L"], index=1)
    protein_val = 0 if "<" in protein_input else 1

    cea_input = st.selectbox("Serum CEA Level", ["≤ 5 ng/mL", "> 5 ng/mL"], index=0)
    cea_val = 0 if "≤" in cea_input else 1

    ca199_input = st.selectbox("Serum CA199 Level", ["≤ 34 U/mL", "> 34 U/mL"], index=0)
    ca199_val = 0 if "≤" in ca199_input else 1

with st.sidebar.expander("🔬 3. Tumor & Pathology Features", expanded=False):
    tumor_dia_input = st.selectbox("Max Tumor Diameter", ["≤ 4 cm", "> 4 cm"], index=0)
    tumor_dia_val = 0 if "≤" in tumor_dia_input else 1

    tumor_dis_input = st.selectbox("Distance to Anal Verge", ["< 10 cm", "≥ 10 cm"], index=0)
    tumor_dis_val = 0 if "<" in tumor_dis_input else 1

    t_stage_input = st.selectbox("Pathological T Stage", ["T1 - T2", "T3 - T4"], index=1)
    t_stage_val = 0 if "T1" in t_stage_input else 1

    n_stage_input = st.selectbox("Lymph Node (N) Stage", ["N0 (Negative)", "N+ (Positive)"], index=0)
    n_stage_val = 0 if "N0" in n_stage_input else 1

    m_stage_input = st.selectbox("Distant Metastasis (M)", ["M0 (No)", "M1 (Yes)"], index=0)
    m_stage_val = 0 if "M0" in m_stage_input else 1

patient_data = {
    'gender': gender_val, 'age': age_val, 'BMI': bmi_val,
    'Aeterial_cl': arterial_val, 'LCA_dis': lca_dis_val, 'tumor_dia': tumor_dia_val,
    'Ctvalue': ctvalue_val, 'diameter': diameter_val, 'tumor_dis': tumor_dis_val,
    'CEA': cea_val, 'CA199': ca199_val, 'protein_lev': protein_val,
    'T_stage': t_stage_val, 'N_stage': n_stage_val, 'M_stage': m_stage_val
}

df_patient = pd.DataFrame([patient_data])[preop_vars]
patient_scaled = scaler.transform(df_patient)

# Dual-Engine Model Predictions
prob_lr = model_lr.predict_proba(patient_scaled)[0, 1]
prob_svm = model_svm.predict_proba(patient_scaled)[0, 1]

# Unified Probability Cutoff Threshold (30.21% Study Consensus)
CUTOFF_THRESH = 0.3021
lr_high = (prob_lr >= CUTOFF_THRESH)
svm_high = (prob_svm >= CUTOFF_THRESH)

# Decision Fusion Logic
final_high_risk = lr_high or svm_high

if lr_high and svm_high:
    selected_engine = "BOTH"
    driver_title = "🎯 CONSENSUS SELECTION: Both Dual-Engine Models Predict High Dependency (Prob ≥ 30.21%)"
    driver_desc = "Logistic Regression and SVM Safety Engine independently confirm High Perfusion Loss Risk (slopDA ≥ 30.21%)."
elif svm_high and not lr_high:
    selected_engine = "SVM"
    driver_title = "🛡️ SAFETY OVERRIDE SELECTION: Clinical Safety Engine (SVM) Triggered"
    driver_desc = "To prevent false negatives and ischemic complications, the high-recall SVM Safety Engine (Sensitivity 85.4%, NLR 0.21) flag triggered a High Dependency warning."
elif lr_high and not svm_high:
    selected_engine = "LR"
    driver_title = "🎯 THRESHOLD EXCEEDED SELECTION: Precision Calibrated Engine (LR) Triggered"
    driver_desc = "The Calibrated Risk Probabilities exceeded the 30.21% consensus cutoff, triggering a High Dependency warning."
else:
    selected_engine = "BOTH"
    driver_title = "🟢 CONSENSUS SELECTION: Both Dual-Engine Models Confirm Low LCA Dependency"
    driver_desc = "Both Logistic Regression and SVM predict low perfusion loss probability (< 30.21%), confirming LCA preservation is not mandatory."

# Streamlined 2-Tab Layout
tab1, tab2 = st.tabs([
    "🩺 Assessment & Surgical Recommendation",
    "📘 Reviewer & Presentation Guide"
])

# ==============================================================================
# TAB 1: Assessment & Surgical Strategy
# ==============================================================================
with tab1:
    st.subheader("1. Dual-Engine Model Outputs & Selected Decision Driver")

    col1, col2 = st.columns(2)

    with col1:
        is_lr_selected = (selected_engine == "LR" or selected_engine == "BOTH")
        card_class_lr = "engine-card-selected" if is_lr_selected else "engine-card"
        badge_class_lr = "selection-badge-active" if is_lr_selected else "selection-badge-inactive"
        badge_text_lr = "✓ SELECTED FOR FINAL DECISION" if is_lr_selected else "SECONDARY ENGINE"

        lr_status_badge = "status-badge-high" if lr_high else "status-badge-low"
        lr_status_text = "🔴 HIGH RISK (≥ 30.21%)" if lr_high else "🟢 LOW RISK (< 30.21%)"

        st.markdown(f"""
        <div class="{card_class_lr}">
            <div><span class="{badge_class_lr}">{badge_text_lr}</span></div>
            <span style="font-size:13px; color:#4A5568; font-weight:700;">Engine 1: Precision Calibrated Engine (Logistic Regression)</span>
            <div style="font-size:34px; font-weight:800; color:#2B6CB0; margin:6px 0;">{prob_lr * 100:.1f}%</div>
            <div style="margin-bottom:8px;"><span class="{lr_status_badge}">{lr_status_text}</span></div>
            <p style="font-size:11px; color:#718096; margin-bottom:0;">
                <b>Cutoff Threshold:</b> 30.21% | <b>Brier Error:</b> 0.174 (Best Calibration) | <b>AUC:</b> 0.809
            </p>
        </div>
        """, unsafe_allow_html=True)

    with col2:
        is_svm_selected = (selected_engine == "SVM" or selected_engine == "BOTH")
        card_class_svm = "engine-card-selected" if is_svm_selected else "engine-card"
        badge_class_svm = "selection-badge-active" if is_svm_selected else "selection-badge-inactive"
        badge_text_svm = "✓ SELECTED FOR FINAL DECISION" if is_svm_selected else "SECONDARY ENGINE"

        svm_status_badge = "status-badge-high" if svm_high else "status-badge-low"
        svm_status_text = "🔴 HIGH RISK (≥ 30.21%)" if svm_high else "🟢 LOW RISK (< 30.21%)"

        st.markdown(f"""
        <div class="{card_class_svm}">
            <div><span class="{badge_class_svm}">{badge_text_svm}</span></div>
            <span style="font-size:13px; color:#4A5568; font-weight:700;">Engine 2: Clinical Safety Engine (Support Vector Machine)</span>
            <div style="font-size:34px; font-weight:800; color:#2D3748; margin:6px 0;">{prob_svm * 100:.1f}%</div>
            <div style="margin-bottom:8px;"><span class="{svm_status_badge}">{svm_status_text}</span></div>
            <p style="font-size:11px; color:#718096; margin-bottom:0;">
                <b>Cutoff Threshold:</b> 30.21% | <b>Sensitivity:</b> 85.4% (Min. False Negatives) | <b>NLR:</b> 0.21
            </p>
        </div>
        """, unsafe_allow_html=True)

    # Arbitration Banner
    st.markdown(f"""
    <div class="arbitration-banner">
        <div style="font-size:14px; font-weight:700; color:#1A365D;">{driver_title}</div>
        <div style="font-size:12px; color:#2D3748; margin-top:4px;">{driver_desc}</div>
    </div>
    """, unsafe_allow_html=True)

    # Final Recommendation Banner
    if final_high_risk:
        st.markdown("""
        <div class="high-risk-card">
            <h3 style="color:#C53030; margin-top:0;">🔴 FINAL RECOMMENDATION: PRESERVE LEFT COLIC ARTERY (Low Ligation)</h3>
            <p style="font-size:15px; font-weight:bold; color:#9B2C2C;">
                RECOMMENDED OPERATIVE STRATEGY: Low Ligation of IMA + Station 253 Lymph Node Dissection
            </p>
            <hr style="border:0; border-top:1px solid #FEB2B2; margin:8px 0;">
            <b>Clinical Rationale & Operative Steps:</b>
            <ul>
                <li>LCA clamping is predicted to cause <b>> 30.21% drop</b> in rectal stump perfusion slope (Mean loss in high-dependency cohort: <b>47.5%</b>).</li>
                <li>Marginal arterial collateral network is insufficient. High ligation at the IMA root poses significant risk of anastomotic stump ischemia.</li>
                <li><b>Operative Execution:</b> Skeletonize the IMA root, dissect Station 253 lymph nodes, and divide IMA distal to the LCA origin to preserve LCA perfusion.</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)
    else:
        st.markdown("""
        <div class="low-risk-card">
            <h3 style="color:#276749; margin-top:0;">🟢 FINAL RECOMMENDATION: FLEXIBLE LIGATION (High or Low Ligation)</h3>
            <p style="font-size:15px; font-weight:bold; color:#22543D;">
                RECOMMENDED OPERATIVE STRATEGY: Ligation Mode Determined by Lymphadenectomy Needs
            </p>
            <hr style="border:0; border-top:1px solid #C6F6D5; margin:8px 0;">
            <b>Clinical Rationale & Operative Steps:</b>
            <ul>
                <li>LCA clamping causes minimal perfusion slope drop (Mean loss in low-dependency cohort: <b>14.2%</b>).</li>
                <li>Marginal collateral arc and SMA retrograde perfusion are physiologically robust. Ligation of LCA for radical lymphadenectomy is safe.</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)

    # Section 2: Patient Anatomical Risk Factor Profile
    st.subheader("2. Patient Anatomical Risk Factor Profile")
    st.caption("This module profiles individualized vascular geometry by mapping active risk factors against multivariable Odds Ratios (OR) from our clinical cohort.")

    risk_factors_present = []
    if ctvalue_val == 1:
        risk_factors_present.append(("LCA/IMA CT Ratio > 0.52", 3.91, "High CT attenuation reflects substantial blood flow volume in LCA."))
    if diameter_val == 1:
        risk_factors_present.append(("LCA/IMA Diameter Ratio > 0.63", 2.72, "Thick vessel diameter causes significant post-clamping drop."))
    if arterial_val == 1:
        risk_factors_present.append(("Type-2 Trifurcation Branching", 3.91, "Trifurcated LCA acts as a primary supply trunk."))
    if lca_dis_val == 0:
        risk_factors_present.append(("Proximal Origin ≤ 3.5 cm", 1.80, "Proximal branching dictates major trunk-level perfusion."))

    col_prof1, col_prof2 = st.columns([1.1, 1])

    with col_prof1:
        if len(risk_factors_present) > 0:
            st.warning(f"⚠️ {len(risk_factors_present)} Independent Anatomical Risk Factor(s) Active in This Patient:")
            for factor, or_val, desc in risk_factors_present:
                st.markdown(f"* **{factor}** (Multivariable **Adjusted OR = {or_val:.2f}**): _{desc}_")
        else:
            st.info("ℹ️ No dominant anatomical risk features detected. Vascular geometry reflects a typical low-dependency profile.")

    with col_prof2:
        fig_attr, ax_attr = plt.subplots(figsize=(6, 2.2), dpi=150)
        features_names = ['CT Ratio > 0.52', 'Diameter Ratio > 0.63', 'Type-2 Trifurcation', 'Proximal Origin ≤ 3.5cm', 'BMI > 24 kg/m²']
        presence = [ctvalue_val, diameter_val, arterial_val, 1 if lca_dis_val == 0 else 0, bmi_val]
        colors_bar = ['#E53E3E' if p == 1 else '#CBD5E0' for p in presence]

        bars = ax_attr.barh(features_names, presence, color=colors_bar, height=0.5)
        ax_attr.set_xlim(0, 1.3)
        ax_attr.set_xticks([0, 1])
        ax_attr.set_xticklabels(['Absent (0)', 'Present (1)'], fontsize=8)
        ax_attr.set_title("Anatomical Risk Factors Active Matching Profile", fontsize=9, fontweight='bold')

        for bar, p in zip(bars, presence):
            if p == 1:
                ax_attr.text(1.03, bar.get_y() + bar.get_height()/2, 'Active', va='center', fontsize=8, fontweight='bold', color='#C53030')

        sns.despine(ax=ax_attr)
        plt.tight_layout()
        st.pyplot(fig_attr)

# TAB 2
with tab2:
    st.subheader("📘 Presentation & Peer-Review Pitching Guide")

    col_a, col_b = st.columns(2)

    with col_a:
        st.markdown("""
        <div style="background:#EDF2F7; padding:15px; border-radius:6px;">
            <h4 style="color:#2B6CB0; margin-top:0;">👨‍⚕️ 1. For Clinical Surgical Reviewers</h4>
            <b>Key Concern:</b> "Is the AI safe? Will it misclassify high-risk cases?"<br><br>
            <b>Recommended Pitch:</b><br>
            "In intraoperative decision support, <b>minimizing false negatives is paramount</b>. Our app deploys a <b>Support Vector Machine (SVM) Safety Engine</b>, achieving the <b>highest sensitivity (85.4%)</b> and <b>lowest negative likelihood ratio (NLR = 0.21)</b> to offer maximum protection against stump ischemia."
        </div>
        """, unsafe_allow_html=True)

    with col_b:
        st.markdown("""
        <div style="background:#EDF2F7; padding:15px; border-radius:6px;">
            <h4 style="color:#2B6CB0; margin-top:0;">📊 2. For Statistical & AI Reviewers</h4>
            <b>Key Concern:</b> "Is the risk percentage calibrated? Has it overfitted?"<br><br>
            <b>Recommended Pitch:</b><br>
            "Our app integrates a <b>Logistic Regression Precision Engine</b> for risk scoring, yielding the <b>lowest calibration error (Brier Score = 0.174)</b> and a test-set AUC of <b>0.809</b>, adhering strictly to TRIPOD-AI guidelines."
        </div>
        """, unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)
    st.info("💡 **Dual-Engine Advantage**: Unifies Calibrated Risk Probability (LR) with Surgical Safety Warning (SVM) to satisfy both clinical safety and statistical rigor.")

# Footer
st.markdown("---")
st.caption("© 2026 LCA-Predict System | Designed for Laparoscopic Rectal Cancer Surgery Analysis")