import os
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import streamlit as st

# Set Page Configuration
st.set_page_config(
    page_title="LCA-Predict: Intraoperative Decision Support System",
    page_icon="🩺",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom Medical Theme Styling
st.markdown("""
<style>
    .main-header {
        font-size: 26px;
        font-weight: 700;
        color: #1A365D;
        padding-bottom: 10px;
        border-bottom: 3px solid #2B6CB0;
        margin-bottom: 20px;
    }
    .metric-card-container {
        background-color: #FFFFFF;
        border: 1px solid #E2E8F0;
        border-radius: 8px;
        padding: 16px;
        box-shadow: 0 1px 3px rgba(0,0,0,0.05);
    }
    .high-risk-card {
        background-color: #FFF5F5;
        border-left: 6px solid #E53E3E;
        padding: 18px;
        border-radius: 6px;
        margin-top: 15px;
        margin-bottom: 15px;
    }
    .low-risk-card {
        background-color: #F0FFF4;
        border-left: 6px solid #38A169;
        padding: 18px;
        border-radius: 6px;
        margin-top: 15px;
        margin-bottom: 15px;
    }
</style>
""", unsafe_allow_html=True)


# Load Pre-trained Model Assets (No raw data leakage)
@st.cache_resource
def load_assets():
    asset_dir = 'model_assets'
    scaler = joblib.load(os.path.join(asset_dir, 'scaler.joblib'))
    model_lr = joblib.load(os.path.join(asset_dir, 'lr_model.joblib'))
    model_svm = joblib.load(os.path.join(asset_dir, 'svm_model.joblib'))
    model_rf = joblib.load(os.path.join(asset_dir, 'rf_model.joblib'))

    preop_vars = [
        'gender', 'age', 'BMI', 'Aeterial_cl', 'LCA_dis', 'tumor_dia',
        'Ctvalue', 'diameter', 'tumor_dis', 'CEA', 'CA199', 'protein_lev',
        'T_stage', 'N_stage', 'M_stage'
    ]
    return scaler, model_lr, model_svm, model_rf, preop_vars


try:
    scaler, model_lr, model_svm, model_rf, preop_vars = load_assets()
except Exception as e:
    st.error(f"Failed to load model assets from 'model_assets/'. Please run export_models.py first. Error: {e}")
    st.stop()

# Title Header
st.markdown('<div class="main-header">🩺 LCA-Predict: Decision Support System for Left Colic Artery Preservation</div>',
            unsafe_allow_html=True)
st.caption(
    "Individualized Risk Assessment for LCA Perfusion Dependency (slopDA ≥ 30.21%) in Laparoscopic Rectal Cancer Surgery | Lancet/BJS Standard")

# Sidebar Input Panel
st.sidebar.header("📋 Clinical & Anatomical Parameters")

with st.sidebar.expander("🧬 1. Vascular & Imaging Geometry", expanded=True):
    ctvalue_input = st.selectbox(
        "LCA/IMA CT Attenuation Ratio",
        options=["≤ 0.52 (Low Risk)", "> 0.52 (High Risk [Adj. OR = 3.91])"],
        index=0
    )
    ctvalue_val = 0 if "≤" in ctvalue_input else 1

    diameter_input = st.selectbox(
        "LCA/IMA Diameter Ratio",
        options=["≤ 0.63 (Slender)", "> 0.63 (Thick [Adj. OR = 2.72])"],
        index=0
    )
    diameter_val = 0 if "≤" in diameter_input else 1

    arterial_input = st.selectbox(
        "LCA Branch Typology",
        options=["Trunk Co-origin / Sigmoid Shared", "Type-2 Trifurcation [Adj. OR = 3.91]"],
        index=0
    )
    arterial_val = 0 if "Co-origin" in arterial_input else 1

    lca_dis_input = st.selectbox(
        "LCA Origin Distance (from IMA root)",
        options=["≤ 3.5 cm (Proximal Origin [Adj. OR = 1.80])", "> 3.5 cm (Distal Origin)"],
        index=0
    )
    lca_dis_val = 0 if "≤" in lca_dis_input else 1

with st.sidebar.expander("📊 2. Patient Demographics & Lab Data", expanded=False):
    gender_input = st.selectbox("Sex", options=["Male", "Female"], index=0)
    gender_val = 0 if gender_input == "Male" else 1

    age_input = st.selectbox("Age Group", options=["≤ 60 years", "> 60 years"], index=0)
    age_val = 0 if "≤" in age_input else 1

    bmi_input = st.selectbox("Body Mass Index (BMI)", options=["≤ 24 kg/m²", "> 24 kg/m²"], index=0)
    bmi_val = 0 if "≤" in bmi_input else 1

    protein_input = st.selectbox("Serum Albumin Level", options=["< 40 g/L", "≥ 40 g/L"], index=1)
    protein_val = 0 if "<" in protein_input else 1

    cea_input = st.selectbox("Serum CEA Level", options=["≤ 5 ng/mL", "> 5 ng/mL"], index=0)
    cea_val = 0 if "≤" in cea_input else 1

    ca199_input = st.selectbox("Serum CA199 Level", options=["≤ 34 U/mL", "> 34 U/mL"], index=0)
    ca199_val = 0 if "≤" in ca199_input else 1

with st.sidebar.expander("🔬 3. Tumor & Pathology Features", expanded=False):
    tumor_dia_input = st.selectbox("Max Tumor Diameter", options=["≤ 4 cm", "> 4 cm"], index=0)
    tumor_dia_val = 0 if "≤" in tumor_dia_input else 1

    tumor_dis_input = st.selectbox("Distance to Anal Verge", options=["< 10 cm", "≥ 10 cm"], index=0)
    tumor_dis_val = 0 if "<" in tumor_dis_input else 1

    t_stage_input = st.selectbox("Pathological T Stage", options=["T1 - T2", "T3 - T4"], index=1)
    t_stage_val = 0 if "T1" in t_stage_input else 1

    n_stage_input = st.selectbox("Lymph Node (N) Stage", options=["N0 (Negative)", "N+ (Positive)"], index=0)
    n_stage_val = 0 if "N0" in n_stage_input else 1

    m_stage_input = st.selectbox("Distant Metastasis (M)", options=["M0 (No Metastasis)", "M1 (Metastatic)"], index=0)
    m_stage_val = 0 if "M0" in m_stage_input else 1

# Input Vector Assembly
patient_data = {
    'gender': gender_val, 'age': age_val, 'BMI': bmi_val,
    'Aeterial_cl': arterial_val, 'LCA_dis': lca_dis_val, 'tumor_dia': tumor_dia_val,
    'Ctvalue': ctvalue_val, 'diameter': diameter_val, 'tumor_dis': tumor_dis_val,
    'CEA': cea_val, 'CA199': ca199_val, 'protein_lev': protein_val,
    'T_stage': t_stage_val, 'N_stage': n_stage_val, 'M_stage': m_stage_val
}

df_patient = pd.DataFrame([patient_data])[preop_vars]
patient_scaled = scaler.transform(df_patient)

# Predictions
prob_lr = model_lr.predict_proba(patient_scaled)[0, 1]
prob_svm = model_svm.predict_proba(patient_scaled)[0, 1]
prob_rf = model_rf.predict_proba(patient_scaled)[0, 1]

# App Navigation Tabs
tab1, tab2, tab3 = st.tabs([
    "🩺 Assessment & Surgical Strategy",
    "📊 Evidence & Benchmarks",
    "📘 Presentation Guide"
])

# TAB 1
with tab1:
    st.subheader("1. Dual-Engine Risk Assessment")

    col1, col2, col3 = st.columns(3)

    with col1:
        st.markdown(f"""
        <div class="metric-card-container">
            <span style="font-size:12px; color:#718096; font-weight:bold;">Precision Calibrated Engine (Logistic Regression)</span>
            <div style="font-size:28px; font-weight:800; color:#2B6CB0; margin:8px 0;">{prob_lr * 100:.1f}%</div>
            <span style="font-size:11px; color:#4A5568;">Brier Score: 0.174 (Best Calibration) | AUC: 0.809</span>
        </div>
        """, unsafe_allow_html=True)

    with col2:
        svm_status_color = "#E53E3E" if prob_svm >= 0.50 else "#38A169"
        svm_status_text = "High Reliance (Preserve LCA)" if prob_svm >= 0.50 else "Low Reliance (Safe)"
        st.markdown(f"""
        <div class="metric-card-container">
            <span style="font-size:12px; color:#718096; font-weight:bold;">Clinical Safety Engine (Support Vector Machine)</span>
            <div style="font-size:24px; font-weight:800; color:{svm_status_color}; margin:8px 0;">{svm_status_text}</div>
            <span style="font-size:11px; color:#4A5568;">Sensitivity: 85.4% (Min. Missed Cases) | NLR: 0.21</span>
        </div>
        """, unsafe_allow_html=True)

    with col3:
        st.markdown(f"""
        <div class="metric-card-container">
            <span style="font-size:12px; color:#718096; font-weight:bold;">Generalization Engine (Random Forest)</span>
            <div style="font-size:28px; font-weight:800; color:#2D3748; margin:8px 0;">{prob_rf * 100:.1f}%</div>
            <span style="font-size:11px; color:#4A5568;">Internal CV AUC: 0.799 | Zero Overfitting</span>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    is_high_risk = (prob_lr >= 0.3021) or (prob_svm >= 0.50)

    if is_high_risk:
        st.markdown("""
        <div class="high-risk-card">
            <h3 style="color:#C53030; margin-top:0;">🔴 HIGH LCA DEPENDENCY (slopDA ≥ 30.21%)</h3>
            <p style="font-size:15px; font-weight:bold; color:#9B2C2C;">
                RECOMMENDED SURGICAL STRATEGY: Preserve Left Colic Artery (Low Ligation + Station 253 LND)
            </p>
            <hr style="border:0; border-top:1px solid #FEB2B2; margin:10px 0;">
            <b>Clinical & Anatomical Rationale:</b>
            <ul>
                <li>Transient clamping of LCA is predicted to cause <b>> 30.21% attenuation</b> in rectal stump perfusion slope (High-dependency mean loss: <b>47.5%</b>).</li>
                <li>Marginal arterial collateral network (Arc of Drummond / Riolan) is insufficient to compensate. Ligation at IMA root (High Ligation) risks severe stump ischemia.</li>
                <li><b>Surgical Execution:</b> Skeletonize IMA root, dissect Station 253 lymph nodes, and divide IMA distal to LCA origin to preserve LCA perfusion.</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)
    else:
        st.markdown("""
        <div class="low-risk-card">
            <h3 style="color:#276749; margin-top:0;">🟢 LOW LCA DEPENDENCY (slopDA < 30.21%)</h3>
            <p style="font-size:15px; font-weight:bold; color:#22543D;">
                RECOMMENDED SURGICAL STRATEGY: Flexible Ligation (High or Low Ligation per LND requirements)
            </p>
            <hr style="border:0; border-top:1px solid #C6F6D5; margin:10px 0;">
            <b>Clinical & Anatomical Rationale:</b>
            <ul>
                <li>Transient LCA clamping causes minimal loss in perfusion slope (Low-dependency mean loss: <b>14.2%</b>).</li>
                <li>Marginal arterial network and SMA retrograde collateral supply are well-established. Ligation of LCA for radical LND is physiologically safe.</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)

    st.subheader("2. Patient Risk Factor Profile")

    risk_factors_present = []
    if ctvalue_val == 1:
        risk_factors_present.append(
            ("LCA/IMA CT Ratio > 0.52", 3.91, "High CT attenuation reflects substantial blood flow volume in LCA."))
    if diameter_val == 1:
        risk_factors_present.append(
            ("LCA/IMA Diameter Ratio > 0.63", 2.72, "Thick vessel diameter leads to significant post-clamping drop."))
    if arterial_val == 1:
        risk_factors_present.append(
            ("Type-2 Trifurcation Branching", 3.91, "Trifurcated LCA acts as a primary supply trunk."))
    if lca_dis_val == 0:
        risk_factors_present.append(("Proximal Origin ≤ 3.5 cm from IMA Root", 1.80,
                                     "Proximal branching dictates major trunk-level perfusion."))

    if len(risk_factors_present) > 0:
        st.warning(f"⚠️ {len(risk_factors_present)} Independent Anatomical Risk Factor(s) Detected:")
        for factor, or_val, desc in risk_factors_present:
            st.markdown(f"* **{factor}** (Adjusted **OR = {or_val:.2f}**): _{desc}_")
    else:
        st.info(
            "ℹ️ No dominant anatomical risk features detected. Vascular geometry reflects a typical low-dependency profile.")

    # Attribution plot
    fig_attr, ax_attr = plt.subplots(figsize=(8, 3.2), dpi=200)
    features_names = ['CT Ratio > 0.52', 'Diameter Ratio > 0.63', 'Type-2 Trifurcation', 'Proximal Origin ≤ 3.5cm',
                      'BMI > 24 kg/m²']
    presence = [ctvalue_val, diameter_val, arterial_val, 1 if lca_dis_val == 0 else 0, bmi_val]
    colors_bar = ['#E53E3E' if p == 1 else '#CBD5E0' for p in presence]

    ax_attr.barh(features_names, presence, color=colors_bar, height=0.55)
    ax_attr.set_xlim(0, 1.2)
    ax_attr.set_xticks([0, 1])
    ax_attr.set_xticklabels(['Absent', 'Present'])
    ax_attr.set_title("Anatomical Risk Factors Matching Profile", fontsize=11, fontweight='bold')
    sns.despine(ax=ax_attr)
    plt.tight_layout()
    st.pyplot(fig_attr)

# TAB 2
with tab2:
    st.subheader("1. Independent Holdout Test Set Performance (Table 3, N=76)")
    table3_data = [
        {"Model": "AdaBoost", "AUC (95% CI)": "0.811 (0.710-0.913)", "Accuracy": "75.0%", "Sensitivity": "75.6%",
         "Specificity": "74.3%", "PPV": "77.5%", "NPV": "72.2%", "F1-Score": "0.765", "Brier Score": "0.246"},
        {"Model": "Logistic Regression", "AUC (95% CI)": "0.809 (0.707-0.910)", "Accuracy": "73.7%",
         "Sensitivity": "75.6%", "Specificity": "71.4%", "PPV": "75.6%", "NPV": "71.4%", "F1-Score": "0.756",
         "Brier Score": "0.174 (Best)"},
        {"Model": "Support Vector Machine", "AUC (95% CI)": "0.798 (0.695-0.901)", "Accuracy": "77.6% (Top)",
         "Sensitivity": "85.4% (Top)", "Specificity": "68.6%", "PPV": "76.1%", "NPV": "80.0% (Top)",
         "F1-Score": "0.805 (Top)", "Brier Score": "0.180"},
        {"Model": "Random Forest", "AUC (95% CI)": "0.797 (0.696-0.898)", "Accuracy": "73.7%", "Sensitivity": "78.0%",
         "Specificity": "68.6%", "PPV": "74.4%", "NPV": "72.7%", "F1-Score": "0.762", "Brier Score": "0.183"},
        {"Model": "Gradient Boosting", "AUC (95% CI)": "0.754 (0.644-0.865)", "Accuracy": "69.7%",
         "Sensitivity": "73.2%", "Specificity": "65.7%", "PPV": "71.4%", "NPV": "67.6%", "F1-Score": "0.723",
         "Brier Score": "0.204"},
        {"Model": "Neural Network (MLP)", "AUC (95% CI)": "0.741 (0.627-0.855)", "Accuracy": "72.4%",
         "Sensitivity": "73.2%", "Specificity": "71.4%", "PPV": "75.0%", "NPV": "69.4%", "F1-Score": "0.741",
         "Brier Score": "0.268"}
    ]
    st.dataframe(pd.DataFrame(table3_data), use_container_width=True)

    st.subheader("2. Pairwise DeLong Test P-Value Matrix (Table 4)")
    delong_data = {
        "Model": ["Random Forest", "Gradient Boosting", "SVM", "Logistic Regression", "AdaBoost", "MLP"],
        "Random Forest": ["-", "0.0002", "0.9593", "0.4804", "0.4348", "0.0005"],
        "Gradient Boosting": ["0.0002", "-", "0.0102", "0.0021", "0.0025", "0.4018"],
        "SVM": ["0.9593", "0.0102", "-", "0.4772", "0.4108", "0.0003"],
        "Logistic Regression": ["0.4804", "0.0021", "0.4772", "-", "0.6399", "0.0002"],
        "AdaBoost": ["0.4348", "0.0025", "0.4108", "0.6399", "-", "0.0003"],
        "Neural Network (MLP)": ["0.0005", "0.4018", "0.0003", "0.0002", "0.0003", "-"]
    }
    st.dataframe(pd.DataFrame(delong_data), use_container_width=True)
    st.caption(
        "Note: DeLong test P > 0.05 indicates no statistically significant difference in AUC. LR, SVM, RF, and AdaBoost show equivalent performance (P > 0.40).")

    st.subheader("3. Unsupervised Consensus Threshold Discovery (30.21%)")
    st.markdown("""
    To eliminate outcome bias, three unsupervised mathematical paradigms were employed:
    * **Otsu Thresholding**: Calculated Cutoff = **0.3074**
    * **Gaussian Mixture Model (GMM)**: PDF Intersection Cutoff = **0.2940**
    * **K-Means Clustering**: Center Midpoint Cutoff = **0.3047**
    * **Consensus Cutoff**: Arithmetic Mean = **30.21% (0.3021)**, effectively segregating Low Dependency (Mean Loss: 14.2%) from High Dependency (Mean Loss: 47.5%).
    """)

# TAB 3
with tab3:
    st.subheader("📘 Presentation & Peer-Review Pitching Strategy")

    col_a, col_b = st.columns(2)

    with col_a:
        st.markdown("""
        <div style="background:#EDF2F7; padding:15px; border-radius:6px;">
            <h4 style="color:#2B6CB0; margin-top:0;">👨‍⚕️ 1. Addressing Clinical Surgical Reviewers</h4>
            <b>Key Concern:</b> "Is the AI safe? Will it misclassify high-risk cases and cause stump ischemia?"<br>
            <b>Recommended Pitch:</b><br>
            "In intraoperative guidance, <b>clinical safety (avoiding false negatives) is paramount</b>. Our app deploys a <b>Support Vector Machine (SVM) Safety Engine</b>, which achieved the <b>highest sensitivity (85.4%)</b> and <b>lowest negative likelihood ratio (NLR = 0.21)</b> in the independent test set, offering maximum rule-out security for surgical decision-making."
        </div>
        """, unsafe_allow_html=True)

    with col_b:
        st.markdown("""
        <div style="background:#EDF2F7; padding:15px; border-radius:6px;">
            <h4 style="color:#2B6CB0; margin-top:0;">📊 2. Addressing Statistical & AI Reviewers</h4>
            <b>Key Concern:</b> "Is the predicted risk percentage accurate? Has the model overfitted?"<br>
            <b>Recommended Pitch:</b><br>
            "Our app integrates a <b>Logistic Regression Precision Engine</b> for risk scoring, which yielded the <b>lowest calibration error (Brier Score = 0.174)</b> and a test-set AUC of <b>0.809</b>. The predicted probabilities tightly match observed risk without overfitting, strictly adhering to TRIPOD-AI guidelines."
        </div>
        """, unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)
    st.info(
        "💡 **Dual-Engine Advantage**: By unifying Calibrated Probability (Logistic Regression) with Surgical Safety Warning (SVM), the app balances medical safety with statistical precision.")

# Footer
st.markdown("---")
st.caption("© 2026 LCA-Predict System | Designed for Laparoscopic Rectal Cancer Surgery Analysis")