# app.py
import matplotlib
matplotlib.use('Agg')
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import joblib

st.set_page_config(
    page_title="ICG-LCA Perfusion Risk Calculator",
    page_icon="🩺",
    layout="wide"
)

# 保持你精美的学术级 CSS 样式
st.markdown("""
    <style>
    .main-title { font-size:26px; font-weight:bold; color:#1e3d59; text-align:center; margin-bottom:5px; }
    .sub-title { font-size:14px; color:#e67e22; text-align:center; margin-bottom:25px; font-style:italic; }
    .section-header { font-size:18px; font-weight:bold; color:#1e3d59; border-bottom:2px solid #ecf0f1; padding-bottom:6px; margin-top:15px; }
    .advice-box { padding: 18px; border-radius: 8px; margin-top: 15px; font-size: 15px; font-weight: 500; line-height: 1.6; }
    .low-risk { background-color: rgba(46, 204, 113, 0.12); border-left: 6px solid #2ecc71; color: #1e7e34; }
    .med-risk { background-color: rgba(243, 156, 18, 0.12); border-left: 6px solid #e67e22; color: #b7791f; }
    .high-risk { background-color: rgba(231, 76, 60, 0.12); border-left: 6px solid #e74c3c; color: #bd2130; }
    .disclaimer-box { font-size: 11px; color: #7f8c8d; background-color: #f8f9fa; padding: 12px; border-radius: 5px; border: 1px solid #e2e8f0; margin-top: 20px; }
    </style>
""", unsafe_allow_html=True)

# 毫秒级安全载入已训练资产
@st.cache_resource
def load_frozen_assets():
    return joblib.load('model_assets.pkl')

assets = load_frozen_assets()
model_pipeline = assets['pipeline']
t_low = assets['t_low']
t_high = assets['t_high']
feat_importances = assets['importances']

st.markdown("<div class='main-title'>Web-Based Calculator for Predicting Rectal Perfusion Loss via ICG Fluorescence</div>", unsafe_allow_html=True)
st.markdown("<div class='sub-title'>Advanced Machine Learning Integration & Individualized Clinical Decision Support System (15-Variable RF Engine)</div>", unsafe_allow_html=True)

# ----------------- 左侧控制面板：收集15项指标 -----------------
st.sidebar.markdown("### 🩺 Patient Characteristics Input")
st.sidebar.markdown("#### 🔹 Vascular Morphology & CT Hemodynamics")
Ctvalue = 0 if st.sidebar.selectbox("1. LCA/IMA CT Value Ratio (Ctvalue):", ["≤ 0.52", "> 0.52"]) == "≤ 0.52" else 1
diameter = 0 if st.sidebar.selectbox("2. LCA/IMA Short Diameter Ratio (diameter):", ["≤ 0.63", "> 0.63"]) == "≤ 0.63" else 1
Aeterial_cl = 0 if "common trunk" in st.sidebar.selectbox("3. LCA Branching Type (Aeterial_cl):", ["Rectosigmoid common trunk type", "Three-branch independent type"]) else 1
LCA_dis = 0 if st.sidebar.selectbox("4. LCA Distance to IMA Origin (LCA_dis):", ["≤ 3.5 cm", "> 3.5 cm"]) == "≤ 3.5 cm" else 1

st.sidebar.markdown("#### 🔹 Tumor Morphology & Pathological Staging")
T_stage = 0 if "Early stage" in st.sidebar.selectbox("5. Tumor T Stage (T_stage):", ["Stage 1-2 (Early stage)", "Stage 3-4 (Locally advanced)"]) else 1
N_stage = 0 if "No nodal" in st.sidebar.selectbox("6. Lymph Node Metastasis (N_stage):", ["N0 (No nodal metastasis)", "N+ (Positive nodal metastasis)"]) else 1
M_stage = 0 if "No distant" in st.sidebar.selectbox("7. Distant Metastasis (M_stage):", ["M0 (No distant metastasis)", "M1 (Distant metastasis present)"]) else 1
tumor_dia = 0 if "≤" in st.sidebar.selectbox("8. Tumor Maximum Diameter (tumor_dia):", ["≤ 4 cm", "> 4 cm"]) else 1
tumor_dis = 0 if "< 10" in st.sidebar.selectbox("9. Tumor Distance to Anus (tumor_dis):", ["< 10 cm", "≥ 10 cm"]) else 1

st.sidebar.markdown("#### 🔹 Patient Baseline & Laboratory Tests")
gender = 0 if "Male" in st.sidebar.selectbox("10. Patient Sex (gender):", ["Male", "Female"]) else 1
age = 0 if "≤" in st.sidebar.selectbox("11. Patient Age Group (age):", ["≤ 60 years old (Middle-aged)", "> 60 years old (Elderly)"]) else 1
BMI = 0 if st.sidebar.selectbox("12. Patient BMI Level (BMI):", ["≤ 24", "> 24"]) == "≤ 24" else 1
CEA = 0 if st.sidebar.selectbox("13. Preoperative CEA Level (CEA):", ["≤ 5 ng/mL", "> 5 ng/mL"]) == "≤ 5 ng/mL" else 1
CA199 = 0 if st.sidebar.selectbox("14. Preoperative CA199 Level (CA199):", ["≤ 34 U/mL", "> 34 U/mL"]) == "≤ 34 U/mL" else 1
protein_lev = 0 if "< 40" in st.sidebar.selectbox("15. Serum Albumin Level (protein_lev):", ["< 40 g/L (Hypoalbuminemia / Malnutrition)", "≥ 40 g/L (Normal nutritional status)"]) else 1

# ----------------- 右侧实时预测与图表渲染 -----------------
input_data = pd.DataFrame([{
    'gender': gender, 'age': age, 'BMI': BMI, 'Aeterial_cl': Aeterial_cl, 'LCA_dis': LCA_dis, 'tumor_dia': tumor_dia,
    'Ctvalue': Ctvalue, 'diameter': diameter, 'tumor_dis': tumor_dis, 'CEA': CEA, 'CA199': CA199,
    'protein_lev': protein_lev, 'T_stage': T_stage, 'N_stage': N_stage, 'M_stage': M_stage
}])

risk_probability = model_pipeline.predict_proba(input_data)[0][1]

col1, col2 = st.columns([1, 1.2])
with col1:
    st.markdown("<div class='section-header'>📊 Full Model Quantification Result</div>", unsafe_allow_html=True)
    st.write("")
    st.metric(label="Predicted Probability of High Perfusion Loss (slopDA > 0.2933)", value=f"{risk_probability * 100:.2f} %")
    st.progress(float(risk_probability))

    st.write("")
    st.markdown("<p style='font-size:13px; font-weight:bold; color:#1e3d59; margin-bottom:2px;'>Global Feature Gini Importance Matrix</p>", unsafe_allow_html=True)
    fig, ax = plt.subplots(figsize=(5.2, 4.8))
    colors_sns = sns.color_palette("coolwarm", len(feat_importances))
    feat_importances.plot(kind='barh', color=colors_sns, ax=ax)
    ax.set_xlabel('Gini Importance Weight', fontsize=9, fontweight='bold')
    ax.tick_params(axis='both', labelsize=8.5)
    sns.despine()
    st.pyplot(fig)

with col2:
    st.markdown("<div class='section-header'>💡 Tailored Surgical Recommendation</div>", unsafe_allow_html=True)
    if risk_probability < t_low:
        st.markdown(f"<div class='advice-box low-risk'><b>Risk Stratification: LOW-RISK ZONE (&lt; {t_low * 100:.1f}%)</b><br><b>🎯 Recommended Strategy:</b> Perform standard <b>High Ligation of IMA</b>. Complete root lymphadenectomy of No.253 nodes to maximize oncologic radicality.</div>", unsafe_allow_html=True)
    elif risk_probability <= t_high:
        st.markdown(f"<div class='advice-box med-risk'><b>Risk Stratification: MEDIUM-RISK ZONE ({t_low * 100:.1f}% ~ {t_high * 100:.1f}%)</b><br><b>🎯 Recommended Strategy:</b> Mandatory <b>Intraoperative Trial Clamping Protocol</b>. Apply temporary vascular clamp on LCA trunk for 5 minutes and verify real-time perfusion.</div>", unsafe_allow_html=True)
    else:
        st.markdown(f"<div class='advice-box high-risk'><b>Risk Stratification: HIGH-RISK ZONE (&gt; {t_high * 100:.1f}%)</b><br><b>🎯 Recommended Strategy:</b> Mandatory <b>LCA-Preserving Low Ligation</b>. Skeletonize IMA for complete No.253 node dissection while fully preserving the intact LCA trunk.</div>", unsafe_allow_html=True)

st.write("---")
st.markdown(f"<p style='font-size:11px; color:#95a5a6; text-align:center;'>Technical Note: Powered by Random Forest Pipeline. Thresholds ({t_low:.4f} and {t_high:.4f}) are dynamically optimized for Sensitivity &ge; 90% and Specificity &ge; 90%.</p>", unsafe_allow_html=True)
st.markdown("<div class='disclaimer-box'><b>🔬 Clinical Research Disclaimer:</b> Interactive calculator for academic validation and translational research. Final intraoperative decision-making relies on comprehensive clinical judgment.</div>", unsafe_allow_html=True)