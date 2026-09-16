# -*- coding: utf-8 -*-
"""
LCA 灌注损失风险预测 — Streamlit 在线计算器（最终版）
运行: streamlit run app.py

模型：Logistic Regression Pipeline（含StandardScaler）
核心特征：Ctvalue, diameter, Aeterial_cl, LCA_dis, CEA（多宇宙LASSO交集，100%稳定性）
风险分层：三分位数法（低<0.343, 中0.343-0.679, 高>0.679）
数据来源：荧光腹腔镜下直肠癌根治术 ICG 灌注研究（n=376）
"""
import os
import numpy as np
import pandas as pd
import joblib
import streamlit as st

# ==============================================================================
# 页面配置
# ==============================================================================
st.set_page_config(
    page_title="LCA 灌注损失风险预测计算器",
    page_icon="🩺",
    layout="centered",
    initial_sidebar_state="expanded",
)

# ==============================================================================
# 加载模型
# ==============================================================================
WORK_DIR = os.path.dirname(os.path.abspath(__file__))
model_path = os.path.join(WORK_DIR, 'best_prediction_model.pkl')
scaler_path = os.path.join(WORK_DIR, 'feature_scaler.pkl')

# 核心特征（多宇宙LASSO交集，100%稳定性）
CORE_VARS = ['Ctvalue', 'diameter', 'Aeterial_cl', 'LCA_dis', 'CEA']

# 风险分层截点（三分位数法，经4种截点方法敏感性分析验证）
RISK_LOW_CUTOFF = 0.343
RISK_HIGH_CUTOFF = 0.679

try:
    best_model = joblib.load(model_path)
    MODEL_LOADED = True
    MODEL_TYPE = type(best_model).__name__
except Exception as e:
    MODEL_LOADED = False
    MODEL_TYPE = "Unknown"
    st.error(f"模型加载失败: {e}")
    st.info("请先运行 prediction_model_complete_analysis.py 生成模型文件。")

# ==============================================================================
# 变量标签和选项
# ==============================================================================
VAR_INFO = {
    'Ctvalue': {
        'label': 'LCA/IMA CT 值比值',
        'help': '左结肠动脉/肠系膜下动脉 CT 值比值（危险因素：>0.52）',
        'options': {0: '≤ 0.52（低风险）', 1: '> 0.52（高风险）'},
        'direction': '危险因素'
    },
    'diameter': {
        'label': 'LCA/IMA 短径比值',
        'help': '左结肠动脉/肠系膜下动脉短径比值（危险因素：>0.63）',
        'options': {0: '≤ 0.63（低风险）', 1: '> 0.63（高风险）'},
        'direction': '危险因素'
    },
    'Aeterial_cl': {
        'label': 'LCA 发出分型',
        'help': '左结肠动脉自肠系膜下动脉发出的解剖分型（危险因素：三支型）',
        'options': {0: '直乙共干型/左乙共干型 (Type 1，低风险)', 1: '三支型 (Type 2，高风险)'},
        'direction': '危险因素'
    },
    'LCA_dis': {
        'label': 'LCA 发出距离',
        'help': '左结肠动脉自肠系膜下动脉发出的距离（保护因素：≤3.5cm为高风险取值）',
        'options': {0: '≤ 3.5 cm（高风险取值）', 1: '> 3.5 cm（低风险取值）'},
        'direction': '保护因素'
    },
    'CEA': {
        'label': 'CEA 癌胚抗原',
        'help': '血清癌胚抗原水平（保护因素：≤5为高风险取值）',
        'options': {0: '≤ 5 ng/mL（高风险取值）', 1: '> 5 ng/mL（低风险取值）'},
        'direction': '保护因素'
    },
}

# ==============================================================================
# 标题和研究背景
# ==============================================================================
st.title("🩺 LCA 灌注损失风险预测计算器")
st.markdown("---")

with st.expander("📖 研究背景与方法学说明", expanded=False):
    st.markdown("""
    **研究背景**：荧光腹腔镜下直肠癌根治术中，吲哚菁绿（ICG）荧光显色评估
    左结肠动脉（LCA）夹闭前后直肠血流灌注变化。slopDA = (slop1 - slop2) / slop1，
    值越高表示夹闭 LCA 后灌注损失百分比越高，即 LCA 依赖度越高。

    **预测目标**：slopDA ≥ 0.30（高 LCA 依赖度，建议保留 LCA）的概率。

    **核心特征**（多宇宙 LASSO 交集，100% 稳定性，经4种截点方法验证）：
    - LCA/IMA CT 值比值（OR=3.80，P<0.0001，最强预测因素）
    - LCA/IMA 短径比值（OR=2.44，P=0.0003）
    - LCA 发出分型（OR=2.88，P=0.011，三支型为危险因素）
    - LCA 发出距离（OR=0.58，P=0.028，保护因素）
    - CEA 癌胚抗原（OR=0.62，P=0.058，边缘显著，保护因素）

    **模型性能**（独立测试集 n=76）：
    - AUC = 0.793，准确率 = 77.6%，敏感度 = 82.9%，特异度 = 71.4%
    - Brier Score = 0.179，Hosmer-Lemeshow 检验 P = 0.720（校准良好）

    **风险分层**（三分位数法，经4种截点方法敏感性分析验证）：
    - 低风险：概率 < 0.343（事件率 18.9%）
    - 中风险：概率 0.343 - 0.679（事件率 63.8%）
    - 高风险：概率 > 0.679（事件率 80.3%）
    """)

if MODEL_LOADED:
    st.success(f"✅ 模型已加载: {MODEL_TYPE}（Logistic Regression Pipeline）")
else:
    st.warning("⚠️ 模型未加载，以下仅为界面演示")

st.markdown("---")

# ==============================================================================
# 输入区域
# ==============================================================================
st.subheader("📋 患者临床参数输入")

input_values = {}
col1, col2 = st.columns(2)

for i, var in enumerate(CORE_VARS):
    info = VAR_INFO.get(var, {'label': var, 'help': '', 'options': {0: '0', 1: '1'}})
    options = info['options']
    display_options = {v: k for k, v in options.items()}
    
    with col1 if i % 2 == 0 else col2:
        direction_icon = "⚠️" if info.get('direction') == '危险因素' else "🛡️"
        choice = st.selectbox(
            f"**{direction_icon} {info['label']}**",
            options=list(display_options.keys()),
            help=info.get('help', ''),
            key=f"input_{var}",
        )
        input_values[var] = display_options[choice]

st.markdown("---")

# ==============================================================================
# 预测按钮
# ==============================================================================
col1, col2, col3 = st.columns([1, 2, 1])
with col2:
    predict_btn = st.button("🔍 开始预测", type="primary", use_container_width=True)

if predict_btn:
    if MODEL_LOADED:
        # 构建输入向量（Pipeline会自动处理scaler）
        X_input = pd.DataFrame([input_values])[CORE_VARS].values.astype(float)
        prob = best_model.predict_proba(X_input)[0, 1]
        pred_class = 1 if prob >= 0.5 else 0

        # 风险分层
        if prob >= RISK_HIGH_CUTOFF:
            risk_level = "🔴 高风险"
            risk_color = "red"
        elif prob >= RISK_LOW_CUTOFF:
            risk_level = "🟡 中风险"
            risk_color = "orange"
        else:
            risk_level = "🟢 低风险"
            risk_color = "green"

        st.markdown("---")
        st.subheader("📊 预测结果")

        # 概率显示
        col_a, col_b, col_c = st.columns(3)
        with col_a:
            st.metric("高 LCA 依赖概率", f"{prob*100:.1f}%")
        with col_b:
            st.metric("预测分类", "高依赖 (≥30%)" if pred_class == 1 else "低依赖 (<30%)")
        with col_c:
            st.metric("风险等级", risk_level)

        # 概率进度条（带颜色）
        st.progress(float(prob))

        # 风险分层说明
        st.info(f"""
        **风险分层标准**（三分位数法，经4种截点方法敏感性分析验证）：
        - 🟢 低风险：概率 < {RISK_LOW_CUTOFF*100:.1f}%（事件率 18.9%）
        - 🟡 中风险：概率 {RISK_LOW_CUTOFF*100:.1f}% - {RISK_HIGH_CUTOFF*100:.1f}%（事件率 63.8%）
        - 🔴 高风险：概率 > {RISK_HIGH_CUTOFF*100:.1f}%（事件率 80.3%）
        """)

        # 临床建议
        st.markdown("### 💡 临床建议")
        if prob >= RISK_HIGH_CUTOFF:
            st.error(f"""
            **高 LCA 依赖度（概率 ≥ {RISK_HIGH_CUTOFF*100:.0f}%）**：该患者直肠血流灌注高度依赖左结肠动脉。
            建议术中**保留 LCA**（低位结扎），以确保吻合口充足血供，降低吻合口瘘风险。
            术中仍需结合 ICG 荧光造影实时评估最终确认。
            """)
        elif prob >= RISK_LOW_CUTOFF:
            st.warning(f"""
            **中等 LCA 依赖度（概率 {RISK_LOW_CUTOFF*100:.0f}%-{RISK_HIGH_CUTOFF*100:.0f}%）**：该患者对左结肠动脉有一定依赖。
            建议术中结合 ICG 荧光造影实时评估，谨慎决定 LCA 取舍。
            若行高位结扎，需确认边缘动脉弓灌注充足。
            """)
        else:
            st.success(f"""
            **低 LCA 依赖度（概率 < {RISK_LOW_CUTOFF*100:.0f}%）**：该患者直肠血流灌注对左结肠动脉依赖较低。
            边缘动脉弓代偿良好，高位结扎（牺牲 LCA）在血供方面相对安全。
            仍需结合术中 ICG 荧光造影最终确认。
            """)

        # 输入回显
        with st.expander("📝 查看输入参数与特征方向"):
            df_display = pd.DataFrame([{
                '特征': VAR_INFO.get(v, {'label': v})['label'],
                '取值': VAR_INFO.get(v, {'options': {0: '0', 1: '1'}})['options'].get(input_values[v], str(input_values[v])),
                '方向': VAR_INFO.get(v, {'direction': '未知'})['direction'],
                '编码值': input_values[v]
            } for v in CORE_VARS])
            st.dataframe(df_display, use_container_width=True, hide_index=True)

        # 列线图评分说明
        with st.expander("📐 列线图评分参考（Logistic Regression）"):
            st.markdown("""
            **各特征高风险取值分数**（最大系数对应100分）：
            - LCA/IMA CT 值比值（>0.52）：100.0 分
            - LCA/IMA 短径比值（>0.63）：70.5 分
            - LCA 发出分型（三支型）：55.7 分
            - LCA 发出距离（≤3.5cm，保护因素）：35.8 分
            - CEA（≤5 ng/mL，保护因素）：32.7 分
            
            **总分范围**：0 - 294.7 分
            **使用方法**：各特征分数相加得到总分，对应风险概率。
            """)

    else:
        st.error("模型未加载，无法预测。请先运行 prediction_model_complete_analysis.py 生成模型文件。")

# ==============================================================================
# 底部信息
# ==============================================================================
st.markdown("---")
st.caption("""
**模型说明**：本计算器基于多宇宙 LASSO 交集核心特征构建，
采用训练集表现最优的 Logistic Regression 算法（AUC=0.793）。
风险分层截点经4种截点确定方法（三分位数、Youden指数、80%灵敏度/特异度、NPV/PPV）
敏感性分析验证，结论稳健。预测结果仅供临床参考，
不能替代医生的专业判断和术中 ICG 荧光造影的实时评估。

**数据来源**：荧光腹腔镜下直肠癌根治术 ICG 灌注研究（n=376，训练集300例，测试集76例）
**模型文件**：best_prediction_model.pkl（Logistic Regression Pipeline）
""")
