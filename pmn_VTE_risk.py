
#基于streamlit构建网页预测平台
import streamlit as st
import joblib
import shap
import pandas as pd
import matplotlib
from matplotlib import pyplot as plt
matplotlib.rc("font", family='YouYuan')

# 加载模型
model = joblib.load('1_knn_BSMO_frequent_train_data_NGBoost.joblib')
# 加载 SHAP 解释器
explainer = shap.TreeExplainer(model)  # 适用于树模型
columns = ['Recurrent nephrotic syndrome (YES/NO)', 'umALB/Ucr (mg/g)', 'Statins (YES/NO)', 'D-Dimer (mg/L)', 'FDP > 5mg/L (YES/NO)', 'International Normalized Ratio (INR)', 'AT III activity(%)', 'Albumin (g/L)', 'PLA2R Antibody（RU/ml）', 'Cholinesterase (KU/L)']
st.set_page_config(page_title="PMN患者VTE风险预测", page_icon=":🧠:", layout="wide", initial_sidebar_state="expanded")

predict_button_pressed = False
# 创建 Streamlit 应用

title = "👨‍⚕️VTE Risk Prediction for Primary Membranous Nephropathy👨‍⚕️"
styled_title = title.replace("VTE", "<span style='color:red'>「VTE」</span>")
st.markdown(f"<h1 style='text-align: center;'>{styled_title}</h1>", unsafe_allow_html=True)
st.divider()
result_placeholder = st.empty()
show_feature_input = True

# 分为两列展示用户输入字段
if show_feature_input:
    with result_placeholder.container():
        st.markdown(
            ':orange[⚠️**Please note the input units**；⚠️**Complete all input fields to calculate risk**；⚠️**Prediction results are for reference only. Please consult a professional physician**。]')
        col1, col2 = st.columns(2)

        with col1:

            umALBUcr = st.number_input(columns[1], min_value=0.0, max_value=50000.0, value=25.0)
            RECURRENCE = st.selectbox(columns[0], ['YES', 'NO'])
            Statins = st.selectbox(columns[2], ['YES', 'NO'])
            DD = st.number_input(columns[3], min_value=0.0, max_value=2000.0, value=1.0)
            FDP  = st.selectbox(columns[4], ['YES', 'NO'])


        with col2:
            PLA2R = st.number_input(columns[8], min_value=0.0, max_value=2000.0, value=105.0)
            ATIII  = st.number_input(columns[6], min_value=0.0, max_value=2000.0, value=89.0)
            ALB = st.number_input(columns[7], min_value=0.0, max_value=2000.0, value=19.0)
            CHE = st.number_input(columns[9], min_value=0.0, max_value=2000.0, value=10.0)
            PTINR = st.number_input(columns[5], min_value=0.5, max_value=200.0, value=1.80)
            predict_button = st.button('Click to Predict')

            # 对非数值型特征进行独热编码
            RECURRENCE_encoded = 1 if RECURRENCE == 'YES' else 0
            Statins_encoded = 1 if Statins == 'YES' else 0
            FDP_encoded = 1 if FDP =='YES' else 0


# 根据用户输入进行预测和解释

if predict_button:
    show_feature_input = False
    input_data = pd.DataFrame({
        'Recurrent nephrotic syndrome': [RECURRENCE_encoded],
        'DD': [DD],
        'umALB/Ucr': [umALBUcr],
        'Statins': [Statins_encoded],
        'ALB': [ALB],
        'CHE': [CHE],
        'FDP > 5mg/L': [FDP_encoded],
        'INR': [PTINR],
        'aPLA2Rab': [PLA2R],
        'AT III activity': [ATIII],

    })

    prediction = model.predict_proba(input_data)[:, 1][0]

    # 清除之前的内容
    result_placeholder.empty()
    if not show_feature_input:
        if st.button('Predict Again'):
            show_feature_input = True
    if prediction < 0.4:
        st.markdown(
            f"<div style='text-align:center;'><span style='color:black; font-weight:bold; font-size:30px;'>Your risk of developing VTE in the next 6 months is：<span style='color:green; font-weight:bold; font-size:40px;'>{prediction:.2%} - Low Risk</span></div>",
            unsafe_allow_html=True)
        st.subheader('Below are the individualized risk analysis results：',divider='rainbow')
    elif prediction < 0.7:
        st.markdown(
            f"<div style='text-align:center;'><span style='color:black; font-weight:bold; font-size:30px;'>Your risk of developing VTE in the next 6 months is：<span style='color:orange; font-weight:bold; font-size:40px;'>{prediction:.2%} - Moderate Risk</span></div>",
            unsafe_allow_html=True)
        st.subheader('Below are the individualized risk analysis results', divider='rainbow')
    else:
        st.markdown(
            f"<div style='text-align:center;'><span style='color:black; font-weight:bold; font-size:30px;'>Your risk of developing VTE in the next 6 months is：<span style='color:red; font-weight:bold; font-size:40px;'>{prediction:.2%} - High Risk</span></div>",
            unsafe_allow_html=True)
        st.subheader('Below are the individualized risk analysis results',divider='rainbow')
    # 医学参考范围
    reference_ranges = {

        'umALB/Ucr': [0.000, 30.00],
        'DD': [0.00, 0.55],
        'INR': [0.96, 1.16],
        'ALB': [40.0, 55.0],
        'AT III activity': [79.4, 112],
        'aPLA2Rab': [0.00, 14.00],
        'CHE': [5.0, 12.0]

    }

    # 计算各指标状态
    def get_status(value, ref_range):
        if value < ref_range[0]:
            return 'Low'
        elif value > ref_range[1]:
            return 'High'
        else:
            return 'Normal'


    # 创建DataFrame存储各指标状态
    column = input_data.columns.tolist()
    column.remove('Recurrent nephrotic syndrome')
    column.remove('Statins')
    column.remove('FDP > 5mg/L')
    status_data = pd.DataFrame(index=['Value','Status'], columns=column)
    for col in column:
        status_data[col]['Value']=input_data[col][0]
        status_data[col]['Status']= get_status(input_data[col][0], reference_ranges[col])
    status_data = status_data.T.reset_index().rename(columns={'index': 'Indicator'})

    st.write('---')

    # 显示 SHAP 图
    shap_values = explainer.shap_values(input_data)
    shap_values2 = explainer(input_data)

    # 创建两列布局
    col1, col2 ,col3= st.columns([1,2,2],gap="medium")

    # 在第一列显示 SHAP Summary Plot
    with col1:
        # 显示用户输入数据和各指标状态
        st.subheader('Status of Each Indicator:')
        st.dataframe(status_data,hide_index=True,width = 400,row_height=50)

    with col2:
        st.subheader('Ranking of Individualized VTE Risk Factors')
        fig, ax = plt.subplots(figsize=(6, 6))  # 设置图片尺寸
        shap.summary_plot(shap_values,features=input_data, plot_type='bar', show=False)
        plt.xlabel('Degree of Impact')
        st.pyplot(fig)

    # 在第二列显示瀑布图
    with col3:
        st.subheader(' Individualized VTE Risk Composition')
        fig2, ax2 = plt.subplots(figsize=(6, 6))  # 设置图片尺寸
        shap.plots.waterfall(shap_values2[0], show=False)
        st.pyplot(fig2)
        mutl = '''📌:red[Red] indicates the indicator increases VTE risk;
        :blue[Blue] indicates the indicator does not increase VTE risk;
        :green[Value] indicates the magnitude of the effect。'''
        st.markdown(mutl)



    
