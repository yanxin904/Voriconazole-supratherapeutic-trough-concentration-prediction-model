#以下内容来源：微信公众号一篇-收藏夹里找
import streamlit as st
import joblib
import numpy as np
import pandas as pd
import shap
import matplotlib.pyplot as plt

# Load the model
model_path=r'D:\常德市一\科研课题\伏立康唑高血药浓度\数据\所有数据分析\模型评估\9个变量\XGB_Final_Model.pkl'
saved_model=joblib.load(model_path)
model=saved_model["model"]

# 如果有分类变量，需要定义特征选项
#Quinolone_options = {
#    1: 'use',
#    0: 'not use'
#}

# Define feature names
feature_names = ['AGE', 'Duration of treatment', 'ALB','A/G', 'ALT', 'LYM', 'hs-CRP', 'PCT', 'TP']

# Streamlit user interface
st.title("Voriconazole trough concentration >5 mg/L")

#设置变量输入
Age = st.number_input("AGE:", min_value=0.0, max_value=100.0, value=65.0) #value是默认值的意思，选择中位数
Duration_of_treatment = st.number_input("Duration of treatment:", min_value=0.0, max_value=3650.0, value=7.0)
ALB = st.number_input("ALB:", min_value=0.0, max_value=100.0, value=35.0)
A_G = st.number_input("A/G:", min_value=0.0, max_value=10.0, value=1.2)
ALT = st.number_input("ALT:", min_value=0.0, max_value=5000.0, value=30.0)
LYM = st.number_input("LYM:", min_value=0.0, max_value=150.0, value=1.0)
hs_CRP = st.number_input("hs-CRP:", min_value=0.0, max_value=1000.0, value=10.0)
PCT = st.number_input("PCT:", min_value=0.0, max_value=1000.0, value=0.5)
TP = st.number_input("TP:", min_value=0.0, max_value=150.0, value=65.0)
# 如果是二分类变量，这里设置选择框
#Carbapenems = st.selectbox("Chest pain type:", options=list(cp_options.keys()), format_func=lambda x: cp_options[x])

# Process inputs and make predictions
feature_values = [Age, Duration_of_treatment,ALB,A_G,ALT,LYM,hs_CRP,PCT,TP]
features = np.array([feature_values])

if st.button("Predict"):
    # Predict class and probabilities
    threshold = 0.439  # 设置新的阈值
    predicted_proba = model.predict_proba(features)[0]  # # 获取第一个样本的预测概率（这是一个概率数组）

    # 自动识别类别1（High=1）对应的概率位置
    classes = np.asarray(model.classes_)
    positive_indices = np.where(classes == 1)[0]

    if len(positive_indices) != 1:
        st.error(f"无法识别High=1对应的概率列，模型类别为：{classes}")
        st.stop()

    positive_idx = positive_indices[0]
    # 始终取High=1，即血药浓度>5 mg/L的预测概率
    probability = float(predicted_proba[positive_idx])
    # 根据固定阈值0.439进行分类
    predicted_class = int(probability >= threshold)
    # 显示结果
    risk_label = "High risk" if predicted_class == 1 else "Low risk"
    st.write(f"**Predicted risk classification:** {risk_label} (Class {predicted_class})")
    st.write(
        f"**Probability of voriconazole trough concentration >5 mg/L:** "
        f"{probability * 100:.1f}%")

    if predicted_class == 1:
        advice = (
            f"According to our model, this patient is at an elevated predicted risk of having a voriconazole trough concentration exceeding 5 mg/L. "
            f"The predicted probability is {probability * 100:.1f}%. "
            f"This estimate reflects the model's prediction and does not confirm that the patient's actual voriconazole concentration is elevated. "
            f"We recommend prioritizing TDM to assess the patient's actual voriconazole trough concentration."
            f"The dosing regimen, liver function, concomitant medications, and potential drug–drug interactions should also be reviewed. "
            f"Voriconazole-related adverse effects, particularly hepatotoxicity and neurological or visual disturbances, should be monitored. "
            f"Any dose adjustment should be individualized according to the measured drug concentration, clinical response, tolerability, and overall clinical status. "
        )
    else:
        advice = (
            f"According to our model, this patient is at a low predicted risk of having a voriconazole trough concentration exceeding 5 mg/L"
            f"The predicted probability is {probability * 100:.1f}%. "
            f"Although the predicted risk is low, this result does not exclude the possibility of supratherapeutic voriconazole exposure."
            f"The need for TDM should continue to be assessed according to the patient's clinical status, treatment phase, liver function, and potential drug-drug interactions."
            f"Treatment decisions should be guided by measured voriconazole concentrations, when available, together with the patient's clinical response, tolerability, and overall clinical condition."
        )

    st.write(advice)

    # Calculate SHAP values and display force plot
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(features)

    # 绘制 SHAP force plot，确保传递 base_value 为第一个参数  如果是RF模型则用explainer.expected_value[1], shap_values[1]
    shap.plots.force(explainer.expected_value, shap_values, pd.DataFrame([feature_values], columns=feature_names),
                     matplotlib=True)

    plt.savefig("shap_force_plot.png", bbox_inches='tight', dpi=1200)

    st.image("shap_force_plot.png")

#确保在命令行中运行 Streamlit 应用程序，使用以下命令：
#按 Win + R 打开运行对话框，输入 cmd 并按回车，打开命令提示符。
# 在python的终端 输入命令：streamlit run "D:\project\伏立康唑高血药浓度\Streamlit-伏立康唑.py"
