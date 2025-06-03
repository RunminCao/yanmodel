import sys
import os
import streamlit as st
import pandas as pd
from sklearn.preprocessing import LabelEncoder
import joblib

# 加载模型
try:
    # 使用相对路径并检查文件存在性
    model_path = 'rf.pkl'
    if not os.path.exists(model_path):
        st.error(f"Model file '{model_path}' not found. Please ensure it's in the same directory as this app.")
        st.stop()
    
    model = joblib.load(model_path)
except Exception as e:
    st.error(f"Error loading model: {str(e)}")
    st.stop()

# 特征范围定义
feature_names = ["Age", "BMI", "Diabetes", "time", "Abdominal"]
feature_ranges = {
    "Age": {"type": "numerical", "min": 18, "max": 100, "default": 18},
    "BMI": {"type": "numerical", "min": 10, "max": 100, "default": 10},
    "Diabetes": {"type": "categorical", "options": ["0", "1"]},
    "time": {"type": "numerical", "min": 100, "max": 600, "default": 100},
    "Abdominal": {"type": "categorical", "options": ["0", "1"]},
}

# 初始化分类特征的编码器
label_encoders = {}
for feature, properties in feature_ranges.items():
    if properties["type"] == "categorical":
        try:
            le = LabelEncoder()
            le.fit(properties["options"])
            label_encoders[feature] = le
        except Exception as e:
            st.error(f"Error initializing encoder for {feature}: {str(e)}")
            st.stop()

# Streamlit 界面
st.title("Early Complications Prediction Model")
st.markdown("""
This app predicts the risk of early complications based on patient characteristics.
Adjust the parameters using the sliders and dropdowns below.
""")

# 创建两列布局
col1, col2 = st.columns(2)

with col1:
    st.subheader("Patient Demographics")
    feature_values = {}
    for feature in ["Age", "BMI"]:
        properties = feature_ranges[feature]
        feature_values[feature] = st.slider(
            label=f"{feature} ({properties['min']} - {properties['max']})",
            min_value=float(properties["min"]),
            max_value=float(properties["max"]),
            value=float(properties["default"]),
            step=1.0
        )

with col2:
    st.subheader("Medical Information")
    for feature in ["Diabetes", "Abdominal"]:
        properties = feature_ranges[feature]
        selected = st.selectbox(
            label=f"{feature}",
            options=properties["options"],
            index=0  # 默认选择第一个选项
        )
        feature_values[feature] = label_encoders[feature].transform([selected])[0]
    
    # 单独处理 time 特征
    properties = feature_ranges["time"]
    feature_values["time"] = st.slider(
        label=f"Procedure Time (minutes)",
        min_value=float(properties["min"]),
        max_value=float(properties["max"]),
        value=float(properties["default"]),
        step=5.0
    )

# 转换为模型输入格式
try:
    features_df = pd.DataFrame([feature_values], columns=feature_names)
    st.subheader("Input Summary")
    st.dataframe(features_df)
except Exception as e:
    st.error(f"Error creating input data: {str(e)}")
    st.stop()

# 预测按钮和结果展示
if st.button("Predict Risk", type="primary"):
    try:
        # 模型预测
        predicted_proba = model.predict_proba(features_df)[0]
        
        # 获取预测概率（假设索引1代表并发症）
        probability = predicted_proba[1] * 100
        
        # 显示预测结果
        st.subheader("Prediction Result")
        
        # 使用进度条和指标显示结果
        st.metric(label="Risk of Early Complications", value=f"{probability:.1f}%")
        
        # 使用颜色编码的进度条
        progress_bar = st.progress(0)
        if probability < 30:
            color = "green"
            message = "✅ Low risk: Routine monitoring recommended"
        elif probability < 70:
            color = "orange"
            message = "⚠️ Medium risk: Close monitoring advised"
        else:
            color = "red"
            message = "❗ High risk: Immediate intervention recommended"
        
        # 设置进度条颜色
        progress_bar.progress(int(probability) / 100)
        
        # 显示风险信息
        if color == "green":
            st.success(message)
        elif color == "orange":
            st.warning(message)
        else:
            st.error(message)
        
        # 添加详细解释
        st.subheader("Risk Factors Breakdown")
        
        # 创建风险因素解释表
        risk_factors = pd.DataFrame({
            "Factor": feature_names,
            "Your Value": features_df.iloc[0].values,
            "Contribution": "To be determined"  # 这里可以添加实际的影响分析
        })
        
        st.table(risk_factors)
        
        # 添加临床建议
        st.subheader("Clinical Recommendations")
        if probability < 30:
            st.info("- Continue regular follow-up visits")
            st.info("- Maintain healthy lifestyle habits")
            st.info("- Annual check-up recommended")
        elif probability < 70:
            st.warning("- Schedule follow-up within 1 month")
            st.warning("- Consider additional diagnostic tests")
            st.warning("- Monitor for symptoms closely")
        else:
            st.error("- Seek immediate medical consultation")
            st.error("- Consider hospitalization for observation")
            st.error("- Perform comprehensive diagnostic evaluation")
    
    except Exception as e:
        st.error(f"Prediction failed: {str(e)}")
