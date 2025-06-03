import sys
import os
import streamlit as st
import pandas as pd
import joblib

# 尝试导入 sklearn
try:
    from sklearn.preprocessing import LabelEncoder
except ImportError:
    st.error("""
    **Required package 'scikit-learn' is not installed.**
    
    Please install it using one of these methods:
    
    1. **Local installation**: Run `pip install scikit-learn`
    2. **Streamlit Cloud**: Add 'scikit-learn' to your requirements.txt file
    """)
    st.stop()

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
        
        # 使用颜色编码的风险指示器
        if probability < 30:
            st.success("✅ Low risk: Routine monitoring recommended")
            color = "green"
        elif probability < 70:
            st.warning("⚠️ Medium risk: Close monitoring advised")
            color = "orange"
        else:
            st.error("❗ High risk: Immediate intervention recommended")
            color = "red"
        
        # 创建自定义进度条
        progress_html = f"""
        <div style="margin-top:10px; margin-bottom:20px;">
            <div style="display:flex; justify-content:space-between; margin-bottom:5px;">
                <span>0%</span>
                <span>50%</span>
                <span>100%</span>
            </div>
            <div style="height:20px; background:#eee; border-radius:10px; overflow:hidden;">
                <div style="height:100%; width:{probability}%; background:{color};"></div>
            </div>
        </div>
        """
        st.markdown(progress_html, unsafe_allow_html=True)
        
        # 添加详细解释
        st.subheader("Risk Factors Breakdown")
        
        # 创建风险因素解释表
        risk_table = f"""
        <table style="width:100%; border-collapse:collapse; margin-bottom:20px;">
            <tr style="background-color:#f0f0f0;">
                <th style="border:1px solid #ddd; padding:8px; text-align:left;">Factor</th>
                <th style="border:1px solid #ddd; padding:8px; text-align:left;">Your Value</th>
                <th style="border:1px solid #ddd; padding:8px; text-align:left;">Risk Level</th>
            </tr>
            <tr>
                <td style="border:1px solid #ddd; padding:8px;">Age</td>
                <td style="border:1px solid #ddd; padding:8px;">{features_df['Age'].iloc[0]}</td>
                <td style="border:1px solid #ddd; padding:8px;">{'High' if features_df['Age'].iloc[0] > 60 else 'Medium' if features_df['Age'].iloc[0] > 40 else 'Low'}</td>
            </tr>
            <tr>
                <td style="border:1px solid #ddd; padding:8px;">BMI</td>
                <td style="border:1px solid #ddd; padding:8px;">{features_df['BMI'].iloc[0]}</td>
                <td style="border:1px solid #ddd; padding:8px;">{'High' if features_df['BMI'].iloc[0] > 30 else 'Medium' if features_df['BMI'].iloc[0] > 25 else 'Low'}</td>
            </tr>
            <tr>
                <td style="border:1px solid #ddd; padding:8px;">Diabetes</td>
                <td style="border:1px solid #ddd; padding:8px;">{'Yes' if features_df['Diabetes'].iloc[0] == 1 else 'No'}</td>
                <td style="border:1px solid #ddd; padding:8px;">{'High' if features_df['Diabetes'].iloc[0] == 1 else 'Low'}</td>
            </tr>
            <tr>
                <td style="border:1px solid #ddd; padding:8px;">Procedure Time</td>
                <td style="border:1px solid #ddd; padding:8px;">{features_df['time'].iloc[0]} minutes</td>
                <td style="border:1px solid #ddd; padding:8px;">{'High' if features_df['time'].iloc[0] > 300 else 'Medium' if features_df['time'].iloc[0] > 200 else 'Low'}</td>
            </tr>
            <tr>
                <td style="border:1px solid #ddd; padding:8px;">Abdominal Issue</td>
                <td style="border:1px solid #ddd; padding:8px;">{'Yes' if features_df['Abdominal'].iloc[0] == 1 else 'No'}</td>
                <td style="border:1px solid #ddd; padding:8px;">{'High' if features_df['Abdominal'].iloc[0] == 1 else 'Low'}</td>
            </tr>
        </table>
        """
        st.markdown(risk_table, unsafe_allow_html=True)
        
        # 添加临床建议
        st.subheader("Clinical Recommendations")
        if probability < 30:
            st.info("**Low Risk Protocol:**")
            st.info("- Continue regular follow-up visits")
            st.info("- Maintain healthy lifestyle habits")
            st.info("- Annual check-up recommended")
        elif probability < 70:
            st.warning("**Medium Risk Protocol:**")
            st.warning("- Schedule follow-up within 1 month")
            st.warning("- Consider additional diagnostic tests")
            st.warning("- Monitor for symptoms closely")
        else:
            st.error("**High Risk Protocol:**")
            st.error("- Seek immediate medical consultation")
            st.error("- Consider hospitalization for observation")
            st.error("- Perform comprehensive diagnostic evaluation")
    
    except Exception as e:
        st.error(f"Prediction failed: {str(e)}")
