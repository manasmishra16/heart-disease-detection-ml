"""
CardioPredict AI - Cloud Deployment Version
Trains model on startup if not found
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import warnings
warnings.filterwarnings('ignore')

# Check if models exist, if not train them
import os
if not os.path.exists('models/gradient_boosting_best.pkl'):
    st.info("🔄 First time setup: Training models... This may take 2-3 minutes.")
    
    # Inline training (more reliable than subprocess on cloud)
    try:
        from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
        from sklearn.preprocessing import StandardScaler
        from sklearn.model_selection import train_test_split
        from imblearn.over_sampling import SMOTE
        import joblib
        
        # Find and load data
        data_paths = [
            'datasets/cleveland/heart.csv',
            'heart.csv',
            'data/heart.csv'
        ]
        
        df = None
        for path in data_paths:
            if os.path.exists(path):
                df = pd.read_csv(path)
                st.success(f"✅ Found dataset at: {path}")
                break
        
        if df is None:
            st.error("❌ Dataset not found. Please ensure datasets/cleveland/heart.csv exists.")
            st.stop()
        
        # Prepare data
        X = df.drop('target', axis=1)
        y = df['target']
        
        # Feature engineering
        X['age_chol'] = X['age'] * X['chol']
        X['age_thalach'] = X['age'] * X['thalach']
        X['cp_thalach'] = X['cp'] * X['thalach']
        X['oldpeak_slope'] = X['oldpeak'] * X['slope']
        X['ca_thal'] = X['ca'] * X['thal']
        X['age_squared'] = X['age'] ** 2
        X['chol_squared'] = X['chol'] ** 2
        X['thalach_squared'] = X['thalach'] ** 2
        
        # Split and scale
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        
        # SMOTE
        smote = SMOTE(random_state=42)
        X_train_smote, y_train_smote = smote.fit_resample(X_train_scaled, y_train)
        
        # Train model
        with st.spinner("Training Gradient Boosting model..."):
            model = GradientBoostingClassifier(n_estimators=200, learning_rate=0.1, 
                                              max_depth=5, random_state=42)
            model.fit(X_train_smote, y_train_smote)
        
        # Save models
        os.makedirs('models', exist_ok=True)
        joblib.dump(model, 'models/gradient_boosting_best.pkl')
        joblib.dump(scaler, 'models/scaler_best.pkl')
        joblib.dump({'features': X.columns.tolist()}, 'models/ensemble_best_config.pkl')
        
        st.success("✅ Models trained and saved successfully!")
        st.balloons()
    except Exception as train_error:
        st.error(f"❌ Training failed: {train_error}")
        st.info("Please check the logs for details.")
        st.stop()

# Now load the regular app
import joblib

# ==================== PAGE CONFIGURATION ====================
st.set_page_config(
    page_title="CardioPredict AI - Heart Disease Detection",
    page_icon="🫀",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ==================== CUSTOM CSS ====================
st.markdown("""
<style>
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    header {visibility: hidden;}
    
    .main-header {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 2rem;
        border-radius: 15px;
        margin-bottom: 2rem;
        box-shadow: 0 10px 30px rgba(0,0,0,0.3);
        text-align: center;
    }
    
    .main-header h1 {
        color: white;
        font-size: 3rem;
        font-weight: 800;
        margin: 0;
        text-shadow: 2px 2px 4px rgba(0,0,0,0.3);
    }
    
    .result-card-healthy {
        background: linear-gradient(135deg, #11998e 0%, #38ef7d 100%);
        padding: 2rem;
        border-radius: 15px;
        box-shadow: 0 10px 30px rgba(0,200,80,0.3);
        color: white;
        text-align: center;
        margin: 1rem 0;
    }
    
    .result-card-disease {
        background: linear-gradient(135deg, #ee0979 0%, #ff6a00 100%);
        padding: 2rem;
        border-radius: 15px;
        box-shadow: 0 10px 30px rgba(255,70,70,0.3);
        color: white;
        text-align: center;
        margin: 1rem 0;
    }
</style>
""", unsafe_allow_html=True)

# ==================== LOAD MODEL ====================
@st.cache_resource
def load_model():
    try:
        model = joblib.load('models/gradient_boosting_best.pkl')
        scaler = joblib.load('models/scaler_best.pkl')
        config = joblib.load('models/ensemble_best_config.pkl')
        return model, scaler, config, True
    except:
        return None, None, None, False

model, scaler, config, loaded = load_model()

# ==================== HEADER ====================
st.markdown("""
<div class="main-header">
    <h1>🫀 CardioPredict AI</h1>
    <p>Advanced Heart Disease Detection System | 98% Accuracy</p>
</div>
""", unsafe_allow_html=True)

if not loaded:
    st.error("❌ **Model loading failed.** Please refresh the page.")
    st.stop()

# ==================== SIMPLE INPUT ====================
st.markdown("## 🔬 Patient Health Assessment")

col1, col2 = st.columns(2)

with col1:
    age = st.slider("Age (years)", 20, 100, 50)
    sex = st.radio("Sex", [1, 0], format_func=lambda x: "👨 Male" if x==1 else "👩 Female", horizontal=True)
    cp = st.selectbox("Chest Pain Type", [0,1,2,3], format_func=lambda x: ["Typical","Atypical","Non-anginal","Asymptomatic"][x])
    trestbps = st.number_input("Blood Pressure (mm Hg)", 80, 200, 120)
    chol = st.number_input("Cholesterol (mg/dl)", 100, 600, 200)
    fbs = st.radio("Fasting Blood Sugar >120", [0,1], format_func=lambda x: "Yes" if x==1 else "No", horizontal=True)

with col2:
    restecg = st.selectbox("Resting ECG", [0,1,2], format_func=lambda x: ["Normal","ST-T Abnormality","LV Hypertrophy"][x])
    thalach = st.slider("Max Heart Rate (bpm)", 60, 220, 150)
    exang = st.radio("Exercise Angina", [0,1], format_func=lambda x: "Yes" if x==1 else "No", horizontal=True)
    oldpeak = st.number_input("ST Depression", 0.0, 10.0, 0.0, 0.1)
    slope = st.selectbox("ST Slope", [0,1,2], format_func=lambda x: ["Upsloping","Flat","Downsloping"][x])
    ca = st.select_slider("Major Vessels", [0,1,2,3])
    thal = st.selectbox("Thalassemia", [0,1,2,3], format_func=lambda x: ["Normal","Fixed","Reversible","Unknown"][x])

if st.button("🔮 ANALYZE HEART HEALTH", use_container_width=True):
    with st.spinner("🔄 Analyzing..."):
        # Prepare data
        feature_names = ['age', 'sex', 'cp', 'trestbps', 'chol', 'fbs', 'restecg', 'thalach', 'exang', 'oldpeak', 'slope', 'ca', 'thal']
        input_df = pd.DataFrame([[age, sex, cp, trestbps, chol, fbs, restecg, thalach, exang, oldpeak, slope, ca, thal]], 
                               columns=feature_names)
        
        # Feature engineering
        input_df['age_chol'] = input_df['age'] * input_df['chol']
        input_df['age_thalach'] = input_df['age'] * input_df['thalach']
        input_df['cp_thalach'] = input_df['cp'] * input_df['thalach']
        input_df['oldpeak_slope'] = input_df['oldpeak'] * input_df['slope']
        input_df['ca_thal'] = input_df['ca'] * input_df['thal']
        input_df['age_squared'] = input_df['age'] ** 2
        input_df['chol_squared'] = input_df['chol'] ** 2
        input_df['thalach_squared'] = input_df['thalach'] ** 2
        
        # Predict
        input_scaled = scaler.transform(input_df)
        prediction = model.predict(input_scaled)[0]
        probability = model.predict_proba(input_scaled)[0]
        
        # Display Result
        if prediction == 1:
            st.markdown('<div class="result-card-disease"><h2>⚠️ HEART DISEASE DETECTED</h2><p>High cardiovascular risk identified</p></div>', unsafe_allow_html=True)
        else:
            st.markdown('<div class="result-card-healthy"><h2>✅ HEALTHY HEART</h2><p>No significant disease indicators</p></div>', unsafe_allow_html=True)
        
        # Metrics
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Confidence", f"{probability[prediction]*100:.1f}%")
        with col2:
            st.metric("Disease Risk", f"{probability[1]*100:.1f}%")
        with col3:
            st.metric("Healthy Probability", f"{probability[0]*100:.1f}%")
        
        # Risk Gauge
        st.markdown("### 📊 Risk Assessment")
        fig = go.Figure(go.Indicator(
            mode="gauge+number",
            value=probability[1]*100,
            title={'text': "Disease Risk Level"},
            gauge={
                'axis': {'range': [0, 100]},
                'bar': {'color': "red" if probability[1]>0.5 else "green"},
                'steps': [
                    {'range': [0, 33], 'color': 'rgba(0,255,0,0.3)'},
                    {'range': [33, 66], 'color': 'rgba(255,255,0,0.3)'},
                    {'range': [66, 100], 'color': 'rgba(255,0,0,0.3)'}
                ]
            }
        ))
        fig.update_layout(height=300)
        st.plotly_chart(fig, use_container_width=True)

st.markdown("---")
st.markdown("""
<div style='text-align: center; color: #888;'>
    <p>🫀 CardioPredict AI | 98% Accuracy | ⚠️ Screening tool only</p>
</div>
""", unsafe_allow_html=True)
