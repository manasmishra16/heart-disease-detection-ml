# Complete Reproduction Guide
## Heart Disease Detection - ML/DL Pipeline

**Last Updated:** October 28, 2025  
**Project Duration:** 5 Days  
**Difficulty:** Intermediate to Advanced

---

## 📋 Table of Contents

1. [Prerequisites](#prerequisites)
2. [Environment Setup](#environment-setup)
3. [Day 1: Setup & Data Loading](#day-1-setup--data-loading)
4. [Day 2: EDA & Preprocessing](#day-2-eda--preprocessing)
5. [Day 3: Baseline Models](#day-3-baseline-models)
6. [Day 4: Deep Learning Models](#day-4-deep-learning-models)
7. [Day 5: API & Demo](#day-5-api--demo)
8. [Testing](#testing)
9. [Troubleshooting](#troubleshooting)
10. [Project Structure](#project-structure)

---

## Prerequisites

### Required Software
- **Python**: 3.11.x recommended (3.9+ compatible)
- **Git**: For version control (optional)
- **Text Editor/IDE**: VS Code, PyCharm, or Jupyter Lab
- **Operating System**: Windows 11, macOS, or Linux

### Required Skills
- Python programming (intermediate)
- Basic machine learning concepts
- Understanding of neural networks
- Familiarity with pandas, numpy, scikit-learn
- Command line basics

### Hardware Requirements
- **RAM**: 8GB minimum (16GB recommended)
- **Storage**: 5GB free space
- **CPU**: Multi-core processor recommended
- **GPU**: Optional (CUDA-compatible for faster training)

---

## Environment Setup

### Step 1: Create Project Directory

```powershell
# Windows PowerShell
mkdir D:\Projects\MiniProject
cd D:\Projects\MiniProject
```

```bash
# Linux/macOS
mkdir -p ~/Projects/MiniProject
cd ~/Projects/MiniProject
```

### Step 2: Create Virtual Environment

```powershell
# Create virtual environment
python -m venv venv

# Activate (Windows PowerShell)
.\venv\Scripts\Activate.ps1

# Activate (Windows CMD)
venv\Scripts\activate.bat

# Activate (Linux/macOS)
source venv/bin/activate
```

**Verify activation:**
```powershell
# You should see (venv) prefix in terminal
(venv) PS D:\Projects\MiniProject>
```

### Step 3: Install Base Dependencies

Create `requirements.txt`:
```txt
# Core Scientific Libraries
numpy>=1.24.0
pandas>=2.0.0
scipy>=1.11.0

# Machine Learning
scikit-learn>=1.3.0
xgboost>=2.0.0
joblib>=1.3.0

# Deep Learning
tensorflow>=2.17.0
keras>=3.10.0

# Data Visualization
matplotlib>=3.8.0
seaborn>=0.13.0
plotly>=5.17.0

# Medical Data Processing
wfdb>=4.1.0

# Testing
pytest>=8.0.0
pytest-cov>=4.1.0

# Utilities
tqdm>=4.66.0
python-dotenv>=1.0.0
```

Install dependencies:
```powershell
pip install -r requirements.txt
```

### Step 4: Create Project Structure

```powershell
# Create directories
mkdir data, datasets, models, results, scripts, tests, docs, app

# Create subdirectories
mkdir datasets\cleveland, datasets\mit-bih
```

**Expected structure:**
```
MiniProject/
├── venv/                    # Virtual environment
├── data/                    # Raw data
├── datasets/               
│   ├── cleveland/          # Cleveland heart disease data
│   └── mit-bih/            # ECG data
├── models/                 # Saved models
├── results/                # Output files
├── scripts/                # Utility scripts
├── tests/                  # Test files
├── docs/                   # Documentation
├── app/                    # Deployment (API + UI)
├── requirements.txt        # Dependencies
└── heart_disease_detection.ipynb  # Main notebook
```

---

## Day 1: Setup & Data Loading

**Objective:** Set up environment and load datasets

### Step 1: Download Datasets

#### Cleveland Heart Disease Dataset
```python
# Download from UCI Repository
import urllib.request
import pandas as pd

url = "https://archive.ics.uci.edu/ml/machine-learning-databases/heart-disease/processed.cleveland.data"
urllib.request.urlretrieve(url, "datasets/cleveland/processed.cleveland.data")
```

**Column names:**
```python
columns = ['age', 'sex', 'cp', 'trestbps', 'chol', 'fbs', 'restecg', 
           'thalach', 'exang', 'oldpeak', 'slope', 'ca', 'thal', 'target']
```

#### MIT-BIH Arrhythmia Database (Optional)
```python
import wfdb

# Download sample ECG record
record = wfdb.rdrecord('100', pn_dir='mitdb', sampfrom=0, sampto=1000)
wfdb.plot_wfdb(record=record, title='ECG Signal')
```

### Step 2: Data Loading & Initial Inspection

Create `heart_disease_detection.ipynb`:

```python
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Load data
df = pd.read_csv('datasets/cleveland/processed.cleveland.data', 
                 names=columns, na_values='?')

# Initial inspection
print(f"Dataset Shape: {df.shape}")
print(f"Missing Values:\n{df.isnull().sum()}")
print(f"Data Types:\n{df.dtypes}")

# Save initial data
df.to_csv('data/raw_data.csv', index=False)
```

### Step 3: Verify Day 1 Completion

```powershell
python tests/test_day1.py
```

**Expected output:**
```
✅ All Day 1 tests passed
- Dataset loaded successfully
- Required columns present
- Data saved correctly
```

---

## Day 2: EDA & Preprocessing

**Objective:** Explore data and prepare for modeling

### Step 1: Exploratory Data Analysis

```python
# Target distribution
print(df['target'].value_counts())

# Convert to binary (0: No disease, 1-4: Disease)
df['target'] = (df['target'] > 0).astype(int)

# Statistical summary
print(df.describe())

# Visualizations
fig, axes = plt.subplots(2, 2, figsize=(15, 12))

# Age distribution
axes[0, 0].hist(df['age'], bins=20, edgecolor='black')
axes[0, 0].set_title('Age Distribution')

# Target distribution
df['target'].value_counts().plot(kind='bar', ax=axes[0, 1])
axes[0, 1].set_title('Disease vs No Disease')

# Correlation heatmap
sns.heatmap(df.corr(), annot=True, fmt='.2f', ax=axes[1, 0])

# Age vs Max Heart Rate
axes[1, 1].scatter(df['age'], df['thalach'], c=df['target'], alpha=0.5)
axes[1, 1].set_xlabel('Age')
axes[1, 1].set_ylabel('Max Heart Rate')

plt.tight_layout()
plt.savefig('results/eda_visualizations.png', dpi=300)
```

### Step 2: Data Preprocessing

```python
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

# Handle missing values
df = df.dropna()  # or use imputation

# Separate features and target
X = df.drop('target', axis=1)
y = df['target']

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# Feature scaling
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Save preprocessed data
pd.DataFrame(X_train_scaled, columns=X.columns).to_csv('results/X_train.csv', index=False)
pd.DataFrame(X_test_scaled, columns=X.columns).to_csv('results/X_test.csv', index=False)
pd.Series(y_train).to_csv('results/y_train.csv', index=False)
pd.Series(y_test).to_csv('results/y_test.csv', index=False)
```

### Step 3: Feature Engineering (Optional)

```python
# Create interaction features
df['age_chol'] = df['age'] * df['chol']
df['trestbps_chol'] = df['trestbps'] * df['chol']

# Polynomial features
from sklearn.preprocessing import PolynomialFeatures
poly = PolynomialFeatures(degree=2, include_bias=False)
X_poly = poly.fit_transform(X)
```

### Step 4: Verify Day 2 Completion

```powershell
python tests/test_day2.py
```

---

## Day 3: Baseline Models

**Objective:** Build and evaluate baseline ML models

### Step 1: Logistic Regression

```python
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score

# Train model
lr_model = LogisticRegression(random_state=42, max_iter=1000)
lr_model.fit(X_train_scaled, y_train)

# Predictions
y_pred_lr = lr_model.predict(X_test_scaled)
y_proba_lr = lr_model.predict_proba(X_test_scaled)[:, 1]

# Evaluation
print("Logistic Regression Results:")
print(classification_report(y_test, y_pred_lr))
print(f"AUC: {roc_auc_score(y_test, y_proba_lr):.4f}")
```

### Step 2: Random Forest

```python
from sklearn.ensemble import RandomForestClassifier

# Train model
rf_model = RandomForestClassifier(
    n_estimators=100,
    max_depth=None,
    min_samples_split=2,
    class_weight='balanced',
    random_state=42
)
rf_model.fit(X_train_scaled, y_train)

# Predictions
y_pred_rf = rf_model.predict(X_test_scaled)
y_proba_rf = rf_model.predict_proba(X_test_scaled)[:, 1]

# Evaluation
print("Random Forest Results:")
print(classification_report(y_test, y_pred_rf))
print(f"AUC: {roc_auc_score(y_test, y_proba_rf):.4f}")

# Feature importance
feature_importance = pd.DataFrame({
    'feature': X.columns,
    'importance': rf_model.feature_importances_
}).sort_values('importance', ascending=False)

print(feature_importance)
```

### Step 3: XGBoost

```python
import xgboost as xgb

# Train model
xgb_model = xgb.XGBClassifier(
    n_estimators=100,
    learning_rate=0.1,
    max_depth=5,
    random_state=42
)
xgb_model.fit(X_train_scaled, y_train)

# Predictions
y_pred_xgb = xgb_model.predict(X_test_scaled)
y_proba_xgb = xgb_model.predict_proba(X_test_scaled)[:, 1]

# Evaluation
print("XGBoost Results:")
print(classification_report(y_test, y_pred_xgb))
print(f"AUC: {roc_auc_score(y_test, y_proba_xgb):.4f}")
```

### Step 4: Model Comparison

```python
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve

# ROC Curves
plt.figure(figsize=(10, 8))

models = {
    'Logistic Regression': y_proba_lr,
    'Random Forest': y_proba_rf,
    'XGBoost': y_proba_xgb
}

for name, y_proba in models.items():
    fpr, tpr, _ = roc_curve(y_test, y_proba)
    auc = roc_auc_score(y_test, y_proba)
    plt.plot(fpr, tpr, label=f'{name} (AUC = {auc:.3f})')

plt.plot([0, 1], [0, 1], 'k--', label='Random')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curves - Baseline Models')
plt.legend()
plt.savefig('results/baseline_roc_curves.png', dpi=300)
```

### Step 5: Save Best Model

```python
import joblib

# Save Random Forest (usually best performer)
joblib.dump(rf_model, 'models/random_forest.pkl')
joblib.dump(scaler, 'models/scaler.pkl')

print("✅ Baseline models saved successfully")
```

### Step 6: Verify Day 3 Completion

```powershell
python tests/test_day3.py
```

---

## Day 4: Deep Learning Models

**Objective:** Build and train neural networks

### Step 1: Simple MLP

```python
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

# Build model
def create_mlp():
    model = keras.Sequential([
        layers.Dense(64, activation='relu', input_shape=(X_train_scaled.shape[1],)),
        layers.Dropout(0.3),
        layers.Dense(32, activation='relu'),
        layers.Dropout(0.2),
        layers.Dense(1, activation='sigmoid')
    ])
    
    model.compile(
        optimizer='adam',
        loss='binary_crossentropy',
        metrics=['accuracy', 'AUC']
    )
    
    return model

# Train model
mlp_model = create_mlp()
history = mlp_model.fit(
    X_train_scaled, y_train,
    validation_split=0.2,
    epochs=100,
    batch_size=32,
    callbacks=[
        keras.callbacks.EarlyStopping(patience=10, restore_best_weights=True)
    ],
    verbose=1
)

# Evaluate
y_pred_mlp = (mlp_model.predict(X_test_scaled) > 0.5).astype(int)
print(classification_report(y_test, y_pred_mlp))
```

### Step 2: Enhanced MLP with BatchNormalization

```python
def create_enhanced_mlp():
    model = keras.Sequential([
        layers.Dense(128, input_shape=(X_train_scaled.shape[1],)),
        layers.BatchNormalization(),
        layers.Activation('relu'),
        layers.Dropout(0.3),
        
        layers.Dense(64),
        layers.BatchNormalization(),
        layers.Activation('relu'),
        layers.Dropout(0.3),
        
        layers.Dense(32),
        layers.BatchNormalization(),
        layers.Activation('relu'),
        layers.Dropout(0.2),
        
        layers.Dense(16),
        layers.BatchNormalization(),
        layers.Activation('relu'),
        layers.Dropout(0.2),
        
        layers.Dense(1, activation='sigmoid')
    ])
    
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=0.001),
        loss='binary_crossentropy',
        metrics=['accuracy', 
                 keras.metrics.Precision(name='precision'),
                 keras.metrics.Recall(name='recall'),
                 keras.metrics.AUC(name='auc')]
    )
    
    return model

# Train enhanced model
enhanced_mlp = create_enhanced_mlp()
history = enhanced_mlp.fit(
    X_train_scaled, y_train,
    validation_split=0.2,
    epochs=150,
    batch_size=32,
    callbacks=[
        keras.callbacks.EarlyStopping(patience=15, restore_best_weights=True),
        keras.callbacks.ReduceLROnPlateau(patience=5, factor=0.5)
    ]
)

# Save model
enhanced_mlp.save('models/mlp_clinical.keras')
enhanced_mlp.save('models/model.h5')  # Main deliverable
```

### Step 3: Transfer Learning (Optional - ECG Images)

```python
# Generate spectrograms from ECG data
from scipy import signal
import matplotlib.pyplot as plt

def create_spectrogram(ecg_signal, save_path):
    f, t, Sxx = signal.spectrogram(ecg_signal, fs=360)
    plt.pcolormesh(t, f, Sxx, shading='gouraud')
    plt.ylabel('Frequency [Hz]')
    plt.xlabel('Time [sec]')
    plt.axis('off')
    plt.savefig(save_path, bbox_inches='tight', pad_inches=0)
    plt.close()

# Use pre-trained model (EfficientNetB0)
from tensorflow.keras.applications import EfficientNetB0

base_model = EfficientNetB0(
    include_top=False,
    weights='imagenet',
    input_shape=(224, 224, 3)
)

# Freeze base model
base_model.trainable = False

# Add custom head
model = keras.Sequential([
    base_model,
    layers.GlobalAveragePooling2D(),
    layers.Dense(256, activation='relu'),
    layers.Dropout(0.5),
    layers.Dense(1, activation='sigmoid')
])

model.compile(
    optimizer='adam',
    loss='binary_crossentropy',
    metrics=['accuracy', 'AUC']
)

# Train on spectrogram images
# (Assuming images are prepared in train/val directories)
```

### Step 4: Ensemble Model

```python
# Combine MLP and Random Forest
def ensemble_prediction(X_test):
    # MLP prediction
    mlp_prob = enhanced_mlp.predict(X_test).flatten()
    
    # Random Forest prediction
    rf_prob = rf_model.predict_proba(X_test)[:, 1]
    
    # Average probabilities
    ensemble_prob = (mlp_prob + rf_prob) / 2
    ensemble_pred = (ensemble_prob >= 0.5).astype(int)
    
    return ensemble_pred, ensemble_prob

# Get predictions
y_pred_ensemble, y_proba_ensemble = ensemble_prediction(X_test_scaled)

# Evaluate
print("Ensemble Results:")
print(classification_report(y_test, y_pred_ensemble))
print(f"AUC: {roc_auc_score(y_test, y_proba_ensemble):.4f}")

# Save ensemble predictions
import pickle
ensemble_results = {
    'predictions': y_pred_ensemble,
    'probabilities': y_proba_ensemble,
    'accuracy': accuracy_score(y_test, y_pred_ensemble),
    'precision': precision_score(y_test, y_pred_ensemble),
    'recall': recall_score(y_test, y_pred_ensemble),
    'f1': f1_score(y_test, y_pred_ensemble),
    'auc': roc_auc_score(y_test, y_proba_ensemble)
}

with open('models/ensemble_predictions.pkl', 'wb') as f:
    pickle.dump(ensemble_results, f)
```

### Step 5: Create Validation Report

```python
# Generate comprehensive validation report
with open('validation_report.md', 'w') as f:
    f.write("# Validation Report\n\n")
    f.write(f"## Model Performance\n\n")
    f.write(f"### Ensemble Model\n")
    f.write(f"- Accuracy: {ensemble_results['accuracy']:.4f}\n")
    f.write(f"- Precision: {ensemble_results['precision']:.4f}\n")
    f.write(f"- Recall: {ensemble_results['recall']:.4f}\n")
    f.write(f"- F1-Score: {ensemble_results['f1']:.4f}\n")
    f.write(f"- AUC: {ensemble_results['auc']:.4f}\n")
```

### Step 6: Verify Day 4 Completion

```powershell
python tests/test_day4.py
```

---

## Day 5: API & Demo

**Objective:** Deploy models with API and web interface

### Step 1: Create FastAPI Backend

Create `app/main.py`:

```python
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
import tensorflow as tf
import joblib
import pandas as pd
from sklearn.preprocessing import StandardScaler

# Initialize FastAPI
app = FastAPI(title="Heart Disease Prediction API")

# Add CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"]
)

# Load models
mlp_model = tf.keras.models.load_model('../models/mlp_clinical.keras')
rf_model = joblib.load('../models/random_forest.pkl')

# Initialize scaler
cleaned_data = pd.read_csv('../results/cleaned_data.csv')
X_train_data = cleaned_data.drop('target', axis=1)
scaler = StandardScaler()
scaler.fit(X_train_data)

# Request model
class PatientData(BaseModel):
    age: int = Field(..., ge=29, le=77)
    sex: int = Field(..., ge=0, le=1)
    cp: int = Field(..., ge=0, le=3)
    trestbps: int = Field(..., ge=94, le=200)
    chol: int = Field(..., ge=126, le=564)
    fbs: int = Field(..., ge=0, le=1)
    restecg: int = Field(..., ge=0, le=2)
    thalach: int = Field(..., ge=71, le=202)
    exang: int = Field(..., ge=0, le=1)
    oldpeak: float = Field(..., ge=0.0, le=6.2)
    slope: int = Field(..., ge=0, le=2)
    ca: int = Field(..., ge=0, le=4)
    thal: int = Field(..., ge=0, le=3)

# Endpoints
@app.get("/")
def root():
    return {"message": "Heart Disease Prediction API", "version": "1.0.0"}

@app.get("/health")
def health_check():
    return {"status": "healthy", "models_loaded": True}

@app.post("/predict")
def predict(data: PatientData):
    # Prepare features
    features = [[
        data.age, data.sex, data.cp, data.trestbps, data.chol, data.fbs,
        data.restecg, data.thalach, data.exang, data.oldpeak, data.slope,
        data.ca, data.thal
    ]]
    
    # Scale features
    features_scaled = scaler.transform(features)
    
    # Get predictions
    mlp_prob = float(mlp_model.predict(features_scaled)[0][0])
    rf_prob = float(rf_model.predict_proba(features_scaled)[0][1])
    ensemble_prob = (mlp_prob + rf_prob) / 2
    
    # Risk level
    if ensemble_prob >= 0.7:
        risk_level = "High Risk"
    elif ensemble_prob >= 0.3:
        risk_level = "Medium Risk"
    else:
        risk_level = "Low Risk"
    
    return {
        "mlp_probability": mlp_prob,
        "rf_probability": rf_prob,
        "ensemble_probability": ensemble_prob,
        "prediction": int(ensemble_prob >= 0.5),
        "risk_level": risk_level,
        "confidence": abs(ensemble_prob - 0.5) * 2
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
```

### Step 2: Create Streamlit Demo

Create `app/demo.py`:

```python
import streamlit as st
import requests
import plotly.graph_objects as go

st.set_page_config(page_title="Heart Disease Prediction", page_icon="❤️", layout="wide")

st.title("❤️ Heart Disease Prediction System")

# API URL
API_URL = "http://localhost:8000"

# Check API health
try:
    response = requests.get(f"{API_URL}/health", timeout=2)
    api_healthy = response.status_code == 200
except:
    api_healthy = False

if not api_healthy:
    st.error("⚠️ API Server is not running!")
    st.stop()

st.success("✅ API Server Connected")

# Sidebar inputs
with st.sidebar:
    st.header("Patient Information")
    
    age = st.slider("Age", 29, 77, 54)
    sex = st.radio("Sex", [0, 1], format_func=lambda x: "Female" if x == 0 else "Male")
    cp = st.slider("Chest Pain Type", 0, 3, 0)
    trestbps = st.slider("Resting Blood Pressure", 94, 200, 130)
    chol = st.slider("Cholesterol", 126, 564, 246)
    fbs = st.radio("Fasting Blood Sugar > 120 mg/dl", [0, 1])
    restecg = st.slider("Resting ECG", 0, 2, 1)
    thalach = st.slider("Max Heart Rate", 71, 202, 150)
    exang = st.radio("Exercise Induced Angina", [0, 1])
    oldpeak = st.slider("ST Depression", 0.0, 6.2, 1.0, 0.1)
    slope = st.slider("Slope of Peak Exercise ST", 0, 2, 2)
    ca = st.slider("Number of Major Vessels", 0, 4, 0)
    thal = st.slider("Thalassemia", 0, 3, 2)
    
    predict_button = st.button("Predict Risk", type="primary")

# Main content
if predict_button:
    patient_data = {
        "age": age, "sex": sex, "cp": cp, "trestbps": trestbps,
        "chol": chol, "fbs": fbs, "restecg": restecg, "thalach": thalach,
        "exang": exang, "oldpeak": oldpeak, "slope": slope, "ca": ca, "thal": thal
    }
    
    response = requests.post(f"{API_URL}/predict", json=patient_data)
    result = response.json()
    
    # Display results
    st.subheader("Prediction Results")
    
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Probability", f"{result['ensemble_probability']*100:.1f}%")
    col2.metric("Prediction", "Disease" if result['prediction'] == 1 else "No Disease")
    col3.metric("Risk Level", result['risk_level'])
    col4.metric("Confidence", f"{result['confidence']*100:.1f}%")
    
    # Gauge chart
    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=result['ensemble_probability'] * 100,
        title={'text': "Disease Probability"},
        gauge={
            'axis': {'range': [0, 100]},
            'bar': {'color': "darkblue"},
            'steps': [
                {'range': [0, 30], 'color': "#2ecc71"},
                {'range': [30, 70], 'color': "#f39c12"},
                {'range': [70, 100], 'color': "#e74c3c"}
            ]
        }
    ))
    st.plotly_chart(fig)
else:
    st.info("👈 Enter patient information and click 'Predict Risk'")
```

### Step 3: Install API Dependencies

Create `app/requirements.txt`:

```txt
# API Framework
fastapi>=0.104.0
uvicorn[standard]>=0.24.0
pydantic>=2.0.0
python-multipart>=0.0.6

# Demo UI
streamlit>=1.28.0
plotly>=5.17.0
requests>=2.31.0

# Core ML (reuse from main requirements)
tensorflow>=2.17.0
scikit-learn>=1.3.0
pandas>=2.0.0
numpy>=1.24.0
joblib>=1.3.0
```

Install:
```powershell
pip install -r app/requirements.txt
```

### Step 4: Run API and Demo

**Terminal 1 - Start API:**
```powershell
cd app
python main.py
```

**Terminal 2 - Start Demo:**
```powershell
cd app
streamlit run demo.py
```

Access:
- API Documentation: http://localhost:8000/docs
- Demo UI: http://localhost:8501

### Step 5: Verify Day 5 Completion

```powershell
python tests/test_day5.py
```

---

## Testing

### Run All Tests

```powershell
# Run master test script
python run_all_tests.py

# Or run individually
python tests/test_day1.py
python tests/test_day2.py
python tests/test_day3.py
python tests/test_day4.py
python tests/test_day5.py
```

### Test Coverage

```powershell
pytest tests/ --cov=. --cov-report=html
```

View coverage report:
```powershell
start htmlcov/index.html  # Windows
open htmlcov/index.html   # macOS
```

---

## Troubleshooting

### Common Issues

#### 1. TensorFlow DLL Error (Windows)
**Error:** `DLL load failed while importing _pywrap_tensorflow_internal`

**Solutions:**
- Install Microsoft Visual C++ Redistributable
- Use conda instead of pip: `conda install tensorflow`
- Run from Jupyter notebook instead of terminal
- Use WSL2 (Windows Subsystem for Linux)

#### 2. Memory Error
**Error:** `MemoryError` during model training

**Solutions:**
- Reduce batch size: `batch_size=16`
- Use data generators for large datasets
- Close other applications
- Upgrade RAM if possible

#### 3. CUDA Not Available
**Issue:** GPU not detected

**Solutions:**
```python
import tensorflow as tf
print("GPUs Available:", tf.config.list_physical_devices('GPU'))

# Force CPU
tf.config.set_visible_devices([], 'GPU')
```

#### 4. Module Not Found
**Error:** `ModuleNotFoundError: No module named 'xxx'`

**Solution:**
```powershell
# Ensure virtual environment is activated
.\venv\Scripts\Activate.ps1

# Reinstall requirements
pip install -r requirements.txt
```

#### 5. Streamlit Connection Error
**Issue:** Demo can't connect to API

**Solutions:**
- Verify API is running: `curl http://localhost:8000/health`
- Check firewall settings
- Use `127.0.0.1` instead of `localhost`
- Run both in same terminal with virtual environment activated

---

## Project Structure

### Final Directory Layout

```
MiniProject/
│
├── venv/                           # Virtual environment
│
├── data/                           # Raw data
│   ├── raw_data.csv
│   └── preprocessed_data.csv
│
├── datasets/                       # Downloaded datasets
│   ├── cleveland/
│   │   └── processed.cleveland.data
│   └── mit-bih/
│       ├── 100.dat
│       └── 100.hea
│
├── models/                         # Saved models
│   ├── model.h5                   # Main deliverable (MLP)
│   ├── mlp_clinical.keras         # Enhanced MLP
│   ├── random_forest.pkl          # Random Forest
│   ├── scaler.pkl                 # StandardScaler
│   ├── ensemble_predictions.pkl   # Ensemble results
│   └── transfer_learning/
│       └── best_model.keras
│
├── results/                        # Output files
│   ├── cleaned_data.csv
│   ├── X_train.csv
│   ├── X_test.csv
│   ├── y_train.csv
│   ├── y_test.csv
│   ├── eda_visualizations.png
│   ├── baseline_roc_curves.png
│   └── confusion_matrices.png
│
├── scripts/                        # Utility scripts
│   ├── download_data.py
│   ├── preprocess.py
│   └── train_models.py
│
├── tests/                          # Test suite
│   ├── test_day1.py               # Day 1 tests
│   ├── test_day2.py               # Day 2 tests
│   ├── test_day3.py               # Day 3 tests
│   ├── test_day4.py               # Day 4 tests
│   └── test_day5.py               # Day 5 tests
│
├── app/                            # Deployment
│   ├── main.py                    # FastAPI backend
│   ├── demo.py                    # Streamlit UI
│   ├── requirements.txt           # API dependencies
│   └── README.md                  # Deployment guide
│
├── docs/                           # Documentation
│   ├── data_dictionary.md
│   ├── model_architecture.md
│   └── api_documentation.md
│
├── heart_disease_detection.ipynb  # Main Jupyter notebook
├── requirements.txt               # Project dependencies
├── run_all_tests.py              # Master test runner
├── REPRODUCTION_GUIDE.md         # This file
├── README.md                     # Project overview
├── validation_report.md          # Model validation
├── DAY1_SUMMARY.md              # Day 1 completion log
├── DAY2_SUMMARY.md              # Day 2 completion log
├── DAY3_SUMMARY.md              # Day 3 completion log
├── DAY4_SUMMARY.md              # Day 4 completion log
├── DAY5_SUMMARY.md              # Day 5 completion log
└── .gitignore                   # Git ignore file
```

---

## Performance Benchmarks

### Expected Results

| Model | Accuracy | Precision | Recall | F1-Score | AUC |
|-------|----------|-----------|--------|----------|-----|
| Logistic Regression | ~82% | ~80% | ~85% | ~82% | ~88% |
| Random Forest | ~90% | ~88% | ~96% | ~92% | ~95% |
| XGBoost | ~87% | ~85% | ~92% | ~88% | ~93% |
| Simple MLP | ~84% | ~82% | ~89% | ~85% | ~91% |
| Enhanced MLP | ~85% | ~83% | ~100% | ~91% | ~96% |
| **Ensemble** | **~89%** | **~82%** | **~96%** | **~89%** | **~96%** |

### Training Time

- Logistic Regression: <1 second
- Random Forest: ~5 seconds
- XGBoost: ~10 seconds
- Simple MLP: ~30 seconds
- Enhanced MLP: ~2 minutes
- Transfer Learning: ~10-15 minutes

### Hardware Used

- CPU: Intel i7 / AMD Ryzen 7
- RAM: 16 GB
- GPU: Optional (NVIDIA GTX 1060 or better)
- OS: Windows 11 / Ubuntu 22.04

---

## Next Steps

### Advanced Features (Optional)

1. **Explainability:**
   - SHAP values for feature importance
   - LIME for local interpretability
   - Grad-CAM for image-based models

2. **Model Optimization:**
   - Hyperparameter tuning (GridSearchCV, Optuna)
   - Model quantization for deployment
   - Pruning for smaller model size

3. **Additional Datasets:**
   - Incorporate more heart disease datasets
   - Multi-center validation
   - External validation datasets

4. **Deployment:**
   - Docker containerization
   - Cloud deployment (AWS, GCP, Azure)
   - CI/CD pipeline
   - Monitoring and logging

5. **UI Enhancements:**
   - Patient history tracking
   - PDF report generation
   - Multi-language support
   - Mobile-responsive design

---

## References

### Datasets
- UCI Heart Disease Dataset: https://archive.ics.uci.edu/ml/datasets/heart+disease
- MIT-BIH Arrhythmia Database: https://physionet.org/content/mitdb/1.0.0/

### Libraries
- TensorFlow: https://www.tensorflow.org/
- scikit-learn: https://scikit-learn.org/
- FastAPI: https://fastapi.tiangolo.com/
- Streamlit: https://streamlit.io/

### Papers
- "Heart Disease Prediction using Machine Learning" - Various authors
- "Deep Learning for Healthcare" - Rajkomar et al., 2019
- "Ensemble Methods in Machine Learning" - Dietterich, 2000

---

## Support & Contribution

### Getting Help
- Check troubleshooting section above
- Review test outputs for specific errors
- Check API logs: `http://localhost:8000/docs`
- Review validation report

### Contributing
- Fork the repository
- Create feature branch
- Add tests for new features
- Submit pull request

---

## License

This project is for educational and research purposes.  
Always consult healthcare professionals for medical decisions.

---

**Last Updated:** October 28, 2025  
**Version:** 1.0.0  
**Author:** AI Assistant  
**Project Status:** Complete ✅

---

## Quick Start Checklist

- [ ] Create virtual environment
- [ ] Install dependencies
- [ ] Download datasets
- [ ] Run Day 1 (Setup)
- [ ] Run Day 2 (EDA & Preprocessing)
- [ ] Run Day 3 (Baseline Models)
- [ ] Run Day 4 (Deep Learning)
- [ ] Run Day 5 (API & Demo)
- [ ] Execute all tests
- [ ] Verify all deliverables
- [ ] Deploy application

**Estimated Time:** 10-15 hours total (2-3 hours per day)

---

*For detailed day-by-day summaries, refer to individual DAY[1-5]_SUMMARY.md files.*
