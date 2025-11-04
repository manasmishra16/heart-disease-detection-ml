# FastAPI Backend for Heart Disease Prediction
# Day 5: API + Demo

from fastapi import FastAPI, HTTPException, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
import tensorflow as tf
import numpy as np
import joblib
import pickle
from typing import List, Dict, Any
import json

# Initialize FastAPI app
app = FastAPI(
    title="Heart Disease Prediction API",
    description="API for predicting heart disease using ML/DL models",
    version="1.0.0"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load models and preprocessors
print("Loading models...")
try:
    # Load main MLP model
    mlp_model = tf.keras.models.load_model('../models/mlp_clinical.keras')
    print("✓ MLP model loaded")
    
    # Load Random Forest
    rf_model = joblib.load('../models/random_forest.pkl')
    print("✓ Random Forest model loaded")
    
    # Load StandardScaler
    import pandas as pd
    from sklearn.preprocessing import StandardScaler
    
    # Recreate scaler from training data
    cleaned_data = pd.read_csv('../results/cleaned_data.csv')
    X_train_data = cleaned_data.drop('target', axis=1)
    scaler = StandardScaler()
    scaler.fit(X_train_data)
    print("✓ Scaler initialized")
    
    print("✅ All models loaded successfully!")
    
except Exception as e:
    print(f"❌ Error loading models: {e}")
    mlp_model = None
    rf_model = None
    scaler = None

# Feature names (13 clinical features)
FEATURE_NAMES = [
    'age', 'sex', 'cp', 'trestbps', 'chol', 
    'fbs', 'restecg', 'thalach', 'exang', 
    'oldpeak', 'slope', 'ca', 'thal'
]

# Pydantic models for request/response
class PatientData(BaseModel):
    """Clinical features for a patient"""
    age: float = Field(..., ge=0, le=120, description="Age in years")
    sex: int = Field(..., ge=0, le=1, description="Sex (1=male, 0=female)")
    cp: int = Field(..., ge=0, le=3, description="Chest pain type (0-3)")
    trestbps: float = Field(..., ge=0, description="Resting blood pressure (mm Hg)")
    chol: float = Field(..., ge=0, description="Serum cholesterol (mg/dl)")
    fbs: int = Field(..., ge=0, le=1, description="Fasting blood sugar > 120 mg/dl (1=true, 0=false)")
    restecg: int = Field(..., ge=0, le=2, description="Resting ECG results (0-2)")
    thalach: float = Field(..., ge=0, description="Maximum heart rate achieved")
    exang: int = Field(..., ge=0, le=1, description="Exercise induced angina (1=yes, 0=no)")
    oldpeak: float = Field(..., ge=0, description="ST depression induced by exercise")
    slope: int = Field(..., ge=0, le=2, description="Slope of peak exercise ST segment (0-2)")
    ca: int = Field(..., ge=0, le=4, description="Number of major vessels colored by fluoroscopy (0-4)")
    thal: int = Field(..., ge=0, le=3, description="Thalassemia (0=normal, 1=fixed defect, 2=reversible defect, 3=unknown)")
    
    class Config:
        schema_extra = {
            "example": {
                "age": 63,
                "sex": 1,
                "cp": 3,
                "trestbps": 145,
                "chol": 233,
                "fbs": 1,
                "restecg": 0,
                "thalach": 150,
                "exang": 0,
                "oldpeak": 2.3,
                "slope": 0,
                "ca": 0,
                "thal": 1
            }
        }

class PredictionResponse(BaseModel):
    """Prediction result"""
    mlp_probability: float
    rf_probability: float
    ensemble_probability: float
    prediction: int
    risk_level: str
    confidence: float
    model_details: Dict[str, Any]

class BatchPredictionRequest(BaseModel):
    """Batch prediction request"""
    patients: List[PatientData]

# Helper functions
def preprocess_input(patient_data: PatientData) -> np.ndarray:
    """Preprocess patient data for model input"""
    # Convert to numpy array
    features = np.array([[
        patient_data.age,
        patient_data.sex,
        patient_data.cp,
        patient_data.trestbps,
        patient_data.chol,
        patient_data.fbs,
        patient_data.restecg,
        patient_data.thalach,
        patient_data.exang,
        patient_data.oldpeak,
        patient_data.slope,
        patient_data.ca,
        patient_data.thal
    ]])
    
    # Scale features
    features_scaled = scaler.transform(features)
    
    return features_scaled

def get_risk_level(probability: float) -> str:
    """Determine risk level from probability"""
    if probability < 0.3:
        return "Low Risk"
    elif probability < 0.7:
        return "Medium Risk"
    else:
        return "High Risk"

def calculate_confidence(probability: float) -> float:
    """Calculate confidence score"""
    # Distance from 0.5 (uncertainty point)
    confidence = abs(probability - 0.5) * 2
    return round(confidence, 4)

# API Endpoints
@app.get("/")
async def root():
    """Root endpoint"""
    return {
        "message": "Heart Disease Prediction API",
        "version": "1.0.0",
        "status": "running",
        "endpoints": {
            "/predict": "POST - Predict heart disease for single patient",
            "/predict/batch": "POST - Predict for multiple patients",
            "/health": "GET - Check API health",
            "/models/info": "GET - Get model information"
        }
    }

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    models_loaded = mlp_model is not None and rf_model is not None and scaler is not None
    
    return {
        "status": "healthy" if models_loaded else "unhealthy",
        "models_loaded": models_loaded,
        "models": {
            "mlp": mlp_model is not None,
            "random_forest": rf_model is not None,
            "scaler": scaler is not None
        }
    }

@app.get("/models/info")
async def models_info():
    """Get information about loaded models"""
    if mlp_model is None or rf_model is None:
        raise HTTPException(status_code=503, detail="Models not loaded")
    
    return {
        "mlp": {
            "type": "Enhanced MLP with BatchNormalization",
            "input_shape": mlp_model.input_shape,
            "output_shape": mlp_model.output_shape,
            "total_params": int(mlp_model.count_params()),
            "layers": len(mlp_model.layers)
        },
        "random_forest": {
            "type": "Random Forest Classifier",
            "n_estimators": rf_model.n_estimators,
            "max_depth": rf_model.max_depth,
            "n_features": rf_model.n_features_in_
        },
        "ensemble": {
            "type": "Probability Averaging",
            "weights": "50% MLP + 50% Random Forest",
            "performance": {
                "accuracy": 0.8852,
                "auc": 0.9643,
                "recall": 0.9643,
                "f1_score": 0.8852
            }
        },
        "features": FEATURE_NAMES
    }

@app.post("/predict", response_model=PredictionResponse)
async def predict(patient: PatientData):
    """
    Predict heart disease for a single patient
    
    Returns:
    - Individual model probabilities (MLP, Random Forest)
    - Ensemble probability (average of both models)
    - Binary prediction (0=No Disease, 1=Disease)
    - Risk level (Low/Medium/High)
    - Confidence score
    """
    if mlp_model is None or rf_model is None or scaler is None:
        raise HTTPException(status_code=503, detail="Models not loaded")
    
    try:
        # Preprocess input
        features_scaled = preprocess_input(patient)
        
        # Get predictions from both models
        mlp_prob = float(mlp_model.predict(features_scaled, verbose=0)[0, 0])
        rf_prob = float(rf_model.predict_proba(features_scaled)[0, 1])
        
        # Ensemble prediction (average)
        ensemble_prob = (mlp_prob + rf_prob) / 2
        
        # Binary prediction (threshold 0.5)
        prediction = int(ensemble_prob > 0.5)
        
        # Risk level
        risk_level = get_risk_level(ensemble_prob)
        
        # Confidence
        confidence = calculate_confidence(ensemble_prob)
        
        return PredictionResponse(
            mlp_probability=round(mlp_prob, 4),
            rf_probability=round(rf_prob, 4),
            ensemble_probability=round(ensemble_prob, 4),
            prediction=prediction,
            risk_level=risk_level,
            confidence=confidence,
            model_details={
                "mlp_contribution": 0.5,
                "rf_contribution": 0.5,
                "threshold": 0.5,
                "interpretation": "Disease" if prediction == 1 else "No Disease"
            }
        )
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction error: {str(e)}")

@app.post("/predict/batch")
async def predict_batch(request: BatchPredictionRequest):
    """
    Predict heart disease for multiple patients
    
    Returns list of predictions for each patient
    """
    if mlp_model is None or rf_model is None or scaler is None:
        raise HTTPException(status_code=503, detail="Models not loaded")
    
    try:
        predictions = []
        
        for idx, patient in enumerate(request.patients):
            # Preprocess input
            features_scaled = preprocess_input(patient)
            
            # Get predictions
            mlp_prob = float(mlp_model.predict(features_scaled, verbose=0)[0, 0])
            rf_prob = float(rf_model.predict_proba(features_scaled)[0, 1])
            ensemble_prob = (mlp_prob + rf_prob) / 2
            prediction = int(ensemble_prob > 0.5)
            
            predictions.append({
                "patient_index": idx,
                "ensemble_probability": round(ensemble_prob, 4),
                "prediction": prediction,
                "risk_level": get_risk_level(ensemble_prob),
                "confidence": calculate_confidence(ensemble_prob)
            })
        
        return {
            "total_patients": len(request.patients),
            "predictions": predictions,
            "summary": {
                "disease_cases": sum(1 for p in predictions if p["prediction"] == 1),
                "no_disease_cases": sum(1 for p in predictions if p["prediction"] == 0)
            }
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Batch prediction error: {str(e)}")

if __name__ == "__main__":
    import uvicorn
    print("=" * 70)
    print("🚀 Starting Heart Disease Prediction API")
    print("=" * 70)
    print("API Documentation: http://localhost:8000/docs")
    print("Health Check: http://localhost:8000/health")
    print("=" * 70)
    uvicorn.run(app, host="0.0.0.0", port=8000)
