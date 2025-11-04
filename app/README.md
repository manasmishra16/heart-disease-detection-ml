# Heart Disease Prediction - API & Demo
**Day 5 Deliverable: Deployment Package**

Complete API server and interactive web demo for heart disease prediction using ensemble ML/DL models.

---

## 🚀 Quick Start

### 1. Install Dependencies

```powershell
# Install Python packages
pip install -r requirements.txt
```

### 2. Start API Server

```powershell
# Navigate to app directory
cd app

# Start FastAPI server
python main.py

# API will run on: http://localhost:8000
# API Documentation: http://localhost:8000/docs
```

### 3. Launch Demo UI

```powershell
# In a new terminal, navigate to app directory
cd app

# Start Streamlit demo
streamlit run demo.py

# Demo will open in browser: http://localhost:8501
```

---

## 📦 Project Structure

```
app/
├── main.py              # FastAPI backend server
├── demo.py              # Streamlit interactive UI
├── requirements.txt     # Python dependencies
└── README.md            # This file

../models/               # Pre-trained models
├── mlp_clinical.keras   # Enhanced MLP model
├── random_forest.pkl    # Random Forest model
└── model.h5             # Main model (copy of MLP)

../results/              # Preprocessing data
└── cleaned_data.csv     # Training data for scaler
```

---

## 🔌 API Endpoints

### Health Check
```bash
GET /health
```

**Response:**
```json
{
  "status": "healthy",
  "models_loaded": true,
  "models": {
    "mlp": true,
    "random_forest": true,
    "scaler": true
  }
}
```

### Single Prediction
```bash
POST /predict
```

**Request Body:**
```json
{
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
```

**Response:**
```json
{
  "mlp_probability": 0.8234,
  "rf_probability": 0.7856,
  "ensemble_probability": 0.8045,
  "prediction": 1,
  "risk_level": "High Risk",
  "confidence": 0.6090,
  "model_details": {
    "mlp_contribution": 0.5,
    "rf_contribution": 0.5,
    "threshold": 0.5,
    "interpretation": "Disease"
  }
}
```

### Batch Prediction
```bash
POST /predict/batch
```

**Request Body:**
```json
{
  "patients": [
    { "age": 45, "sex": 1, "cp": 0, ... },
    { "age": 67, "sex": 1, "cp": 3, ... }
  ]
}
```

### Model Information
```bash
GET /models/info
```

**Response:**
```json
{
  "mlp": {
    "type": "Enhanced MLP with BatchNormalization",
    "input_shape": [null, 13],
    "total_params": 48641,
    "layers": 10
  },
  "random_forest": {
    "type": "Random Forest Classifier",
    "n_estimators": 100,
    "n_features": 13
  },
  "ensemble": {
    "type": "Probability Averaging",
    "weights": "50% MLP + 50% Random Forest",
    "performance": {
      "accuracy": 0.8852,
      "auc": 0.9643
    }
  }
}
```

---

## 🖥️ Demo UI Features

### Interactive Input
- **Quick Load Examples:** Pre-configured patient profiles (Healthy, Medium Risk, High Risk)
- **Manual Input:** Sliders and selectors for all 13 clinical features
- **Real-time Validation:** Input validation with helpful tooltips

### Visualizations
- **Probability Gauge:** Interactive gauge showing disease probability
- **Model Comparison:** Bar chart comparing MLP, Random Forest, and Ensemble predictions
- **Risk Assessment:** Color-coded risk levels (Low/Medium/High)

### Results Display
- **Prediction Metrics:** Probability, Confidence, Risk Level
- **Model Breakdown:** Individual model contributions
- **Clinical Recommendations:** Actionable advice based on risk level
- **Detailed Analysis:** JSON view of complete prediction results

---

## 🧪 Testing the API

### Using cURL

```bash
# Health check
curl http://localhost:8000/health

# Single prediction
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{
    "age": 63, "sex": 1, "cp": 3, "trestbps": 145,
    "chol": 233, "fbs": 1, "restecg": 0, "thalach": 150,
    "exang": 0, "oldpeak": 2.3, "slope": 0, "ca": 0, "thal": 1
  }'
```

### Using Python

```python
import requests

# API endpoint
url = "http://localhost:8000/predict"

# Patient data
patient = {
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

# Make prediction
response = requests.post(url, json=patient)
result = response.json()

print(f"Prediction: {result['prediction']}")
print(f"Probability: {result['ensemble_probability']:.2%}")
print(f"Risk Level: {result['risk_level']}")
```

---

## 📊 Feature Descriptions

| Feature | Description | Range | Example |
|---------|-------------|-------|---------|
| **age** | Patient age in years | 29-77 | 63 |
| **sex** | Sex (1=male, 0=female) | 0-1 | 1 |
| **cp** | Chest pain type | 0-3 | 3 |
| **trestbps** | Resting blood pressure (mm Hg) | 94-200 | 145 |
| **chol** | Serum cholesterol (mg/dl) | 126-564 | 233 |
| **fbs** | Fasting blood sugar > 120 mg/dl | 0-1 | 1 |
| **restecg** | Resting ECG results | 0-2 | 0 |
| **thalach** | Maximum heart rate achieved | 71-202 | 150 |
| **exang** | Exercise induced angina | 0-1 | 0 |
| **oldpeak** | ST depression induced by exercise | 0.0-6.2 | 2.3 |
| **slope** | Slope of peak exercise ST segment | 0-2 | 0 |
| **ca** | Number of major vessels (0-4) | 0-4 | 0 |
| **thal** | Thalassemia type | 0-3 | 1 |

---

## 🎯 Model Performance

### Ensemble Model (Production)
- **Accuracy:** 88.52%
- **Precision:** 81.82%
- **Recall:** 96.43% (only 1 false negative!)
- **F1-Score:** 88.52%
- **AUC:** 96.43% ⭐

### Individual Models
| Model | Accuracy | Recall | AUC |
|-------|----------|--------|-----|
| Enhanced MLP | 85.25% | 100.00% | 96.37% |
| Random Forest | 90.16% | 96.43% | 95.13% |
| **Ensemble** | **88.52%** | **96.43%** | **96.43%** |

---

## 🔧 Configuration

### API Server (main.py)

```python
# Change port
uvicorn.run(app, host="0.0.0.0", port=8000)

# Enable CORS for specific origins
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:8501"],  # Streamlit default
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
```

### Demo UI (demo.py)

```python
# Change API endpoint
API_URL = "http://localhost:8000"

# Customize page
st.set_page_config(
    page_title="Heart Disease Prediction",
    page_icon="❤️",
    layout="wide"
)
```

---

## 🐛 Troubleshooting

### API won't start
```bash
# Check if models exist
ls ../models/

# Install missing dependencies
pip install -r requirements.txt --upgrade

# Check port availability
netstat -ano | findstr :8000
```

### Demo UI connection error
```bash
# Ensure API is running first
curl http://localhost:8000/health

# Check Streamlit port
netstat -ano | findstr :8501
```

### Model loading errors
```bash
# Verify TensorFlow installation
python -c "import tensorflow as tf; print(tf.__version__)"

# Check model files
python -c "import tensorflow as tf; model = tf.keras.models.load_model('../models/mlp_clinical.keras'); print('Model loaded OK')"
```

---

## 📸 Screenshots

### API Documentation (Swagger UI)
Access at: `http://localhost:8000/docs`

**Features:**
- Interactive API testing
- Request/response schemas
- Try-it-out functionality

### Streamlit Demo
Access at: `http://localhost:8501`

**Features:**
- Patient data input form
- Real-time predictions
- Interactive visualizations
- Risk assessment dashboard

---

## 🚢 Deployment Options

### Local Development
```bash
python main.py  # API on localhost:8000
streamlit run demo.py  # UI on localhost:8501
```

### Docker (Future)
```dockerfile
FROM python:3.11-slim
WORKDIR /app
COPY requirements.txt .
RUN pip install -r requirements.txt
COPY . .
EXPOSE 8000
CMD ["python", "main.py"]
```

### Cloud Deployment
- **API:** Deploy to Heroku, AWS Lambda, or Google Cloud Run
- **Demo:** Deploy to Streamlit Cloud (free tier available)
- **Models:** Store in cloud storage (S3, GCS)

---

## 📝 Development Notes

### Adding New Features

**New API Endpoint:**
```python
@app.post("/predict/custom")
async def custom_prediction(data: CustomData):
    # Your logic here
    return {"result": "value"}
```

**New Demo Section:**
```python
st.markdown('<div class="sub-header">New Section</div>', unsafe_allow_html=True)
# Your content here
```

### Testing

```bash
# Run API tests
pytest tests/test_api.py

# Test specific endpoint
curl -X POST http://localhost:8000/predict -d @test_patient.json
```

---

## 🎓 Technical Details

### Model Architecture
- **MLP:** 4-layer deep network with BatchNormalization and Dropout
- **Random Forest:** 100 decision trees with balanced class weights
- **Ensemble:** Simple probability averaging (50-50 weighting)

### Preprocessing
- **StandardScaler:** Z-score normalization fitted on training data
- **Feature Range:** All 13 clinical features scaled to mean=0, std=1
- **Missing Values:** Handled during training (median/mode imputation)

### Inference Pipeline
1. Input validation (Pydantic models)
2. Feature scaling (StandardScaler)
3. Model predictions (MLP + RF)
4. Probability averaging
5. Risk stratification
6. Response formatting

---

## ✅ Day 5 Deliverables Checklist

- [x] FastAPI backend server (`main.py`)
- [x] `/predict` endpoint for single patient
- [x] `/predict/batch` endpoint for multiple patients
- [x] `/health` and `/models/info` endpoints
- [x] Streamlit interactive demo (`demo.py`)
- [x] Input form with validation
- [x] Real-time visualizations (gauges, charts)
- [x] Risk assessment and recommendations
- [x] Requirements file (`requirements.txt`)
- [x] Comprehensive documentation (this README)
- [ ] Screenshots of working demo
- [ ] Test inference examples

---

## 📞 Support

**Issues?** Check:
1. All dependencies installed: `pip list`
2. Models exist in `../models/`
3. API is running: `curl http://localhost:8000/health`
4. Ports are available (8000, 8501)

**Questions?** Refer to:
- API Documentation: `http://localhost:8000/docs`
- Streamlit Docs: https://docs.streamlit.io
- FastAPI Docs: https://fastapi.tiangolo.com

---

## 🎉 Success Criteria

✅ **API Running:** Health check returns status "healthy"  
✅ **Predictions Working:** POST to /predict returns valid results  
✅ **Demo Functional:** Streamlit UI loads and makes predictions  
✅ **Models Loaded:** All 3 models (MLP, RF, Scaler) load successfully  
✅ **Visualizations:** Gauges and charts display correctly  

**All criteria met = Day 5 Complete! 🎊**

---

*Generated: October 28, 2025*  
*Project: Heart Disease Detection - ML/DL Pipeline*  
*Status: Day 5 Deployment Ready* ✅
