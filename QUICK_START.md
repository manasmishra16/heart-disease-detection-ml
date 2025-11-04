# Quick Start Guide
## Heart Disease Prediction System

**Get the demo running in 15 minutes!**

---

## Prerequisites
- Python 3.11 installed
- 5GB free disk space
- Internet connection (for downloading data)

---

## Step 1: Setup Environment (5 minutes)

```powershell
# 1. Create virtual environment
python -m venv venv

# 2. Activate virtual environment
.\venv\Scripts\Activate.ps1

# 3. Install dependencies
pip install -r requirements.txt

# 4. Install app dependencies
pip install -r app/requirements.txt
```

**Verify installation:**
```powershell
python -c "import tensorflow; import sklearn; import streamlit; print('✅ All libraries installed!')"
```

---

## Step 2: Verify Project Files (2 minutes)

**Required files checklist:**
- [x] `models/model.h5` - Main MLP model
- [x] `models/mlp_clinical.keras` - Enhanced MLP model
- [x] `models/random_forest.pkl` - Random Forest model
- [x] `models/scaler.pkl` - Feature scaler
- [x] `results/cleaned_data.csv` - Preprocessed data
- [x] `app/main.py` - FastAPI backend
- [x] `app/demo.py` - Streamlit frontend

**Check models:**
```powershell
dir models\
```

**Expected output:**
```
model.h5
mlp_clinical.keras
random_forest.pkl
scaler.pkl
ensemble_predictions.pkl
```

---

## Step 3: Run Tests (3 minutes)

```powershell
# Run all tests
python run_all_tests.py
```

**Expected result:**
```
========================================
Running Tests: Day 1 - Setup & Data Loading
========================================
✅ Day 1 PASSED

... (Days 2-5)

========================================
Test Summary
========================================
✅ 5/5 test suites passed (100.00%)
🎉 All tests passed!
```

---

## Step 4: Launch Demo (5 minutes)

### Option A: Standalone Mode (Recommended - No API needed)

**Single command:**
```powershell
cd app
streamlit run demo.py
```

**Access demo:**
- Open browser: http://localhost:8502
- UI automatically loads in standalone mode
- Uses Random Forest model (90.16% accuracy)

**Demo features:**
- 📊 Interactive patient input form
- 🎯 Real-time risk prediction
- 📈 Probability gauge visualization
- 📋 Detailed feature explanations

---

### Option B: Full Stack (API + Demo)

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

**Access:**
- Demo UI: http://localhost:8501
- API Docs: http://localhost:8000/docs

**Note:** API may have TensorFlow DLL issues on Windows. Use standalone mode if errors occur.

---

## Step 5: Test Prediction (2 minutes)

### Example Patient Data

**Low Risk Patient:**
```
Age: 40
Sex: Female (0)
Chest Pain Type: 0 (Asymptomatic)
Resting BP: 120
Cholesterol: 200
Fasting Blood Sugar: 0
Resting ECG: 0
Max Heart Rate: 180
Exercise Angina: 0
ST Depression: 0.0
Slope: 2
Vessels Colored: 0
Thalassemia: 2
```

**Expected Result:** Low Risk (10-20% probability)

---

**High Risk Patient:**
```
Age: 65
Sex: Male (1)
Chest Pain Type: 3 (Typical Angina)
Resting BP: 160
Cholesterol: 350
Fasting Blood Sugar: 1
Resting ECG: 2
Max Heart Rate: 100
Exercise Angina: 1
ST Depression: 3.0
Slope: 0
Vessels Colored: 3
Thalassemia: 3
```

**Expected Result:** High Risk (80-95% probability)

---

## Troubleshooting

### Issue: TensorFlow DLL Error
**Solution:** Use standalone mode (no API required)
```powershell
cd app
streamlit run demo.py
```

### Issue: Port Already in Use
**Solution:** Change port
```powershell
streamlit run demo.py --server.port 8502
```

### Issue: Module Not Found
**Solution:** Reinstall dependencies
```powershell
pip install -r requirements.txt --force-reinstall
```

### Issue: Models Not Found
**Solution:** Check working directory
```powershell
# Run from project root
cd D:\Projects\MiniProject
python app/demo.py  # Wrong!

# Run from app directory
cd D:\Projects\MiniProject\app
streamlit run demo.py  # Correct!
```

---

## Feature Overview

### 1. Patient Input Form
- 13 clinical features
- Intuitive sliders and radio buttons
- Input validation
- Helpful descriptions

### 2. Risk Assessment
- Three risk levels (Low/Medium/High)
- Confidence score
- Probability percentage
- Color-coded visualization

### 3. Visualizations
- Probability gauge chart
- Feature importance plot
- Risk distribution
- Model comparison

### 4. Model Information
- MLP architecture details
- Random Forest parameters
- Ensemble strategy
- Performance metrics

---

## Performance Metrics

**Ensemble Model:**
- **Accuracy:** 90.16%
- **Recall:** 96.43% (catches 96% of disease cases)
- **Precision:** 82.69% (82% of positive predictions are correct)
- **AUC:** 95.13% (excellent discrimination)
- **False Negatives:** 1 (minimal missed cases)

**Individual Models:**
- Random Forest: 90.16% accuracy
- Enhanced MLP: 85.25% accuracy
- Ensemble: Best of both worlds

---

## Next Steps

1. **Explore the UI:**
   - Try different patient profiles
   - Check risk levels
   - View model explanations

2. **Review Code:**
   - `app/demo.py` - Frontend implementation
   - `app/main.py` - API endpoints
   - `models/` - Saved models

3. **Run Experiments:**
   - Modify patient features
   - Compare predictions
   - Analyze edge cases

4. **Extend Functionality:**
   - Add patient history tracking
   - Export predictions to PDF
   - Create custom visualizations

---

## Demo Screenshots

### Welcome Screen
![Welcome Screen](docs/screenshots/welcome_screen.png)

### Prediction Interface
![Prediction](docs/screenshots/prediction_interface.png)

### Results Dashboard
![Results](docs/screenshots/results_dashboard.png)

---

## API Usage (Optional)

### Health Check
```powershell
curl http://localhost:8000/health
```

### Make Prediction
```powershell
curl -X POST "http://localhost:8000/predict" `
  -H "Content-Type: application/json" `
  -d '{
    "age": 54, "sex": 1, "cp": 0, "trestbps": 130, "chol": 246,
    "fbs": 0, "restecg": 0, "thalach": 150, "exang": 0,
    "oldpeak": 1.0, "slope": 2, "ca": 0, "thal": 2
  }'
```

### Response
```json
{
  "mlp_probability": 0.23,
  "rf_probability": 0.18,
  "ensemble_probability": 0.21,
  "prediction": 0,
  "risk_level": "Low Risk",
  "confidence": 0.58
}
```

---

## File Locations

**Models:**
```
models/
├── model.h5                 # Main deliverable
├── mlp_clinical.keras       # Enhanced MLP
├── random_forest.pkl        # RF classifier
└── scaler.pkl              # Feature scaler
```

**Data:**
```
results/
├── cleaned_data.csv        # Preprocessed data
├── X_train.csv            # Training features
├── X_test.csv             # Test features
├── y_train.csv            # Training labels
└── y_test.csv             # Test labels
```

**Application:**
```
app/
├── main.py                # FastAPI backend
├── demo.py                # Streamlit UI
├── requirements.txt       # Dependencies
└── README.md             # Deployment guide
```

---

## Support

### Documentation
- **Reproduction Guide:** `REPRODUCTION_GUIDE.md` - Complete setup from scratch
- **API Documentation:** `app/README.md` - API endpoints and usage
- **Day Summaries:** `DAY[1-5]_SUMMARY.md` - Daily progress logs
- **Validation Report:** `validation_report.md` - Model performance

### Common Commands
```powershell
# Activate environment
.\venv\Scripts\Activate.ps1

# Run tests
python run_all_tests.py

# Start demo (standalone)
cd app ; streamlit run demo.py

# Start API
cd app ; python main.py

# View logs
streamlit run demo.py --server.headless false
```

### Getting Help
1. Check troubleshooting section above
2. Review `REPRODUCTION_GUIDE.md` for detailed instructions
3. Check test outputs: `python run_all_tests.py`
4. Verify models exist: `dir models\`

---

## Success Criteria

✅ **You have successfully completed setup if:**
- Virtual environment is activated
- All dependencies installed
- Tests pass (5/5)
- Demo loads at http://localhost:8502
- Predictions work with example patients
- Performance metrics display correctly

---

## Estimated Time

- **Setup:** 5 minutes
- **Verification:** 2 minutes
- **Testing:** 3 minutes
- **Demo Launch:** 5 minutes
- **Total:** **15 minutes**

---

**Ready to start? Run:**
```powershell
.\venv\Scripts\Activate.ps1
cd app
streamlit run demo.py
```

**Then open:** http://localhost:8502

---

**Version:** 1.0.0  
**Last Updated:** October 28, 2025  
**Status:** Production Ready ✅
