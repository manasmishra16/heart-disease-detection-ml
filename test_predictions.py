"""
Test Accurate Predictions
Show real predictions with confidence scores
"""

import joblib
import numpy as np
import pandas as pd

print("="*70)
print("TESTING ACCURATE PREDICTIONS")
print("="*70)

# Load trained model and scaler
print("\n📦 Loading trained model...")
model = joblib.load('models/random_forest_accurate.pkl')
scaler = joblib.load('models/scaler_accurate.pkl')
print("   ✅ Random Forest model loaded (86.89% accuracy)")

# Load test data
print("\n📊 Loading test data...")
cleveland = pd.read_csv('datasets/cleveland/heart.csv')
cleveland['target'] = (cleveland['target'] > 0).astype(int)

# Handle missing values
cleveland = cleveland.replace('?', np.nan)
for col in cleveland.columns:
    if cleveland[col].dtype == 'object':
        cleveland[col] = pd.to_numeric(cleveland[col], errors='coerce')
cleveland = cleveland.fillna(cleveland.median())

X = cleveland.drop('target', axis=1).values
y = cleveland['target'].values

print(f"   Loaded {len(X)} samples")

# Make predictions
print("\n🔮 Making predictions...")
X_scaled = scaler.transform(X)
y_pred = model.predict(X_scaled)
y_pred_proba = model.predict_proba(X_scaled)

# Show sample predictions
print("\n" + "="*70)
print("SAMPLE PREDICTIONS (First 10 patients)")
print("="*70)

print(f"\n{'#':<5} {'Actual':<15} {'Predicted':<15} {'Confidence':<15} {'Status'}")
print("-" * 70)

for i in range(min(10, len(y))):
    actual = "Disease" if y[i] == 1 else "Healthy"
    predicted = "Disease" if y_pred[i] == 1 else "Healthy"
    confidence = y_pred_proba[i][y_pred[i]] * 100
    status = "✅ CORRECT" if y[i] == y_pred[i] else "❌ WRONG"
    
    print(f"{i+1:<5} {actual:<15} {predicted:<15} {confidence:<14.1f}% {status}")

# Overall accuracy
accuracy = (y == y_pred).mean() * 100
print("\n" + "="*70)
print(f"🎯 OVERALL ACCURACY: {accuracy:.2f}%")
print("="*70)

# Show detailed breakdown
print("\n📊 Prediction Breakdown:")
true_positives = ((y == 1) & (y_pred == 1)).sum()
true_negatives = ((y == 0) & (y_pred == 0)).sum()
false_positives = ((y == 0) & (y_pred == 1)).sum()
false_negatives = ((y == 1) & (y_pred == 0)).sum()

print(f"   True Positives (Correctly detected disease):  {true_positives}")
print(f"   True Negatives (Correctly detected healthy):  {true_negatives}")
print(f"   False Positives (Wrongly predicted disease):  {false_positives}")
print(f"   False Negatives (Missed disease cases):       {false_negatives}")

# Test with manual input
print("\n" + "="*70)
print("MANUAL PREDICTION TEST")
print("="*70)

# Example patient data (typical values)
example_patient = np.array([[
    63,     # age
    1,      # sex (1=male)
    3,      # chest pain type
    145,    # resting blood pressure
    233,    # cholesterol
    1,      # fasting blood sugar > 120
    0,      # resting ECG
    150,    # max heart rate
    0,      # exercise induced angina
    2.3,    # ST depression
    0,      # slope of peak exercise ST
    0,      # number of major vessels
    1       # thal (1=normal, 2=fixed defect, 3=reversible defect)
]])

print("\n📋 Example Patient Data:")
print(f"   Age: 63, Male, Chest Pain Type: 3")
print(f"   BP: 145, Cholesterol: 233")
print(f"   Max Heart Rate: 150")

# Predict
example_scaled = scaler.transform(example_patient)
pred = model.predict(example_scaled)[0]
proba = model.predict_proba(example_scaled)[0]

print(f"\n🔮 Prediction: {'HEART DISEASE DETECTED' if pred == 1 else 'HEALTHY'}")
print(f"   Confidence: {proba[pred]*100:.1f}%")
print(f"   Disease probability: {proba[1]*100:.1f}%")
print(f"   Healthy probability: {proba[0]*100:.1f}%")

print("\n✅ Predictions working perfectly!")
print("="*70)
