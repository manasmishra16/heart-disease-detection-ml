"""
COMPREHENSIVE ACCURACY TEST
Show exact accuracy for ALL trained models
"""

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, roc_auc_score

print("="*80)
print("COMPREHENSIVE ACCURACY TEST - ALL MODELS")
print("="*80)

# Load test data
print("\n📊 Loading Cleveland test dataset...")
cleveland = pd.read_csv('datasets/cleveland/heart.csv')
cleveland['target'] = (cleveland['target'] > 0).astype(int)
cleveland = cleveland.replace('?', np.nan)
for col in cleveland.columns:
    if cleveland[col].dtype == 'object':
        cleveland[col] = pd.to_numeric(cleveland[col], errors='coerce')
cleveland = cleveland.fillna(cleveland.median())

print(f"   Loaded: {cleveland.shape[0]} samples")
print(f"   Healthy: {(cleveland['target'] == 0).sum()}")
print(f"   Disease: {(cleveland['target'] == 1).sum()}")

# Test each model
models_to_test = [
    {
        'name': 'Random Forest (Accurate)',
        'model_file': 'models/random_forest_accurate.pkl',
        'scaler_file': 'models/scaler_accurate.pkl',
        'feature_engineering': False
    },
    {
        'name': 'Gradient Boosting (Accurate)',
        'model_file': 'models/gradient_boosting_accurate.pkl',
        'scaler_file': 'models/scaler_accurate.pkl',
        'feature_engineering': False
    },
    {
        'name': 'Random Forest (Best)',
        'model_file': 'models/random_forest_best.pkl',
        'scaler_file': 'models/scaler_best.pkl',
        'feature_engineering': True
    },
    {
        'name': 'Extra Trees (Best)',
        'model_file': 'models/extra_trees_best.pkl',
        'scaler_file': 'models/scaler_best.pkl',
        'feature_engineering': True
    },
    {
        'name': 'Gradient Boosting (Best)',
        'model_file': 'models/gradient_boosting_best.pkl',
        'scaler_file': 'models/scaler_best.pkl',
        'feature_engineering': True
    }
]

results = []

for model_info in models_to_test:
    try:
        print(f"\n{'='*80}")
        print(f"Testing: {model_info['name']}")
        print(f"{'='*80}")
        
        # Load model and scaler
        model = joblib.load(model_info['model_file'])
        scaler = joblib.load(model_info['scaler_file'])
        
        # Prepare data
        X = cleveland.drop('target', axis=1)
        y = cleveland['target'].values
        
        # Feature engineering if needed
        if model_info['feature_engineering']:
            X['age_chol'] = X['age'] * X['chol']
            X['age_thalach'] = X['age'] * X['thalach']
            X['cp_thalach'] = X['cp'] * X['thalach']
            X['oldpeak_slope'] = X['oldpeak'] * X['slope']
            X['ca_thal'] = X['ca'] * X['thal']
            X['age_squared'] = X['age'] ** 2
            X['chol_squared'] = X['chol'] ** 2
            X['thalach_squared'] = X['thalach'] ** 2
        
        # Scale and predict
        X_scaled = scaler.transform(X)
        y_pred = model.predict(X_scaled)
        y_pred_proba = model.predict_proba(X_scaled)[:, 1]
        
        # Calculate metrics
        accuracy = accuracy_score(y, y_pred)
        auc = roc_auc_score(y, y_pred_proba)
        
        print(f"\n📊 Results:")
        print(f"   Accuracy:  {accuracy*100:.2f}%")
        print(f"   AUC Score: {auc*100:.2f}%")
        
        # Confusion matrix
        cm = confusion_matrix(y, y_pred)
        tn, fp, fn, tp = cm.ravel()
        
        print(f"\n   Confusion Matrix:")
        print(f"   [[{tn:3d}  {fp:3d}]")
        print(f"    [{fn:3d}  {tp:3d}]]")
        
        print(f"\n   True Positives:  {tp} (Correctly detected disease)")
        print(f"   True Negatives:  {tn} (Correctly detected healthy)")
        print(f"   False Positives: {fp} (Wrongly predicted disease)")
        print(f"   False Negatives: {fn} (Missed disease cases)")
        
        sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0
        specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        
        print(f"\n   Sensitivity (Recall): {sensitivity*100:.2f}%")
        print(f"   Specificity:          {specificity*100:.2f}%")
        print(f"   Precision:            {precision*100:.2f}%")
        
        # Sample predictions
        print(f"\n   Sample Predictions (First 5):")
        for i in range(min(5, len(y))):
            actual = "Disease" if y[i] == 1 else "Healthy"
            predicted = "Disease" if y_pred[i] == 1 else "Healthy"
            confidence = y_pred_proba[i] if y_pred[i] == 1 else (1 - y_pred_proba[i])
            status = "✅" if y[i] == y_pred[i] else "❌"
            print(f"   {i+1}. Actual: {actual:<8} | Predicted: {predicted:<8} | Confidence: {confidence*100:5.1f}% {status}")
        
        results.append({
            'Model': model_info['name'],
            'Accuracy': accuracy * 100,
            'AUC': auc * 100,
            'TP': tp,
            'TN': tn,
            'FP': fp,
            'FN': fn,
            'Sensitivity': sensitivity * 100,
            'Specificity': specificity * 100
        })
        
        print(f"\n✅ {model_info['name']}: {accuracy*100:.2f}% accuracy")
        
    except FileNotFoundError:
        print(f"\n❌ Model not found: {model_info['model_file']}")
        print(f"   Run training script to create this model")
    except Exception as e:
        print(f"\n❌ Error testing {model_info['name']}: {e}")

# Final comparison
print(f"\n{'='*80}")
print("FINAL COMPARISON - ALL MODELS")
print(f"{'='*80}")

if results:
    # Sort by accuracy
    results_sorted = sorted(results, key=lambda x: x['Accuracy'], reverse=True)
    
    print(f"\n{'Rank':<6} {'Model':<35} {'Accuracy':<12} {'AUC':<12} {'Errors'}")
    print("-" * 80)
    
    for i, result in enumerate(results_sorted, 1):
        marker = "🏆" if i == 1 else f"{i}."
        errors = result['FP'] + result['FN']
        print(f"{marker:<6} {result['Model']:<35} {result['Accuracy']:>10.2f}%  {result['AUC']:>10.2f}%  {errors}")
    
    # Best model
    best = results_sorted[0]
    print(f"\n🏆 BEST MODEL: {best['Model']}")
    print(f"   Accuracy:    {best['Accuracy']:.2f}%")
    print(f"   AUC Score:   {best['AUC']:.2f}%")
    print(f"   Sensitivity: {best['Sensitivity']:.2f}%")
    print(f"   Specificity: {best['Specificity']:.2f}%")
    print(f"   Total Errors: {best['FP'] + best['FN']} out of {cleveland.shape[0]} samples")
    
    print(f"\n📊 Detailed Performance:")
    print(f"   ✅ Correct Predictions: {best['TP'] + best['TN']}")
    print(f"   ❌ Wrong Predictions:   {best['FP'] + best['FN']}")
    print(f"   ⚠️ Missed Disease Cases: {best['FN']} (most critical)")
    print(f"   ⚠️ False Alarms:        {best['FP']} (less critical)")

print(f"\n{'='*80}")
print("✅ Testing Complete!")
print(f"{'='*80}")
