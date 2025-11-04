#!/usr/bin/env python3
"""
Retrain models to achieve 98%+ accuracy
Fix prediction issues by using proper hyperparameters
"""

import pandas as pd
import numpy as np
import joblib
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import xgboost as xgb
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import warnings
warnings.filterwarnings('ignore')

print("=" * 70)
print("RETRAINING MODELS FOR 98%+ ACCURACY")
print("=" * 70)

# Load data
print("\n📊 Loading data...")
data = pd.read_csv('results/cleaned_data.csv')
X = data.drop('target', axis=1)
y = data['target']

print(f"   Dataset: {len(data)} samples")
print(f"   Features: {X.shape[1]}")
print(f"   Disease cases: {y.sum()} ({y.sum()/len(y)*100:.1f}%)")
print(f"   Healthy cases: {(1-y).sum()} ({(1-y).sum()/len(y)*100:.1f}%)")

# Split data
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

print(f"\n   Training set: {len(X_train)} samples")
print(f"   Test set: {len(X_test)} samples")

# Scale features
print("\n⚙️  Scaling features...")
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Save scaler (needed for demo)
joblib.dump(scaler, 'models/scaler.pkl')
print("   ✓ Scaler saved")

# ============================================================================
# 1. ULTRA-TUNED RANDOM FOREST
# ============================================================================
print("\n" + "=" * 70)
print("1. TRAINING ULTRA-TUNED RANDOM FOREST")
print("=" * 70)

rf_model = RandomForestClassifier(
    n_estimators=500,          # More trees
    max_depth=15,              # Deeper trees
    min_samples_split=2,       # More splitting
    min_samples_leaf=1,        # Detailed leaves
    max_features='sqrt',
    bootstrap=True,
    class_weight='balanced',   # Handle imbalance
    random_state=42,
    n_jobs=-1
)

print("Training Random Forest...")
rf_model.fit(X_train_scaled, y_train)

y_pred_rf = rf_model.predict(X_test_scaled)
rf_acc = accuracy_score(y_test, y_pred_rf) * 100

print(f"\n✓ Random Forest Accuracy: {rf_acc:.2f}%")
print("\nClassification Report:")
print(classification_report(y_test, y_pred_rf, target_names=['No Disease', 'Disease']))

joblib.dump(rf_model, 'models/random_forest.pkl')
print("✓ Model saved to models/random_forest.pkl")

# ============================================================================
# 2. ULTRA-TUNED XGBOOST
# ============================================================================
print("\n" + "=" * 70)
print("2. TRAINING ULTRA-TUNED XGBOOST")
print("=" * 70)

xgb_model = xgb.XGBClassifier(
    n_estimators=500,
    max_depth=8,
    learning_rate=0.05,
    subsample=0.9,
    colsample_bytree=0.9,
    gamma=0.1,
    reg_alpha=0.1,
    reg_lambda=1.0,
    scale_pos_weight=1.0,
    random_state=42,
    use_label_encoder=False,
    eval_metric='logloss'
)

print("Training XGBoost...")
xgb_model.fit(X_train_scaled, y_train)

y_pred_xgb = xgb_model.predict(X_test_scaled)
xgb_acc = accuracy_score(y_test, y_pred_xgb) * 100

print(f"\n✓ XGBoost Accuracy: {xgb_acc:.2f}%")
print("\nClassification Report:")
print(classification_report(y_test, y_pred_xgb, target_names=['No Disease', 'Disease']))

joblib.dump(xgb_model, 'models/xgboost.pkl')
print("✓ Model saved to models/xgboost.pkl")

# ============================================================================
# 3. DEEP NEURAL NETWORK (ENHANCED MLP)
# ============================================================================
print("\n" + "=" * 70)
print("3. TRAINING DEEP NEURAL NETWORK")
print("=" * 70)

def create_deep_model(input_dim):
    model = keras.Sequential([
        layers.Input(shape=(input_dim,)),
        
        # Deep architecture with batch normalization
        layers.Dense(256, activation='relu', kernel_regularizer=keras.regularizers.l2(0.001)),
        layers.BatchNormalization(),
        layers.Dropout(0.3),
        
        layers.Dense(128, activation='relu', kernel_regularizer=keras.regularizers.l2(0.001)),
        layers.BatchNormalization(),
        layers.Dropout(0.3),
        
        layers.Dense(64, activation='relu', kernel_regularizer=keras.regularizers.l2(0.001)),
        layers.BatchNormalization(),
        layers.Dropout(0.2),
        
        layers.Dense(32, activation='relu', kernel_regularizer=keras.regularizers.l2(0.001)),
        layers.BatchNormalization(),
        layers.Dropout(0.2),
        
        layers.Dense(16, activation='relu'),
        layers.Dropout(0.1),
        
        layers.Dense(1, activation='sigmoid')
    ])
    
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=0.001),
        loss='binary_crossentropy',
        metrics=['accuracy', keras.metrics.Precision(), keras.metrics.Recall(), keras.metrics.AUC()]
    )
    
    return model

mlp_model = create_deep_model(X_train_scaled.shape[1])

print("Training Deep Neural Network...")
print(f"Architecture: 256 -> 128 -> 64 -> 32 -> 16 -> 1")

early_stopping = keras.callbacks.EarlyStopping(
    monitor='val_loss',
    patience=20,
    restore_best_weights=True
)

reduce_lr = keras.callbacks.ReduceLROnPlateau(
    monitor='val_loss',
    factor=0.5,
    patience=10,
    min_lr=1e-6
)

history = mlp_model.fit(
    X_train_scaled, y_train,
    validation_split=0.2,
    epochs=200,
    batch_size=16,
    callbacks=[early_stopping, reduce_lr],
    verbose=0
)

# Evaluate
y_pred_mlp_prob = mlp_model.predict(X_test_scaled, verbose=0)
y_pred_mlp = (y_pred_mlp_prob > 0.5).astype(int).flatten()
mlp_acc = accuracy_score(y_test, y_pred_mlp) * 100

print(f"\n✓ Deep Neural Network Accuracy: {mlp_acc:.2f}%")
print("\nClassification Report:")
print(classification_report(y_test, y_pred_mlp, target_names=['No Disease', 'Disease']))

mlp_model.save('models/mlp_clinical.keras')
print("✓ Model saved to models/mlp_clinical.keras")

# ============================================================================
# 4. ENSEMBLE PREDICTIONS
# ============================================================================
print("\n" + "=" * 70)
print("4. ENSEMBLE EVALUATION")
print("=" * 70)

# Get probabilities from all models
rf_probs = rf_model.predict_proba(X_test_scaled)[:, 1]
xgb_probs = xgb_model.predict_proba(X_test_scaled)[:, 1]
mlp_probs = mlp_model.predict(X_test_scaled, verbose=0).flatten()

# Weighted ensemble (optimized weights)
ensemble_probs = 0.30 * rf_probs + 0.35 * xgb_probs + 0.35 * mlp_probs
y_pred_ensemble = (ensemble_probs >= 0.5).astype(int)

ensemble_acc = accuracy_score(y_test, y_pred_ensemble) * 100

print(f"\n✓ Ensemble Accuracy: {ensemble_acc:.2f}%")
print("\nClassification Report:")
print(classification_report(y_test, y_pred_ensemble, target_names=['No Disease', 'Disease']))

print("\nConfusion Matrix:")
cm = confusion_matrix(y_test, y_pred_ensemble)
print(cm)
print(f"\nTrue Negatives: {cm[0,0]}")
print(f"False Positives: {cm[0,1]}")
print(f"False Negatives: {cm[1,0]}")
print(f"True Positives: {cm[1,1]}")

# Calculate additional metrics
from sklearn.metrics import precision_score, recall_score, f1_score, roc_auc_score

precision = precision_score(y_test, y_pred_ensemble) * 100
recall = recall_score(y_test, y_pred_ensemble) * 100
f1 = f1_score(y_test, y_pred_ensemble) * 100
auc = roc_auc_score(y_test, ensemble_probs) * 100

print(f"\nPrecision: {precision:.2f}%")
print(f"Recall (Sensitivity): {recall:.2f}%")
print(f"F1-Score: {f1:.2f}%")
print(f"AUC-ROC: {auc:.2f}%")

# ============================================================================
# 5. SAVE ENSEMBLE PREDICTIONS
# ============================================================================
ensemble_predictions = {
    'rf_probs': rf_probs,
    'xgb_probs': xgb_probs,
    'mlp_probs': mlp_probs,
    'ensemble_probs': ensemble_probs,
    'predictions': y_pred_ensemble,
    'actual': y_test.values
}

joblib.dump(ensemble_predictions, 'models/ensemble_predictions.pkl')
print("\n✓ Ensemble predictions saved")

# ============================================================================
# SUMMARY
# ============================================================================
print("\n" + "=" * 70)
print("TRAINING COMPLETE - SUMMARY")
print("=" * 70)

print(f"\n📊 Individual Model Accuracies:")
print(f"   Random Forest:      {rf_acc:.2f}%")
print(f"   XGBoost:            {xgb_acc:.2f}%")
print(f"   Deep Neural Net:    {mlp_acc:.2f}%")
print(f"\n🎯 ENSEMBLE ACCURACY:  {ensemble_acc:.2f}%")
print(f"\n✅ Precision:          {precision:.2f}%")
print(f"✅ Recall:             {recall:.2f}%")
print(f"✅ F1-Score:           {f1:.2f}%")
print(f"✅ AUC-ROC:            {auc:.2f}%")

if ensemble_acc >= 98.0:
    print("\n🎉 TARGET ACHIEVED! Ensemble accuracy >= 98%")
else:
    print(f"\n⚠️  Ensemble accuracy is {ensemble_acc:.2f}% (target: 98%)")
    print("   Note: 98%+ on this small dataset may require careful validation")

print("\n" + "=" * 70)
print("All models saved to models/ directory")
print("Ready to use in demo.py")
print("=" * 70)
