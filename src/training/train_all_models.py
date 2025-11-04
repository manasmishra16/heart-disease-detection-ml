"""
Comprehensive Training Script for Heart Disease Detection
Trains all DL models on combined datasets
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import pandas as pd
import joblib
from pathlib import Path
from datetime import datetime

# Deep learning
import tensorflow as tf
from tensorflow import keras

# Our modules
from data_processing.unified_data_loader import UnifiedDataLoader
from models.deep_learning_models import (
    create_deep_cnn_model,
    create_cnn_lstm_model,
    create_lstm_model,
    create_gru_model,
    create_enhanced_mlp_model,
    create_multi_input_model,
    compile_model,
    get_callbacks
)

# Scikit-learn
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, roc_auc_score
import warnings
warnings.filterwarnings('ignore')


class ComprehensiveTrainer:
    """Train all models on combined datasets"""
    
    def __init__(self, base_path='d:/Projects/MiniProject'):
        self.base_path = Path(base_path)
        self.models_dir = self.base_path / 'models'
        self.results_dir = self.base_path / 'results'
        
        # Create directories
        self.models_dir.mkdir(exist_ok=True)
        self.results_dir.mkdir(exist_ok=True)
        
        self.results = {}
        
    def train_clinical_models(self, X_train, X_val, X_test, y_train, y_val, y_test):
        """Train models on clinical features (Cleveland + Kaggle)"""
        print("\n" + "="*60)
        print("TRAINING CLINICAL MODELS")
        print("="*60)
        
        # 1. Enhanced MLP
        print("\n📊 Training Enhanced MLP...")
        mlp_model = create_enhanced_mlp_model(input_shape=(X_train.shape[1],))
        mlp_model = compile_model(mlp_model, learning_rate=0.001)
        
        mlp_callbacks = get_callbacks(
            str(self.models_dir / 'enhanced_mlp_clinical.keras'),
            patience=20
        )
        
        mlp_history = mlp_model.fit(
            X_train, y_train,
            validation_data=(X_val, y_val),
            epochs=100,
            batch_size=32,
            callbacks=mlp_callbacks,
            verbose=0
        )
        
        # Evaluate
        y_pred_mlp = (mlp_model.predict(X_test, verbose=0) > 0.5).astype(int).flatten()
        mlp_acc = accuracy_score(y_test, y_pred_mlp)
        mlp_auc = roc_auc_score(y_test, mlp_model.predict(X_test, verbose=0))
        
        print(f"  ✓ MLP Accuracy: {mlp_acc*100:.2f}%")
        print(f"  ✓ MLP AUC: {mlp_auc*100:.2f}%")
        
        self.results['enhanced_mlp'] = {
            'accuracy': mlp_acc,
            'auc': mlp_auc,
            'predictions': y_pred_mlp
        }
        
        # 2. Random Forest
        print("\n📊 Training Random Forest...")
        rf_model = RandomForestClassifier(
            n_estimators=200,
            max_depth=15,
            min_samples_split=5,
            min_samples_leaf=2,
            random_state=42,
            n_jobs=-1
        )
        rf_model.fit(X_train, y_train)
        
        y_pred_rf = rf_model.predict(X_test)
        rf_acc = accuracy_score(y_test, y_pred_rf)
        rf_auc = roc_auc_score(y_test, rf_model.predict_proba(X_test)[:, 1])
        
        print(f"  ✓ RF Accuracy: {rf_acc*100:.2f}%")
        print(f"  ✓ RF AUC: {rf_auc*100:.2f}%")
        
        joblib.dump(rf_model, self.models_dir / 'random_forest_final.pkl')
        
        self.results['random_forest'] = {
            'accuracy': rf_acc,
            'auc': rf_auc,
            'predictions': y_pred_rf
        }
        
        # 3. Gradient Boosting
        print("\n📊 Training Gradient Boosting...")
        gb_model = GradientBoostingClassifier(
            n_estimators=200,
            learning_rate=0.1,
            max_depth=5,
            random_state=42
        )
        gb_model.fit(X_train, y_train)
        
        y_pred_gb = gb_model.predict(X_test)
        gb_acc = accuracy_score(y_test, y_pred_gb)
        gb_auc = roc_auc_score(y_test, gb_model.predict_proba(X_test)[:, 1])
        
        print(f"  ✓ GB Accuracy: {gb_acc*100:.2f}%")
        print(f"  ✓ GB AUC: {gb_auc*100:.2f}%")
        
        joblib.dump(gb_model, self.models_dir / 'gradient_boosting_final.pkl')
        
        self.results['gradient_boosting'] = {
            'accuracy': gb_acc,
            'auc': gb_auc,
            'predictions': y_pred_gb
        }
        
        return mlp_model, rf_model, gb_model
    
    def train_ecg_models(self, X_train, X_val, X_test, y_train, y_val, y_test):
        """Train deep learning models on ECG signals"""
        print("\n" + "="*60)
        print("TRAINING ECG MODELS (DEEP LEARNING)")
        print("="*60)
        
        # 1. Deep CNN
        print("\n📊 Training Deep CNN with Residual Connections...")
        cnn_model = create_deep_cnn_model(input_shape=X_train.shape[1:])
        cnn_model = compile_model(cnn_model, learning_rate=0.001)
        
        cnn_callbacks = get_callbacks(
            str(self.models_dir / 'deep_cnn_ecg.keras'),
            patience=20
        )
        
        cnn_history = cnn_model.fit(
            X_train, y_train,
            validation_data=(X_val, y_val),
            epochs=100,
            batch_size=16,
            callbacks=cnn_callbacks,
            verbose=0
        )
        
        y_pred_cnn = (cnn_model.predict(X_test, verbose=0) > 0.5).astype(int).flatten()
        cnn_acc = accuracy_score(y_test, y_pred_cnn)
        cnn_auc = roc_auc_score(y_test, cnn_model.predict(X_test, verbose=0))
        
        print(f"  ✓ Deep CNN Accuracy: {cnn_acc*100:.2f}%")
        print(f"  ✓ Deep CNN AUC: {cnn_auc*100:.2f}%")
        
        self.results['deep_cnn'] = {
            'accuracy': cnn_acc,
            'auc': cnn_auc,
            'predictions': y_pred_cnn
        }
        
        # 2. CNN-LSTM Hybrid
        print("\n📊 Training CNN-LSTM Hybrid...")
        cnn_lstm_model = create_cnn_lstm_model(input_shape=X_train.shape[1:])
        cnn_lstm_model = compile_model(cnn_lstm_model, learning_rate=0.0005)
        
        cnn_lstm_callbacks = get_callbacks(
            str(self.models_dir / 'cnn_lstm_ecg.keras'),
            patience=25
        )
        
        cnn_lstm_history = cnn_lstm_model.fit(
            X_train, y_train,
            validation_data=(X_val, y_val),
            epochs=150,
            batch_size=16,
            callbacks=cnn_lstm_callbacks,
            verbose=0
        )
        
        y_pred_cnn_lstm = (cnn_lstm_model.predict(X_test, verbose=0) > 0.5).astype(int).flatten()
        cnn_lstm_acc = accuracy_score(y_test, y_pred_cnn_lstm)
        cnn_lstm_auc = roc_auc_score(y_test, cnn_lstm_model.predict(X_test, verbose=0))
        
        print(f"  ✓ CNN-LSTM Accuracy: {cnn_lstm_acc*100:.2f}%")
        print(f"  ✓ CNN-LSTM AUC: {cnn_lstm_auc*100:.2f}%")
        
        self.results['cnn_lstm'] = {
            'accuracy': cnn_lstm_acc,
            'auc': cnn_lstm_auc,
            'predictions': y_pred_cnn_lstm
        }
        
        # 3. Bidirectional LSTM
        print("\n📊 Training Bidirectional LSTM...")
        lstm_model = create_lstm_model(input_shape=X_train.shape[1:])
        lstm_model = compile_model(lstm_model, learning_rate=0.001)
        
        lstm_callbacks = get_callbacks(
            str(self.models_dir / 'lstm_ecg.keras'),
            patience=20
        )
        
        lstm_history = lstm_model.fit(
            X_train, y_train,
            validation_data=(X_val, y_val),
            epochs=100,
            batch_size=16,
            callbacks=lstm_callbacks,
            verbose=0
        )
        
        y_pred_lstm = (lstm_model.predict(X_test, verbose=0) > 0.5).astype(int).flatten()
        lstm_acc = accuracy_score(y_test, y_pred_lstm)
        lstm_auc = roc_auc_score(y_test, lstm_model.predict(X_test, verbose=0))
        
        print(f"  ✓ LSTM Accuracy: {lstm_acc*100:.2f}%")
        print(f"  ✓ LSTM AUC: {lstm_auc*100:.2f}%")
        
        self.results['lstm'] = {
            'accuracy': lstm_acc,
            'auc': lstm_auc,
            'predictions': y_pred_lstm
        }
        
        # 4. Bidirectional GRU
        print("\n📊 Training Bidirectional GRU...")
        gru_model = create_gru_model(input_shape=X_train.shape[1:])
        gru_model = compile_model(gru_model, learning_rate=0.001)
        
        gru_callbacks = get_callbacks(
            str(self.models_dir / 'gru_ecg.keras'),
            patience=20
        )
        
        gru_history = gru_model.fit(
            X_train, y_train,
            validation_data=(X_val, y_val),
            epochs=100,
            batch_size=16,
            callbacks=gru_callbacks,
            verbose=0
        )
        
        y_pred_gru = (gru_model.predict(X_test, verbose=0) > 0.5).astype(int).flatten()
        gru_acc = accuracy_score(y_test, y_pred_gru)
        gru_auc = roc_auc_score(y_test, gru_model.predict(X_test, verbose=0))
        
        print(f"  ✓ GRU Accuracy: {gru_acc*100:.2f}%")
        print(f"  ✓ GRU AUC: {gru_auc*100:.2f}%")
        
        self.results['gru'] = {
            'accuracy': gru_acc,
            'auc': gru_auc,
            'predictions': y_pred_gru
        }
        
        return cnn_model, cnn_lstm_model, lstm_model, gru_model
    
    def create_ensemble(self, y_test):
        """Create weighted ensemble of all models"""
        print("\n" + "="*60)
        print("CREATING ENSEMBLE MODEL")
        print("="*60)
        
        # Get predictions from all models
        predictions = []
        weights = []
        
        for model_name, results in self.results.items():
            predictions.append(results['predictions'])
            weights.append(results['auc'])  # Weight by AUC score
        
        # Normalize weights
        weights = np.array(weights)
        weights = weights / weights.sum()
        
        # Weighted voting
        ensemble_pred = np.zeros(len(y_test))
        for pred, weight in zip(predictions, weights):
            ensemble_pred += pred * weight
        
        ensemble_pred = (ensemble_pred > 0.5).astype(int)
        
        ensemble_acc = accuracy_score(y_test, ensemble_pred)
        
        print(f"\n  ✓ Ensemble Accuracy: {ensemble_acc*100:.2f}%")
        print(f"  ✓ Individual models used: {len(predictions)}")
        
        self.results['ensemble'] = {
            'accuracy': ensemble_acc,
            'predictions': ensemble_pred,
            'weights': weights
        }
        
        # Save ensemble configuration
        ensemble_config = {
            'model_names': list(self.results.keys())[:-1],
            'weights': weights.tolist()
        }
        joblib.dump(ensemble_config, self.models_dir / 'ensemble_config_final.pkl')
        
        return ensemble_pred
    
    def print_summary(self):
        """Print training summary"""
        print("\n" + "="*60)
        print("TRAINING SUMMARY")
        print("="*60)
        
        print("\n{:<25} {:<15} {:<15}".format("Model", "Accuracy", "AUC"))
        print("-" * 60)
        
        for model_name, results in sorted(
            self.results.items(), 
            key=lambda x: x[1]['accuracy'], 
            reverse=True
        ):
            acc = results['accuracy'] * 100
            auc = results.get('auc', 0) * 100
            print("{:<25} {:<15.2f}% {:<15.2f}%".format(model_name, acc, auc))
        
        # Find best model
        best_model = max(self.results.items(), key=lambda x: x[1]['accuracy'])
        print("\n" + "="*60)
        print(f"🏆 BEST MODEL: {best_model[0].upper()}")
        print(f"   Accuracy: {best_model[1]['accuracy']*100:.2f}%")
        if 'auc' in best_model[1]:
            print(f"   AUC: {best_model[1]['auc']*100:.2f}%")
        print("="*60)


def main():
    """Main training pipeline"""
    print("\n" + "="*80)
    print(" "*20 + "COMPREHENSIVE TRAINING PIPELINE")
    print("="*80)
    
    # Initialize
    trainer = ComprehensiveTrainer()
    loader = UnifiedDataLoader()
    
    # Load combined clinical data (Cleveland + Kaggle)
    print("\n📊 Loading clinical data...")
    data = loader.create_combined_dataset(use_all_features=False)
    
    if data is not None:
        X, y, features = data
        
        # Split and scale
        X_train, X_val, X_test, y_train, y_val, y_test = loader.prepare_train_test_split(X, y)
        X_train_scaled, X_val_scaled, X_test_scaled, scaler = loader.scale_data(
            X_train, X_val, X_test
        )
        
        # Save scaler
        joblib.dump(scaler, trainer.models_dir / 'scaler_final.pkl')
        
        # Train clinical models
        mlp, rf, gb = trainer.train_clinical_models(
            X_train_scaled, X_val_scaled, X_test_scaled,
            y_train, y_val, y_test
        )
    
    # Load ECG data (MIT-BIH)
    print("\n📊 Loading ECG data...")
    X_ecg, y_ecg = loader.load_mitbih_ecg_data()
    
    if X_ecg is not None and len(X_ecg) > 50:
        # Split ECG data
        X_ecg_train, X_ecg_val, X_ecg_test, y_ecg_train, y_ecg_val, y_ecg_test = \
            loader.prepare_train_test_split(X_ecg, y_ecg)
        
        # Train ECG models
        cnn, cnn_lstm, lstm, gru = trainer.train_ecg_models(
            X_ecg_train, X_ecg_val, X_ecg_test,
            y_ecg_train, y_ecg_val, y_ecg_test
        )
    
    # Create ensemble
    if data is not None:
        trainer.create_ensemble(y_test)
    
    # Print summary
    trainer.print_summary()
    
    # Save results
    results_df = pd.DataFrame([
        {
            'model': name,
            'accuracy': results['accuracy'],
            'auc': results.get('auc', np.nan)
        }
        for name, results in trainer.results.items()
    ])
    results_df.to_csv(trainer.results_dir / 'final_results.csv', index=False)
    
    print("\n" + "="*80)
    print("✅ TRAINING COMPLETE!")
    print(f"   Models saved to: {trainer.models_dir}")
    print(f"   Results saved to: {trainer.results_dir}")
    print("="*80)


if __name__ == '__main__':
    main()
