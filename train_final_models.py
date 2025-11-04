"""
Master Script - Final Project Execution
Run this to train all models and get final results
"""

import sys
import os
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / 'src'))

import warnings
warnings.filterwarnings('ignore')

import tensorflow as tf
# Set memory growth to avoid OOM errors
physical_devices = tf.config.list_physical_devices('GPU')
if physical_devices:
    tf.config.experimental.set_memory_growth(physical_devices[0], True)

from data_processing.unified_data_loader import UnifiedDataLoader
from training.train_all_models import ComprehensiveTrainer
import joblib
import numpy as np
import pandas as pd


def print_banner(text):
    """Print formatted banner"""
    print("\n" + "="*80)
    print(" "*((80-len(text))//2) + text)
    print("="*80 + "\n")


def main():
    print_banner("HEART DISEASE DETECTION - FINAL TRAINING")
    print("🚀 Starting comprehensive training pipeline...")
    print("   This will train:")
    print("   - Enhanced MLP (Clinical)")
    print("   - Random Forest (Clinical)")
    print("   - Gradient Boosting (Clinical)")
    print("   - Deep CNN (ECG)")
    print("   - CNN-LSTM Hybrid (ECG)")
    print("   - Bidirectional LSTM (ECG)")
    print("   - Bidirectional GRU (ECG)")
    print("   - Weighted Ensemble (All models)")
    
    # Initialize
    base_path = Path(__file__).parent
    trainer = ComprehensiveTrainer(str(base_path))
    loader = UnifiedDataLoader(str(base_path))
    
    # ========================================================================
    # PART 1: CLINICAL DATA (Cleveland + Kaggle)
    # ========================================================================
    print_banner("PART 1: CLINICAL DATA TRAINING")
    
    print("📊 Loading and combining Cleveland + Kaggle datasets...")
    data = loader.create_combined_dataset(use_all_features=False)
    
    if data is None:
        print("❌ Failed to load clinical data!")
        return
    
    X, y, features = data
    print(f"✅ Combined dataset: {X.shape[0]} samples, {X.shape[1]} features")
    
    # Split and scale
    X_train, X_val, X_test, y_train, y_val, y_test = loader.prepare_train_test_split(X, y)
    X_train_scaled, X_val_scaled, X_test_scaled, scaler = loader.scale_data(
        X_train, X_val, X_test
    )
    
    # Save scaler
    joblib.dump(scaler, base_path / 'models' / 'scaler_final.pkl')
    print("💾 Saved scaler: models/scaler_final.pkl")
    
    # Train clinical models
    print("\n🎓 Training clinical models (MLP, RF, GB)...")
    mlp, rf, gb = trainer.train_clinical_models(
        X_train_scaled, X_val_scaled, X_test_scaled,
        y_train, y_val, y_test
    )
    
    # ========================================================================
    # PART 2: ECG DATA (MIT-BIH)
    # ========================================================================
    print_banner("PART 2: ECG DATA TRAINING")
    
    print("📊 Loading MIT-BIH ECG data...")
    X_ecg, y_ecg = loader.load_mitbih_ecg_data(num_records=5)
    
    if X_ecg is not None and len(X_ecg) > 50:
        print(f"✅ ECG dataset: {X_ecg.shape[0]} segments")
        
        # Split ECG data
        X_ecg_train, X_ecg_val, X_ecg_test, y_ecg_train, y_ecg_val, y_ecg_test = \
            loader.prepare_train_test_split(X_ecg, y_ecg)
        
        # Train ECG models
        print("\n🎓 Training ECG models (CNN, CNN-LSTM, LSTM, GRU)...")
        cnn, cnn_lstm, lstm, gru = trainer.train_ecg_models(
            X_ecg_train, X_ecg_val, X_ecg_test,
            y_ecg_train, y_ecg_val, y_ecg_test
        )
    else:
        print("⚠️  Insufficient ECG data for training")
    
    # ========================================================================
    # PART 3: ENSEMBLE
    # ========================================================================
    print_banner("PART 3: ENSEMBLE CREATION")
    
    print("🔧 Creating weighted ensemble from all models...")
    ensemble_pred = trainer.create_ensemble(y_test)
    
    # ========================================================================
    # PART 4: RESULTS
    # ========================================================================
    print_banner("FINAL RESULTS")
    
    trainer.print_summary()
    
    # Save detailed results
    results_df = pd.DataFrame([
        {
            'Model': name.replace('_', ' ').title(),
            'Accuracy (%)': f"{results['accuracy']*100:.2f}",
            'AUC (%)': f"{results.get('auc', 0)*100:.2f}" if 'auc' in results else 'N/A'
        }
        for name, results in sorted(
            trainer.results.items(), 
            key=lambda x: x[1]['accuracy'], 
            reverse=True
        )
    ])
    
    results_path = base_path / 'results' / 'final_model_comparison.csv'
    results_df.to_csv(results_path, index=False)
    print(f"\n💾 Results saved to: {results_path}")
    
    # Save predictions
    predictions_df = pd.DataFrame({
        'True_Label': y_test,
        **{name: results['predictions'] for name, results in trainer.results.items()}
    })
    predictions_path = base_path / 'results' / 'all_predictions.csv'
    predictions_df.to_csv(predictions_path, index=False)
    print(f"💾 Predictions saved to: {predictions_path}")
    
    # ========================================================================
    # COMPLETION
    # ========================================================================
    print_banner("TRAINING COMPLETED SUCCESSFULLY!")
    
    print("📊 Models Summary:")
    print(f"   Total models trained: {len(trainer.results)}")
    print(f"   Best accuracy: {max(r['accuracy'] for r in trainer.results.values())*100:.2f}%")
    print(f"   Models directory: {base_path / 'models'}")
    print(f"   Results directory: {base_path / 'results'}")
    
    print("\n🎉 All models trained and saved successfully!")
    print("   You can now use these models for prediction and deployment.")
    print("\n📝 Next steps:")
    print("   1. Review results in: results/final_model_comparison.csv")
    print("   2. Test models with: python test_final_models.py")
    print("   3. Deploy with: cd app && streamlit run demo.py")
    
    print("\n" + "="*80)


if __name__ == '__main__':
    try:
        main()
    except KeyboardInterrupt:
        print("\n\n⚠️  Training interrupted by user")
    except Exception as e:
        print(f"\n\n❌ Error during training: {e}")
        import traceback
        traceback.print_exc()
