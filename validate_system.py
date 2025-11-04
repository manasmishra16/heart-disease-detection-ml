"""
Quick Validation Script
Tests the new system setup
"""

import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / 'src'))

import warnings
warnings.filterwarnings('ignore')


def test_imports():
    """Test if all modules can be imported"""
    print("="*60)
    print("TESTING IMPORTS")
    print("="*60)
    
    try:
        print("\n✓ Testing TensorFlow...")
        import tensorflow as tf
        print(f"  TensorFlow version: {tf.__version__}")
        
        print("\n✓ Testing data loader...")
        from data_processing.unified_data_loader import UnifiedDataLoader
        print("  ✅ Data loader imported successfully")
        
        print("\n✓ Testing models...")
        from models.deep_learning_models import (
            create_deep_cnn_model,
            create_cnn_lstm_model,
            create_lstm_model,
            create_enhanced_mlp_model
        )
        print("  ✅ Model definitions imported successfully")
        
        print("\n✓ Testing training module...")
        from training.train_all_models import ComprehensiveTrainer
        print("  ✅ Training module imported successfully")
        
        print("\n✓ Testing scikit-learn...")
        from sklearn.ensemble import RandomForestClassifier
        from sklearn.preprocessing import StandardScaler
        print("  ✅ Scikit-learn imported successfully")
        
        print("\n✓ Testing wfdb (ECG processing)...")
        import wfdb
        print("  ✅ WFDB imported successfully")
        
        print("\n" + "="*60)
        print("✅ ALL IMPORTS SUCCESSFUL")
        print("="*60)
        return True
        
    except Exception as e:
        print(f"\n❌ Import failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_data_loading():
    """Test data loading functionality"""
    print("\n" + "="*60)
    print("TESTING DATA LOADING")
    print("="*60)
    
    try:
        from data_processing.unified_data_loader import UnifiedDataLoader
        
        loader = UnifiedDataLoader()
        
        print("\n✓ Testing Cleveland data...")
        cleveland = loader.load_cleveland_data()
        print(f"  ✅ Loaded {len(cleveland)} samples")
        
        print("\n✓ Testing Kaggle data...")
        kaggle = loader.load_kaggle_data()
        print(f"  ✅ Loaded {len(kaggle)} samples")
        
        print("\n✓ Testing combined dataset...")
        data = loader.create_combined_dataset(use_all_features=False)
        if data is not None:
            X, y, features = data
            print(f"  ✅ Combined: {X.shape[0]} samples, {X.shape[1]} features")
        
        print("\n✓ Testing ECG data...")
        X_ecg, y_ecg = loader.load_mitbih_ecg_data(num_records=2)
        if X_ecg is not None:
            print(f"  ✅ ECG: {X_ecg.shape[0]} segments")
        else:
            print("  ⚠️ ECG data not available (optional)")
        
        print("\n" + "="*60)
        print("✅ DATA LOADING SUCCESSFUL")
        print("="*60)
        return True
        
    except Exception as e:
        print(f"\n❌ Data loading failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_model_creation():
    """Test model creation"""
    print("\n" + "="*60)
    print("TESTING MODEL CREATION")
    print("="*60)
    
    try:
        from models.deep_learning_models import (
            create_deep_cnn_model,
            create_cnn_lstm_model,
            create_lstm_model,
            create_gru_model,
            create_enhanced_mlp_model
        )
        
        print("\n✓ Creating Deep CNN...")
        cnn = create_deep_cnn_model(input_shape=(1000, 1))
        print(f"  ✅ Parameters: {cnn.count_params():,}")
        
        print("\n✓ Creating CNN-LSTM...")
        cnn_lstm = create_cnn_lstm_model(input_shape=(1000, 1))
        print(f"  ✅ Parameters: {cnn_lstm.count_params():,}")
        
        print("\n✓ Creating LSTM...")
        lstm = create_lstm_model(input_shape=(1000, 1))
        print(f"  ✅ Parameters: {lstm.count_params():,}")
        
        print("\n✓ Creating GRU...")
        gru = create_gru_model(input_shape=(1000, 1))
        print(f"  ✅ Parameters: {gru.count_params():,}")
        
        print("\n✓ Creating Enhanced MLP...")
        mlp = create_enhanced_mlp_model(input_shape=(13,))
        print(f"  ✅ Parameters: {mlp.count_params():,}")
        
        print("\n" + "="*60)
        print("✅ MODEL CREATION SUCCESSFUL")
        print("="*60)
        return True
        
    except Exception as e:
        print(f"\n❌ Model creation failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_directory_structure():
    """Test if directory structure is correct"""
    print("\n" + "="*60)
    print("TESTING DIRECTORY STRUCTURE")
    print("="*60)
    
    base_dir = Path(__file__).parent
    
    required_dirs = [
        'src',
        'src/data_processing',
        'src/models',
        'src/training',
        'datasets',
        'datasets/cleveland',
        'datasets/kaggle',
        'datasets/mit-bih',
        'models',
        'results',
        'app',
        'configs',
        'tests'
    ]
    
    all_exist = True
    for dir_path in required_dirs:
        full_path = base_dir / dir_path
        if full_path.exists():
            print(f"  ✅ {dir_path}")
        else:
            print(f"  ❌ {dir_path} - MISSING")
            all_exist = False
    
    if all_exist:
        print("\n" + "="*60)
        print("✅ DIRECTORY STRUCTURE CORRECT")
        print("="*60)
    else:
        print("\n" + "="*60)
        print("⚠️ SOME DIRECTORIES MISSING")
        print("="*60)
    
    return all_exist


def test_datasets_exist():
    """Test if required datasets exist"""
    print("\n" + "="*60)
    print("TESTING DATASET FILES")
    print("="*60)
    
    base_dir = Path(__file__).parent
    
    datasets = {
        'Cleveland': base_dir / 'datasets' / 'cleveland' / 'heart.csv',
        'Kaggle': base_dir / 'datasets' / 'kaggle' / 'heart_disease.csv',
        'MIT-BIH (100)': base_dir / 'datasets' / 'mit-bih' / '100.dat',
    }
    
    all_exist = True
    for name, path in datasets.items():
        if path.exists():
            size = path.stat().st_size / 1024  # KB
            print(f"  ✅ {name}: {size:.1f} KB")
        else:
            print(f"  ❌ {name} - NOT FOUND")
            all_exist = False
    
    if all_exist:
        print("\n" + "="*60)
        print("✅ ALL DATASETS AVAILABLE")
        print("="*60)
    else:
        print("\n" + "="*60)
        print("⚠️ SOME DATASETS MISSING")
        print("="*60)
    
    return all_exist


def main():
    """Run all validation tests"""
    print("\n" + "="*80)
    print(" "*20 + "SYSTEM VALIDATION")
    print("="*80)
    
    results = {}
    
    # Test 1: Imports
    results['imports'] = test_imports()
    
    # Test 2: Directory structure
    results['directories'] = test_directory_structure()
    
    # Test 3: Datasets
    results['datasets'] = test_datasets_exist()
    
    # Test 4: Data loading
    if results['imports'] and results['datasets']:
        results['data_loading'] = test_data_loading()
    else:
        print("\n⚠️ Skipping data loading test (dependencies not met)")
        results['data_loading'] = False
    
    # Test 5: Model creation
    if results['imports']:
        results['model_creation'] = test_model_creation()
    else:
        print("\n⚠️ Skipping model creation test (imports failed)")
        results['model_creation'] = False
    
    # Summary
    print("\n" + "="*80)
    print(" "*25 + "VALIDATION SUMMARY")
    print("="*80)
    
    for test_name, passed in results.items():
        status = "✅ PASS" if passed else "❌ FAIL"
        print(f"  {test_name.replace('_', ' ').title():<30} {status}")
    
    all_passed = all(results.values())
    
    print("\n" + "="*80)
    if all_passed:
        print("🎉 ALL TESTS PASSED - SYSTEM READY")
        print("\nNext steps:")
        print("  1. Train models: python train_final_models.py")
        print("  2. Launch demo: streamlit run app/demo_updated.py")
    else:
        print("⚠️ SOME TESTS FAILED - CHECK SETUP")
        print("\nTroubleshooting:")
        print("  1. Ensure virtual environment is activated")
        print("  2. Install dependencies: pip install -r requirements.txt")
        print("  3. Verify datasets are in correct locations")
    print("="*80)
    
    return all_passed


if __name__ == '__main__':
    success = main()
    sys.exit(0 if success else 1)
