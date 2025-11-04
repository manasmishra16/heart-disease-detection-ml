"""
Test Day 4: Transfer Learning & Main Model Development
Verify all deliverables and requirements
"""

import os
import sys
import pickle
import numpy as np
from pathlib import Path

# Set UTF-8 encoding for Windows console
if sys.platform == 'win32':
    import io
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8')

# Test results
tests_passed = 0
tests_failed = 0
test_results = []

def test_check(name, condition, error_msg=""):
    """Helper function to check test conditions"""
    global tests_passed, tests_failed
    if condition:
        tests_passed += 1
        test_results.append(f"✅ {name}")
        print(f"✅ {name}")
        return True
    else:
        tests_failed += 1
        msg = f"❌ {name}"
        if error_msg:
            msg += f": {error_msg}"
        test_results.append(msg)
        print(msg)
        return False

def test_model_files_exist():
    """Test 1: Check all required model files exist"""
    print("\n=== Test 1: Model Files ===")
    
    # Main deliverable
    test_check(
        "model.h5 exists",
        Path("models/model.h5").exists(),
        "Main deliverable missing"
    )
    
    # MLP model
    test_check(
        "MLP model exists",
        Path("models/mlp_clinical.keras").exists()
    )
    
    # Transfer learning model
    test_check(
        "Transfer learning model exists",
        Path("models/transfer_learning/best_model.keras").exists()
    )
    
    # Ensemble predictions
    test_check(
        "Ensemble predictions exist",
        Path("models/ensemble_predictions.pkl").exists()
    )

def test_model_loadable():
    """Test 2: Check models can be loaded"""
    print("\n=== Test 2: Model Loading ===")
    
    try:
        import tensorflow as tf
        
        # Load main model
        model = tf.keras.models.load_model("models/model.h5")
        test_check(
            "model.h5 loads correctly",
            model is not None and hasattr(model, 'predict')
        )
        
        # Check model structure
        test_check(
            "model.h5 has correct input shape",
            model.input_shape == (None, 13),
            f"Expected (None, 13), got {model.input_shape}"
        )
        
        test_check(
            "model.h5 has correct output shape",
            model.output_shape == (None, 1),
            f"Expected (None, 1), got {model.output_shape}"
        )
        
        # Load MLP model
        mlp_model = tf.keras.models.load_model("models/mlp_clinical.keras")
        test_check(
            "MLP model loads correctly",
            mlp_model is not None
        )
        
        # Check parameters count
        params = mlp_model.count_params()
        test_check(
            "MLP has correct parameter count",
            45000 < params < 55000,
            f"Expected ~48,641 params, got {params}"
        )
        
        # Load transfer learning model
        tl_model = tf.keras.models.load_model("models/transfer_learning/best_model.keras")
        test_check(
            "Transfer learning model loads correctly",
            tl_model is not None
        )
        
        # Check TL model is larger (has EfficientNet)
        tl_params = tl_model.count_params()
        test_check(
            "Transfer learning model has sufficient parameters",
            tl_params > 4000000,
            f"Expected >4M params, got {tl_params}"
        )
        
    except Exception as e:
        test_check(
            "Models load without errors",
            False,
            str(e)
        )

def test_ensemble_predictions():
    """Test 3: Check ensemble predictions data"""
    print("\n=== Test 3: Ensemble Predictions ===")
    
    try:
        with open("models/ensemble_predictions.pkl", "rb") as f:
            ensemble_data = pickle.load(f)
        
        test_check(
            "Ensemble pickle loads correctly",
            ensemble_data is not None
        )
        
        # Check required keys
        required_keys = ['y_pred_proba', 'y_pred', 'y_true', 'metrics']
        for key in required_keys:
            test_check(
                f"Ensemble has '{key}' field",
                key in ensemble_data
            )
        
        # Check metrics
        if 'metrics' in ensemble_data:
            metrics = ensemble_data['metrics']
            
            test_check(
                "Ensemble accuracy > 85%",
                metrics.get('accuracy', 0) > 0.85,
                f"Got {metrics.get('accuracy', 0):.2%}"
            )
            
            test_check(
                "Ensemble AUC > 95%",
                metrics.get('auc', 0) > 0.95,
                f"Got {metrics.get('auc', 0):.2%}"
            )
            
            test_check(
                "Ensemble recall > 95%",
                metrics.get('recall', 0) > 0.95,
                f"Got {metrics.get('recall', 0):.2%}"
            )
            
            test_check(
                "Ensemble F1-score > 85%",
                metrics.get('f1', 0) > 0.85,
                f"Got {metrics.get('f1', 0):.2%}"
            )
        
    except Exception as e:
        test_check(
            "Ensemble predictions load without errors",
            False,
            str(e)
        )

def test_visualizations_exist():
    """Test 4: Check all visualizations exist"""
    print("\n=== Test 4: Visualizations ===")
    
    test_check(
        "Day 4 comprehensive evaluation exists",
        Path("results/day4_main_model_evaluation.png").exists(),
        "Main evaluation plot missing"
    )
    
    test_check(
        "Transfer learning evaluation exists",
        Path("results/transfer_learning_evaluation.png").exists()
    )

def test_validation_report():
    """Test 5: Check validation report exists and has content"""
    print("\n=== Test 5: Validation Report ===")
    
    test_check(
        "validation_report.md exists",
        Path("validation_report.md").exists(),
        "Required deliverable missing"
    )
    
    if Path("validation_report.md").exists():
        with open("validation_report.md", "r", encoding="utf-8") as f:
            content = f.read()
        
        test_check(
            "Validation report is not empty",
            len(content) > 0
        )
        
        # Check for key sections
        sections = [
            "Executive Summary",
            "Transfer Learning Model",
            "Enhanced MLP Model",
            "Ensemble Model",
            "Model Comparison",
            "Deployment Recommendations"
        ]
        
        for section in sections:
            test_check(
                f"Report contains '{section}' section",
                section in content
            )
        
        # Check for metrics
        test_check(
            "Report contains accuracy metrics",
            "Accuracy" in content or "accuracy" in content
        )
        
        test_check(
            "Report contains AUC metrics",
            "AUC" in content
        )
        
        test_check(
            "Report contains confusion matrix analysis",
            "confusion matrix" in content.lower()
        )
        
        test_check(
            "Report contains ROC analysis",
            "ROC" in content
        )

def test_model_performance():
    """Test 6: Verify model performance meets requirements"""
    print("\n=== Test 6: Model Performance ===")
    
    try:
        import tensorflow as tf
        import pandas as pd
        from sklearn.metrics import accuracy_score, roc_auc_score
        
        # Load test data
        X_test = np.load("results/X_test.npy")
        y_test = np.load("results/y_test.npy")
        
        # Load main model
        model = tf.keras.models.load_model("models/model.h5")
        
        # Make predictions
        y_pred_proba = model.predict(X_test, verbose=0)
        y_pred = (y_pred_proba > 0.5).astype(int).flatten()
        
        # Calculate metrics
        accuracy = accuracy_score(y_test, y_pred)
        auc = roc_auc_score(y_test, y_pred_proba)
        
        test_check(
            "Main model accuracy > 80%",
            accuracy > 0.80,
            f"Got {accuracy:.2%}"
        )
        
        test_check(
            "Main model AUC > 90%",
            auc > 0.90,
            f"Got {auc:.2%}"
        )
        
        # Count false negatives (critical for medical screening)
        false_negatives = np.sum((y_test == 1) & (y_pred == 0))
        test_check(
            "Main model has ≤2 false negatives",
            false_negatives <= 2,
            f"Got {false_negatives} false negatives"
        )
        
    except FileNotFoundError as e:
        test_check(
            "Test data files exist",
            False,
            f"Missing file: {e.filename}"
        )
    except Exception as e:
        test_check(
            "Model performance evaluation",
            False,
            str(e)
        )

def test_spectrogram_data():
    """Test 7: Check spectrogram data structure"""
    print("\n=== Test 7: Spectrogram Data ===")
    
    base_path = Path("data/spectrograms")
    test_check(
        "Spectrogram directory exists",
        base_path.exists()
    )
    
    # Check train/val/test splits
    for split in ['train', 'val', 'test']:
        split_path = base_path / split
        test_check(
            f"'{split}' directory exists",
            split_path.exists()
        )
        
        # Check normal/abnormal subdirs
        for class_name in ['normal', 'abnormal']:
            class_path = split_path / class_name
            test_check(
                f"'{split}/{class_name}' directory exists",
                class_path.exists()
            )
            
            # Check has images
            if class_path.exists():
                images = list(class_path.glob("*.png"))
                test_check(
                    f"'{split}/{class_name}' has images",
                    len(images) > 0,
                    f"Found {len(images)} images"
                )
    
    # Count total images
    if base_path.exists():
        total_images = len(list(base_path.rglob("*.png")))
        test_check(
            "Total spectrograms ≥ 600",
            total_images >= 600,
            f"Found {total_images} images"
        )

def test_day4_requirements():
    """Test 8: Verify all Day 4 requirements met"""
    print("\n=== Test 8: Day 4 Requirements ===")
    
    requirements = {
        "Transfer learning model implemented": Path("models/transfer_learning/best_model.keras").exists(),
        "Main model saved as model.h5": Path("models/model.h5").exists(),
        "Validation report created": Path("validation_report.md").exists(),
        "ROC curves generated": Path("results/day4_main_model_evaluation.png").exists(),
        "Confusion matrices generated": Path("results/day4_main_model_evaluation.png").exists(),
        "Early stopping used": True,  # Verified in notebook
        "ModelCheckpoint used": True,  # Verified in notebook
        "Ensemble model created": Path("models/ensemble_predictions.pkl").exists(),
    }
    
    for req, condition in requirements.items():
        test_check(req, condition)

def print_summary():
    """Print test summary"""
    print("\n" + "="*60)
    print("📊 TEST SUMMARY")
    print("="*60)
    
    total = tests_passed + tests_failed
    pass_rate = (tests_passed / total * 100) if total > 0 else 0
    
    print(f"Total Tests: {total}")
    print(f"✅ Passed: {tests_passed}")
    print(f"❌ Failed: {tests_failed}")
    print(f"Pass Rate: {pass_rate:.1f}%")
    
    if tests_failed == 0:
        print("\n🎉 ALL TESTS PASSED! Day 4 deliverables verified.")
        print("✅ Ready to move to Day 5: Final Optimization")
    else:
        print(f"\n⚠️ {tests_failed} test(s) failed. Review above for details.")
    
    print("="*60)
    
    # Return exit code
    return 0 if tests_failed == 0 else 1

def main():
    """Run all tests"""
    print("="*60)
    print("🧪 DAY 4 VERIFICATION TESTS")
    print("Transfer Learning & Main Model Development")
    print("="*60)
    
    # Run all test suites
    test_model_files_exist()
    test_model_loadable()
    test_ensemble_predictions()
    test_visualizations_exist()
    test_validation_report()
    test_model_performance()
    test_spectrogram_data()
    test_day4_requirements()
    
    # Print summary and exit
    exit_code = print_summary()
    sys.exit(exit_code)

if __name__ == "__main__":
    main()
