import os
import sys

print("=" * 60)
print("DAY 3 DELIVERABLES TEST SUITE")
print("=" * 60)

tests_passed = 0
tests_failed = 0

# Test 1: Results document
print("\nTEST 1: Results Documentation")
if os.path.exists("results_baseline.md"):
    size = os.path.getsize("results_baseline.md")
    print(f"PASS: results_baseline.md ({size:,} bytes)")
    tests_passed += 1
else:
    print("FAIL: results_baseline.md not found")
    tests_failed += 1

# Test 2: ML Models
print("\nTEST 2: Machine Learning Models")
models = ["logistic_regression.pkl", "random_forest.pkl", "xgboost.pkl", "svm.pkl", "knn.pkl"]
for model in models:
    path = f"models/{model}"
    if os.path.exists(path):
        size = os.path.getsize(path)
        print(f"PASS: {path} ({size:,} bytes)")
        tests_passed += 1
    else:
        print(f"FAIL: {path} not found")
        tests_failed += 1

# Test 3: Visualizations
print("\nTEST 3: Visualizations")
viz_files = ["confusion_matrices.png", "roc_curves.png"]
for viz in viz_files:
    path = f"results/{viz}"
    if os.path.exists(path):
        size = os.path.getsize(path)
        print(f"PASS: {path} ({size:,} bytes)")
        tests_passed += 1
    else:
        print(f"FAIL: {path} not found")
        tests_failed += 1

# Test 4: CNN Model (optional)
print("\nTEST 4: Deep Learning Model (Optional)")
if os.path.exists("models/cnn_ecg_baseline.keras"):
    size = os.path.getsize("models/cnn_ecg_baseline.keras")
    print(f"PASS: models/cnn_ecg_baseline.keras ({size:,} bytes)")
    tests_passed += 1
else:
    print("SKIP: CNN model not found (optional)")
    tests_passed += 1

# Summary
print("\n" + "=" * 60)
print("SUMMARY")
print("=" * 60)
print(f"Passed: {tests_passed}")
print(f"Failed: {tests_failed}")

if tests_failed == 0:
    print("\nSTATUS: ALL TESTS PASSED")
    sys.exit(0)
else:
    print(f"\nSTATUS: {tests_failed} TEST(S) FAILED")
    sys.exit(1)
