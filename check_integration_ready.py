#!/usr/bin/env python3
"""
Quick checker to see if you're ready to integrate the Kaggle dataset
"""

import os
import sys

print("=" * 70)
print("KAGGLE INTEGRATION - READINESS CHECK")
print("=" * 70)

checks = []

# Check 1: Kaggle directory exists
kaggle_dir = 'datasets/kaggle'
if os.path.exists(kaggle_dir):
    print(f"✅ Kaggle directory exists: {kaggle_dir}")
    checks.append(True)
else:
    print(f"❌ Kaggle directory missing: {kaggle_dir}")
    print(f"   Creating it now...")
    os.makedirs(kaggle_dir, exist_ok=True)
    print(f"   ✅ Created!")
    checks.append(True)

# Check 2: Kaggle dataset downloaded
kaggle_file = 'datasets/kaggle/heart_disease.csv'
if os.path.exists(kaggle_file):
    file_size = os.path.getsize(kaggle_file) / (1024 * 1024)
    print(f"✅ Kaggle dataset found: {kaggle_file} ({file_size:.2f} MB)")
    checks.append(True)
else:
    print(f"❌ Kaggle dataset not found: {kaggle_file}")
    print(f"   Download from: https://www.kaggle.com/datasets/oktayrdeki/heart-disease")
    print(f"   Extract to: {kaggle_file}")
    checks.append(False)

# Check 3: Cleveland dataset exists
cleveland_file = 'datasets/cleveland/heart.csv'
if os.path.exists(cleveland_file):
    import pandas as pd
    df = pd.read_csv(cleveland_file)
    print(f"✅ Cleveland dataset found: {len(df)} samples")
    checks.append(True)
else:
    print(f"❌ Cleveland dataset missing: {cleveland_file}")
    checks.append(False)

# Check 4: Integration script exists
if os.path.exists('integrate_kaggle_dataset.py'):
    print(f"✅ Integration script ready")
    checks.append(True)
else:
    print(f"❌ Integration script missing")
    checks.append(False)

# Check 5: Training script exists
if os.path.exists('train_on_combined_dataset.py'):
    print(f"✅ Training script ready")
    checks.append(True)
else:
    print(f"❌ Training script missing")
    checks.append(False)

# Check 6: Required packages
try:
    import pandas
    import numpy
    import sklearn
    import tensorflow
    import xgboost
    print(f"✅ All required packages installed")
    checks.append(True)
except ImportError as e:
    print(f"❌ Missing package: {e}")
    checks.append(False)

# Summary
print("\n" + "=" * 70)
print("SUMMARY")
print("=" * 70)

passed = sum(checks)
total = len(checks)

print(f"\nChecks passed: {passed}/{total}")

if all(checks):
    print("\n🎉 ALL READY! You can now run:")
    print("   1. python integrate_kaggle_dataset.py")
    print("   2. python train_on_combined_dataset.py")
    print("\nExpected result: 92-95%+ accuracy! 🚀")
elif not checks[1]:
    print("\n📥 NEXT STEP: Download Kaggle dataset")
    print("\n   Go to: https://www.kaggle.com/datasets/oktayrdeki/heart-disease")
    print("   Click 'Download' button")
    print(f"   Extract heart_disease.csv to: {kaggle_file}")
    print("\n   Then run this check again: python check_integration_ready.py")
else:
    print("\n⚠️ Some issues found. Please resolve them before proceeding.")

print("\n" + "=" * 70)
