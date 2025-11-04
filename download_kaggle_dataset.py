#!/usr/bin/env python3
"""
Automatic Kaggle Dataset Downloader
Downloads the heart disease dataset using Kaggle API
"""

import os
import sys
import json
from pathlib import Path

print("=" * 70)
print("KAGGLE DATASET AUTO-DOWNLOADER")
print("=" * 70)

# Check if kaggle credentials exist
kaggle_dir = Path.home() / '.kaggle'
kaggle_json = kaggle_dir / 'kaggle.json'

if not kaggle_json.exists():
    print("\n❌ Kaggle API credentials not found!")
    print("\nTo set up Kaggle API:")
    print("1. Go to: https://www.kaggle.com/settings/account")
    print("2. Scroll to 'API' section")
    print("3. Click 'Create New Token'")
    print("4. This will download 'kaggle.json'")
    print(f"5. Move it to: {kaggle_json}")
    print("\nOR enter your credentials now:\n")
    
    username = input("Enter your Kaggle username (or press Enter to skip): ").strip()
    
    if username:
        key = input("Enter your Kaggle API key: ").strip()
        
        # Create .kaggle directory
        kaggle_dir.mkdir(parents=True, exist_ok=True)
        
        # Create kaggle.json
        credentials = {
            "username": username,
            "key": key
        }
        
        with open(kaggle_json, 'w') as f:
            json.dump(credentials, f)
        
        # Set permissions (important for security)
        if os.name != 'nt':  # Unix-like systems
            os.chmod(kaggle_json, 0o600)
        
        print(f"\n✅ Credentials saved to: {kaggle_json}")
    else:
        print("\n⚠️ Skipping download. Please set up credentials manually.")
        print("\nQuick setup:")
        print(f"  1. Create directory: {kaggle_dir}")
        print(f"  2. Create file: {kaggle_json}")
        print('  3. Content: {"username":"YOUR_USERNAME","key":"YOUR_API_KEY"}')
        sys.exit(1)

print(f"\n✅ Found Kaggle credentials at: {kaggle_json}")

# Try to download
try:
    print("\n📥 Downloading heart disease dataset from Kaggle...")
    print("   Dataset: oktayrdeki/heart-disease")
    print("   Size: ~1.5 MB")
    
    import subprocess
    
    # Create output directory
    output_dir = 'datasets/kaggle'
    os.makedirs(output_dir, exist_ok=True)
    
    # Download using kaggle CLI
    result = subprocess.run(
        ['kaggle', 'datasets', 'download', '-d', 'oktayrdeki/heart-disease', 
         '-p', output_dir, '--unzip'],
        capture_output=True,
        text=True
    )
    
    if result.returncode == 0:
        print("\n✅ SUCCESS! Dataset downloaded!")
        print(f"   Location: {output_dir}/heart_disease.csv")
        
        # Verify file exists
        if os.path.exists(f'{output_dir}/heart_disease.csv'):
            file_size = os.path.getsize(f'{output_dir}/heart_disease.csv') / (1024 * 1024)
            print(f"   File size: {file_size:.2f} MB")
            
            # Count rows
            import pandas as pd
            df = pd.read_csv(f'{output_dir}/heart_disease.csv')
            print(f"   Samples: {len(df)}")
            print(f"   Features: {len(df.columns)}")
            
            print("\n" + "=" * 70)
            print("NEXT STEPS")
            print("=" * 70)
            print("\n1. Integrate datasets:")
            print("   python integrate_kaggle_dataset.py")
            print("\n2. Train on combined data:")
            print("   python train_on_combined_dataset.py")
            print("\n3. Expected result: 92-95%+ accuracy! 🎉")
            
        else:
            print("\n⚠️ Download succeeded but file not found. Check output directory.")
    else:
        print(f"\n❌ Download failed!")
        print(f"Error: {result.stderr}")
        
        if "401" in result.stderr or "403" in result.stderr:
            print("\n⚠️ Authentication failed. Your credentials may be incorrect.")
            print("   Please check your Kaggle username and API key.")
        elif "404" in result.stderr:
            print("\n⚠️ Dataset not found. URL may have changed.")
        else:
            print("\n💡 Alternative: Download manually from:")
            print("   https://www.kaggle.com/datasets/oktayrdeki/heart-disease")
            print(f"   Extract to: {output_dir}/heart_disease.csv")
        
        sys.exit(1)
        
except Exception as e:
    print(f"\n❌ Error during download: {e}")
    print("\n💡 Try downloading manually from:")
    print("   https://www.kaggle.com/datasets/oktayrdeki/heart-disease")
    print(f"   Extract to: {output_dir}/heart_disease.csv")
    sys.exit(1)

print("\n✅ Auto-download complete!")
