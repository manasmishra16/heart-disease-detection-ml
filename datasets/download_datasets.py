"""
Script to download the required datasets for the heart disease detection project.
Run this script to automatically download Cleveland and MIT-BIH datasets.
"""

import os
import urllib.request
import wfdb

def download_cleveland_dataset():
    """Download Cleveland Heart Disease dataset from UCI repository."""
    print("Downloading Cleveland Heart Disease dataset...")
    
    url = "https://archive.ics.uci.edu/ml/machine-learning-databases/heart-disease/processed.cleveland.data"
    output_path = "datasets/cleveland/processed.cleveland.data"
    
    os.makedirs("datasets/cleveland", exist_ok=True)
    
    try:
        urllib.request.urlretrieve(url, output_path)
        print(f"✓ Cleveland dataset downloaded to {output_path}")
        
        # Create a more user-friendly CSV version
        import pandas as pd
        column_names = ['age', 'sex', 'cp', 'trestbps', 'chol', 'fbs', 'restecg', 
                       'thalach', 'exang', 'oldpeak', 'slope', 'ca', 'thal', 'target']
        
        df = pd.read_csv(output_path, names=column_names, na_values='?')
        df.to_csv("datasets/cleveland/heart.csv", index=False)
        print(f"✓ Created heart.csv with proper column names")
        
    except Exception as e:
        print(f"✗ Error downloading Cleveland dataset: {e}")
        print("  Alternative: Download from https://www.kaggle.com/datasets/johnsmith88/heart-disease-dataset")

def download_mitbih_dataset():
    """Download a subset of MIT-BIH Arrhythmia Database."""
    print("\nDownloading MIT-BIH Arrhythmia dataset (subset)...")
    
    os.makedirs("datasets/mit-bih", exist_ok=True)
    
    # Download a small subset of records for quick testing
    records = ['100', '101', '102', '103', '104']
    
    try:
        for record in records:
            print(f"  Downloading record {record}...")
            wfdb.dl_files('mitdb', 'datasets/mit-bih', [f"{record}.dat", f"{record}.hea", f"{record}.atr"])
        
        print(f"✓ MIT-BIH subset downloaded ({len(records)} records)")
        print(f"  Records: {', '.join(records)}")
        
    except Exception as e:
        print(f"✗ Error downloading MIT-BIH dataset: {e}")
        print("  Make sure 'wfdb' package is installed: pip install wfdb")

if __name__ == "__main__":
    print("=" * 60)
    print("Heart Disease Detection - Dataset Downloader")
    print("=" * 60)
    
    download_cleveland_dataset()
    download_mitbih_dataset()
    
    print("\n" + "=" * 60)
    print("Dataset download complete!")
    print("=" * 60)
    print("\nNext steps:")
    print("1. Check datasets/cleveland/heart.csv for tabular data")
    print("2. Check datasets/mit-bih/ for ECG signal files")
    print("3. Run the Jupyter notebook to start analysis")
