"""
Unified Data Loader for Heart Disease Detection
Combines Cleveland, Kaggle, and MIT-BIH datasets
"""

import pandas as pd
import numpy as np
import wfdb
from pathlib import Path
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
import warnings
warnings.filterwarnings('ignore')

class UnifiedDataLoader:
    """Load and preprocess all heart disease datasets"""
    
    def __init__(self, base_path='d:/Projects/MiniProject'):
        self.base_path = Path(base_path)
        self.cleveland_path = self.base_path / 'datasets' / 'cleveland'
        self.kaggle_path = self.base_path / 'datasets' / 'kaggle'
        self.mitbih_path = self.base_path / 'datasets' / 'mit-bih'
        
    def load_cleveland_data(self):
        """Load Cleveland Heart Disease Dataset (303 samples)"""
        print("📊 Loading Cleveland dataset...")
        df = pd.read_csv(self.cleveland_path / 'heart.csv')
        
        # Binary classification: 0 = no disease, 1 = disease
        df['target'] = (df['target'] > 0).astype(int)
        
        print(f"  ✓ Cleveland: {df.shape[0]} samples, {df.shape[1]} features")
        print(f"  ✓ Disease cases: {df['target'].sum()} ({df['target'].mean()*100:.1f}%)")
        
        return df
    
    def load_kaggle_data(self):
        """Load Kaggle Heart Disease Dataset (10,000 samples)"""
        print("\n📊 Loading Kaggle dataset...")
        df = pd.read_csv(self.kaggle_path / 'heart_disease.csv')
        
        # Rename target column to match Cleveland
        if 'Heart Disease Status' in df.columns:
            df['target'] = df['Heart Disease Status'].map({'Yes': 1, 'No': 0})
            df = df.drop('Heart Disease Status', axis=1)
        
        # Handle missing values
        df = df.fillna(df.median())
        
        print(f"  ✓ Kaggle: {df.shape[0]} samples, {df.shape[1]} features")
        print(f"  ✓ Disease cases: {df['target'].sum()} ({df['target'].mean()*100:.1f}%)")
        
        return df
    
    def align_features(self, cleveland_df, kaggle_df):
        """Align common features between datasets"""
        print("\n🔧 Aligning features across datasets...")
        
        # Encode categorical variables in Kaggle dataset
        kaggle_df = kaggle_df.copy()
        
        # Gender: Male/Female -> 1/0
        if 'Gender' in kaggle_df.columns:
            kaggle_df['Gender'] = kaggle_df['Gender'].map({'Male': 1, 'Female': 0})
        
        # Yes/No columns -> 1/0
        yes_no_cols = ['Smoking', 'Family Heart Disease', 'Diabetes', 
                       'High Blood Pressure', 'Low HDL Cholesterol', 
                       'High LDL Cholesterol']
        for col in yes_no_cols:
            if col in kaggle_df.columns:
                kaggle_df[col] = kaggle_df[col].map({'Yes': 1, 'No': 0})
        
        # Ordinal encoding for Exercise, Consumption, Stress
        ordinal_mapping = {'Low': 0, 'Medium': 1, 'High': 2}
        ordinal_cols = ['Exercise Habits', 'Alcohol Consumption', 'Stress Level', 
                        'Sugar Consumption']
        for col in ordinal_cols:
            if col in kaggle_df.columns:
                kaggle_df[col] = kaggle_df[col].map(ordinal_mapping)
        
        # Standardize column names
        kaggle_df.columns = kaggle_df.columns.str.lower().str.replace(' ', '_')
        cleveland_df.columns = cleveland_df.columns.str.lower()
        
        # Extract common numerical features
        common_features = []
        for col in cleveland_df.columns:
            if col != 'target' and col in kaggle_df.columns:
                common_features.append(col)
        
        print(f"  ✓ Common features: {len(common_features)}")
        print(f"  ✓ Features: {common_features}")
        
        return cleveland_df, kaggle_df, common_features
    
    def create_combined_dataset(self, use_all_features=True):
        """Create combined dataset from Cleveland and Kaggle"""
        print("\n" + "="*60)
        print("CREATING COMBINED DATASET")
        print("="*60)
        
        # Load datasets
        cleveland_df = self.load_cleveland_data()
        kaggle_df = self.load_kaggle_data()
        
        # Align features
        cleveland_df, kaggle_df, common_features = self.align_features(
            cleveland_df, kaggle_df
        )
        
        if use_all_features:
            # Use all available features from each dataset
            cleveland_features = [col for col in cleveland_df.columns if col != 'target']
            kaggle_features = [col for col in kaggle_df.columns if col != 'target']
            
            # Create separate datasets
            X_cleveland = cleveland_df[cleveland_features].values
            y_cleveland = cleveland_df['target'].values
            
            X_kaggle = kaggle_df[kaggle_features].values
            y_kaggle = kaggle_df['target'].values
            
            print(f"\n  ✓ Cleveland: {X_cleveland.shape}")
            print(f"  ✓ Kaggle: {X_kaggle.shape}")
            
            return {
                'cleveland': (X_cleveland, y_cleveland, cleveland_features),
                'kaggle': (X_kaggle, y_kaggle, kaggle_features),
                'cleveland_df': cleveland_df,
                'kaggle_df': kaggle_df
            }
        else:
            # Use only common features and combine
            if common_features:
                X_cleveland = cleveland_df[common_features].values
                y_cleveland = cleveland_df['target'].values
                
                X_kaggle = kaggle_df[common_features].values
                y_kaggle = kaggle_df['target'].values
                
                # Combine datasets
                X_combined = np.vstack([X_cleveland, X_kaggle])
                y_combined = np.hstack([y_cleveland, y_kaggle])
                
                print(f"\n  ✓ Combined dataset: {X_combined.shape}")
                print(f"  ✓ Total samples: {len(y_combined)}")
                print(f"  ✓ Disease cases: {y_combined.sum()} ({y_combined.mean()*100:.1f}%)")
                
                return X_combined, y_combined, common_features
            else:
                print("  ⚠️  No common features found!")
                return None
    
    def load_mitbih_ecg_data(self, num_records=5):
        """Load MIT-BIH ECG signals"""
        print("\n📊 Loading MIT-BIH ECG data...")
        
        records = []
        labels = []
        
        record_ids = ['100', '101', '102', '103', '104'][:num_records]
        
        for record_id in record_ids:
            try:
                # Read ECG signal
                record = wfdb.rdrecord(
                    str(self.mitbih_path / record_id),
                    channels=[0]
                )
                
                # Read annotations
                annotation = wfdb.rdann(
                    str(self.mitbih_path / record_id),
                    'atr'
                )
                
                # Extract signal and process
                signal = record.p_signal[:, 0]
                
                # Segment into 10-second windows (3600 samples at 360 Hz)
                segment_length = 3600
                num_segments = len(signal) // segment_length
                
                for i in range(num_segments):
                    start = i * segment_length
                    end = start + segment_length
                    segment = signal[start:end]
                    
                    # Simple labeling: presence of annotations in segment
                    seg_annotations = [
                        ann for ann in annotation.sample 
                        if start <= ann < end
                    ]
                    label = 1 if len(seg_annotations) > 5 else 0
                    
                    records.append(segment)
                    labels.append(label)
                
                print(f"  ✓ Record {record_id}: {num_segments} segments")
                
            except Exception as e:
                print(f"  ✗ Error loading record {record_id}: {e}")
                continue
        
        if records:
            X_ecg = np.array(records)
            y_ecg = np.array(labels)
            
            print(f"\n  ✓ Total ECG segments: {X_ecg.shape}")
            print(f"  ✓ Abnormal cases: {y_ecg.sum()} ({y_ecg.mean()*100:.1f}%)")
            
            # Reshape for 1D-CNN: (samples, timesteps, channels)
            X_ecg = X_ecg.reshape(X_ecg.shape[0], X_ecg.shape[1], 1)
            
            return X_ecg, y_ecg
        
        return None, None
    
    def prepare_train_test_split(self, X, y, test_size=0.2, val_size=0.1):
        """Split data into train, validation, and test sets"""
        print("\n🔧 Splitting data...")
        
        # First split: train+val vs test
        X_temp, X_test, y_temp, y_test = train_test_split(
            X, y, test_size=test_size, stratify=y, random_state=42
        )
        
        # Second split: train vs val
        val_size_adjusted = val_size / (1 - test_size)
        X_train, X_val, y_train, y_val = train_test_split(
            X_temp, y_temp, test_size=val_size_adjusted, 
            stratify=y_temp, random_state=42
        )
        
        print(f"  ✓ Train: {X_train.shape[0]} samples ({y_train.mean()*100:.1f}% disease)")
        print(f"  ✓ Val:   {X_val.shape[0]} samples ({y_val.mean()*100:.1f}% disease)")
        print(f"  ✓ Test:  {X_test.shape[0]} samples ({y_test.mean()*100:.1f}% disease)")
        
        return X_train, X_val, X_test, y_train, y_val, y_test
    
    def scale_data(self, X_train, X_val, X_test):
        """Scale features using StandardScaler"""
        print("\n🔧 Scaling features...")
        
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_val_scaled = scaler.transform(X_val)
        X_test_scaled = scaler.transform(X_test)
        
        print(f"  ✓ Scaler fitted on train set")
        print(f"  ✓ Mean: {scaler.mean_[:3]} ...")
        print(f"  ✓ Std:  {scaler.scale_[:3]} ...")
        
        return X_train_scaled, X_val_scaled, X_test_scaled, scaler


def main():
    """Demo usage"""
    loader = UnifiedDataLoader()
    
    # Load combined clinical data
    data = loader.create_combined_dataset(use_all_features=True)
    
    # Load ECG data
    X_ecg, y_ecg = loader.load_mitbih_ecg_data()
    
    print("\n" + "="*60)
    print("DATA LOADING COMPLETE!")
    print("="*60)


if __name__ == '__main__':
    main()
