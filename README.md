# Heart Disease Detection - Mini Project

## Project Overview
Machine learning and deep learning models for heart disease detection using tabular clinical data and ECG signals.

## Project Structure
```
MiniProject/
├── README.md                           # Project documentation
├── completion_log.md                   # Daily progress tracking
├── heart_disease_detection.ipynb       # Main Jupyter notebook
│
├── datasets/                           # Dataset storage
│   ├── cleveland/                      # Cleveland/UCI heart disease data
│   └── mit-bih/                        # MIT-BIH ECG recordings
│
├── models/                             # Saved model files
│   ├── baseline/                       # Traditional ML models
│   └── deep_learning/                  # Neural network models
│
├── results/                            # Output files
│   ├── figures/                        # Plots and visualizations
│   ├── metrics/                        # Performance metrics
│   └── reports/                        # Generated reports
│
├── scripts/                            # Utility scripts
│   ├── data_preprocessing.py           # Data preprocessing functions
│   ├── model_training.py               # Model training utilities
│   └── evaluation.py                   # Evaluation utilities
│
├── tests/                              # Test scripts
│   ├── test_day1.py                    # Day 1 verification
│   ├── test_day2.py                    # Day 2 verification
│   ├── test_day3.py                    # Day 3 verification
│   └── test_day4.py                    # Day 4 verification
│
└── docs/                               # Additional documentation
    └── project_requirements.md         # Detailed requirements
```

## Datasets

### 1. Cleveland/UCI Heart Disease Dataset
- **Type:** Tabular clinical features
- **Features:** Age, sex, chest pain type, blood pressure, cholesterol, etc.
- **Target:** Binary classification (disease presence)
- **Source:** UCI ML Repository / Kaggle

### 2. MIT-BIH Arrhythmia Database
- **Type:** ECG signal data
- **Sampling Rate:** 360 Hz
- **Use Case:** Arrhythmia classification from 5-10s segments
- **Source:** PhysioNet

## Setup Instructions

### 1. Environment Setup
```bash
# Install required packages
pip install tensorflow scikit-learn pandas numpy matplotlib scipy wfdb librosa
```

### 2. Run Tests
```bash
# Test Day 1 setup
python tests/test_day1.py

# Test Day 2 preprocessing
python tests/test_day2.py

# Test Day 3 deep learning
python tests/test_day3.py

# Test Day 4 evaluation
python tests/test_day4.py
```

### 3. Run Notebook
Open `heart_disease_detection.ipynb` in Jupyter or VS Code and execute cells sequentially.

## Development Timeline

- **Day 1:** Setup, dataset selection, and planning
- **Day 2:** Data exploration and baseline ML models
- **Day 3:** Deep learning model development
- **Day 4:** Evaluation, optimization, and reporting

## Requirements
- Python 3.8+
- TensorFlow 2.x
- scikit-learn
- pandas, numpy
- matplotlib, seaborn
- scipy
- wfdb (for ECG data)
- librosa (for signal processing)

## Author
Mini Project - October 2025

## License
Educational Project
