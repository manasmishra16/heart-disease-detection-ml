# Heart Disease Detection Mini Project 🫀

A comprehensive machine learning and deep learning project for detecting heart disease using:
1. **Clinical tabular data** (Cleveland/UCI Heart Disease Dataset)
2. **ECG signal data** (MIT-BIH Arrhythmia Database from PhysioNet)

## 📋 Project Overview

This project implements multiple approaches to heart disease detection:
- Traditional ML models (Logistic Regression, Random Forest, SVM, etc.)
- Deep Learning models for tabular data (Fully Connected Neural Networks)
- Deep Learning models for ECG signals (CNN, RNN/LSTM)

### Datasets

#### 1. Cleveland Heart Disease Dataset (UCI)
- **Source:** UCI Machine Learning Repository
- **Size:** 303 instances, 14 attributes
- **Task:** Binary classification (heart disease present/absent)
- **Features:** Age, sex, chest pain type, blood pressure, cholesterol, etc.

#### 2. MIT-BIH Arrhythmia Database
- **Source:** PhysioNet
- **Records:** 5 subjects (100-104) for this project
- **Sampling Rate:** 360 Hz
- **Task:** ECG-based arrhythmia classification
- **Data Format:** Signal data (.dat), headers (.hea), annotations (.atr)

## 🚀 Getting Started

### Prerequisites
- Python 3.11+ (recommended)
- pip package manager
- Git (for cloning)

### Installation

1. **Clone the repository** (or navigate to project directory)
```bash
cd d:\Projects\MiniProject
```

2. **Create and activate virtual environment**
```bash
# Create virtual environment
python -m venv venv

# Activate on Windows (PowerShell)
.\venv\Scripts\Activate.ps1

# Activate on Windows (CMD)
.\venv\Scripts\activate.bat

# Activate on Linux/Mac
source venv/bin/activate
```

3. **Install dependencies**
```bash
pip install -r requirements.txt
```

4. **Download datasets** (if not already present)
```bash
python datasets/download_datasets.py
```

5. **Verify setup**
```bash
python tests/test_day1.py
```

Expected output: "✓ ALL TESTS PASSED! Day 1 setup is complete."

## 📁 Project Structure

```
MiniProject/
├── venv/                          # Virtual environment (not in git)
├── datasets/
│   ├── cleveland/
│   │   ├── heart.csv             # Main Cleveland dataset
│   │   ├── processed.cleveland.data
│   │   └── README.md
│   ├── mit-bih/
│   │   ├── 100.dat, 100.hea, 100.atr
│   │   ├── 101.dat, 101.hea, 101.atr
│   │   └── ... (records 102-104)
│   └── download_datasets.py      # Automated dataset downloader
├── docs/                          # Additional documentation
├── models/                        # Saved model files (created during training)
├── results/                       # Plots, metrics, outputs (created during evaluation)
├── scripts/                       # Utility and helper scripts
├── tests/
│   ├── test_day1.py              # Day 1 verification tests ✅
│   ├── test_day2.py              # Day 2 verification tests
│   └── test_day3.py              # Day 3 verification tests
├── heart_disease_detection.ipynb  # Main Jupyter notebook
├── requirements.txt               # Python dependencies
├── completion_log_day1.md         # Day 1 completion log
└── README.md                      # This file
```

## 🎯 Project Timeline

### ✅ Day 1: Setup, Dataset Selection & Planning (COMPLETED)
- [x] Environment setup (virtual environment + packages)
- [x] Dataset download (Cleveland + MIT-BIH subset)
- [x] Notebook skeleton with 5 major sections
- [x] Automated test suite (41 tests passing)
- [x] Documentation and completion log

**Deliverables:** Working environment, datasets, notebook skeleton, test suite

### 📝 Day 2: Data Preprocessing & EDA (Planned)
- [ ] Cleveland dataset preprocessing (missing values, scaling)
- [ ] ECG signal preprocessing (filtering, segmentation)
- [ ] Exploratory data analysis
- [ ] Feature engineering
- [ ] Train/validation/test splits

**Deliverables:** Preprocessed datasets, EDA visualizations, test_day2.py

### 🤖 Day 3: Baseline Machine Learning Models (Planned)
- [ ] Logistic Regression
- [ ] Random Forest
- [ ] Gradient Boosting
- [ ] SVM
- [ ] KNN
- [ ] Model comparison and selection

**Deliverables:** Trained ML models, performance metrics, test_day3.py

### 🧠 Day 4: Deep Learning Models (Planned)
- [ ] Fully connected neural network for tabular data
- [ ] CNN for ECG signal classification
- [ ] RNN/LSTM for ECG sequences
- [ ] Hyperparameter tuning

**Deliverables:** Trained DL models, saved checkpoints, learning curves

### 📊 Day 5: Evaluation & Final Results (Planned)
- [ ] Comprehensive model evaluation
- [ ] Confusion matrices and ROC curves
- [ ] Feature importance analysis
- [ ] Final comparison and recommendations
- [ ] Project documentation

**Deliverables:** Final report, visualizations, model comparison

## 🛠️ Technologies Used

### Core Libraries
- **TensorFlow 2.20.0** - Deep learning framework
- **scikit-learn 1.7.2** - Machine learning algorithms
- **pandas 2.3.3** - Data manipulation
- **numpy 2.3.4** - Numerical computing

### Visualization
- **matplotlib 3.10.7** - Plotting and visualization
- **seaborn** - Statistical visualizations

### Signal Processing
- **wfdb 4.3.0** - PhysioNet/WFDB format handling
- **scipy 1.16.2** - Scientific computing and signal processing
- **librosa 0.11.0** - Audio/signal analysis

### Development
- **Jupyter** - Interactive notebook environment

## 📊 Notebook Structure

The main notebook (`heart_disease_detection.ipynb`) is organized into 5 sections:

### Section A: Data Loading and Exploration
- Import libraries
- Load Cleveland dataset
- Load MIT-BIH ECG data
- Initial visualization and statistics

### Section B: Data Preprocessing
- Handle missing values
- Feature scaling and normalization
- ECG signal filtering
- Data segmentation

### Section C: Baseline Machine Learning Models
- Traditional ML algorithms
- Cross-validation
- Initial performance metrics

### Section D: Deep Learning Models
- Neural networks for tabular data
- CNN for ECG signals
- RNN/LSTM for sequential data

### Section E: Evaluation Metrics and Visualization
- Comprehensive metrics
- Confusion matrices
- ROC curves
- Model comparison

## 🧪 Testing

Each day has an associated test script to verify completion:

```bash
# Run Day 1 tests
python tests/test_day1.py

# Run Day 2 tests (when available)
python tests/test_day2.py
```

## 📈 Usage

### Running the Notebook

1. **Start Jupyter**
```bash
jupyter notebook
```

2. **Open** `heart_disease_detection.ipynb`

3. **Run cells sequentially** or use "Run All"

### Command Line Execution

For automated execution:
```bash
jupyter nbconvert --to notebook --execute heart_disease_detection.ipynb
```

## 📝 Notes

- Always activate the virtual environment before running scripts
- Dataset downloads are automated but require internet connection
- ECG processing is computationally intensive
- GPU acceleration recommended for deep learning models

## 🔍 Troubleshooting

### Virtual Environment Issues
```bash
# Windows: If execution policy prevents activation
Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser
```

### Missing Packages
```bash
# Reinstall all requirements
pip install -r requirements.txt --force-reinstall
```

### Dataset Download Failures
```bash
# Manually download datasets
python datasets/download_datasets.py
```

### Test Failures
```bash
# Run tests with verbose output
python tests/test_day1.py -v
```

## 📚 References

- [UCI Heart Disease Dataset](https://archive.ics.uci.edu/ml/datasets/heart+disease)
- [MIT-BIH Arrhythmia Database](https://physionet.org/content/mitdb/1.0.0/)
- [TensorFlow Documentation](https://www.tensorflow.org/)
- [scikit-learn Documentation](https://scikit-learn.org/)
- [WFDB Documentation](https://wfdb.readthedocs.io/)

---

**Current Status:** Day 1 Complete ✅ | 41/41 Tests Passing

**Last Updated:** October 28, 2025
