# Quick Reference Guide - Heart Disease Detection Project

## 🚀 Common Commands

### Environment Management
```bash
# Activate virtual environment (Windows PowerShell)
.\venv\Scripts\Activate.ps1

# Deactivate
deactivate

# Reinstall packages
pip install -r requirements.txt
```

### Running Tests
```bash
# Run Day 1 tests
python tests/test_day1.py

# Run specific test (future)
python tests/test_day2.py
```

### Dataset Management
```bash
# Download/re-download datasets
python datasets/download_datasets.py
```

### Jupyter Notebook
```bash
# Start Jupyter
jupyter notebook

# Convert notebook to HTML
jupyter nbconvert --to html heart_disease_detection.ipynb

# Execute notebook from command line
jupyter nbconvert --to notebook --execute heart_disease_detection.ipynb
```

---

## 📂 File Locations

### Datasets
- **Cleveland CSV:** `datasets/cleveland/heart.csv`
- **ECG Signals:** `datasets/mit-bih/100.dat` (and 101-104)

### Code
- **Main Notebook:** `heart_disease_detection.ipynb`
- **Tests:** `tests/test_day*.py`
- **Utilities:** `datasets/download_datasets.py`

### Documentation
- **Project Overview:** `README_PROJECT.md`
- **Day 1 Log:** `completion_log_day1.md`
- **Quick Summary:** `DAY1_SUMMARY.md`

---

## 🔍 Troubleshooting

### Issue: Virtual environment won't activate
```bash
# Windows PowerShell - Set execution policy
Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser
```

### Issue: Module not found
```bash
# Make sure venv is activated, then reinstall
pip install -r requirements.txt --force-reinstall
```

### Issue: Dataset files missing
```bash
# Re-download datasets
python datasets/download_datasets.py
```

### Issue: Jupyter kernel not found
```bash
# Install ipykernel in venv
pip install ipykernel
python -m ipykernel install --user --name=miniproject
```

---

## 📊 Notebook Sections

1. **Section A:** Data Loading & Exploration
2. **Section B:** Data Preprocessing  
3. **Section C:** Baseline ML Models
4. **Section D:** Deep Learning Models
5. **Section E:** Evaluation & Visualization

---

## 🧪 Testing Checklist

- [ ] Virtual environment activated
- [ ] All packages installed
- [ ] Datasets downloaded
- [ ] Test suite passes (41/41)
- [ ] Notebook opens without errors

Run: `python tests/test_day1.py` to verify all

---

## 📦 Installed Packages

| Package | Version | Purpose |
|---------|---------|---------|
| tensorflow | 2.20.0 | Deep Learning |
| scikit-learn | 1.7.2 | Machine Learning |
| pandas | 2.3.3 | Data Analysis |
| numpy | 2.3.4 | Numerical Computing |
| matplotlib | 3.10.7 | Visualization |
| scipy | 1.16.2 | Scientific Computing |
| wfdb | 4.3.0 | ECG Signal Processing |
| librosa | 0.11.0 | Signal Analysis |

---

## 🎯 Project Goals

### Short Term (Days 1-2)
- ✅ Setup environment
- ✅ Download datasets
- 🔄 Preprocess data
- 🔄 Exploratory analysis

### Medium Term (Days 3-4)
- ⏳ Train baseline ML models
- ⏳ Build deep learning models
- ⏳ Optimize hyperparameters

### Long Term (Day 5)
- ⏳ Comprehensive evaluation
- ⏳ Model comparison
- ⏳ Final documentation
- ⏳ Results visualization

---

## 💡 Tips

1. **Always activate venv** before running any Python commands
2. **Run tests frequently** to catch issues early
3. **Save notebook often** - use Ctrl+S
4. **Check logs** in completion_log_day*.md for detailed info
5. **Use relative paths** for cross-platform compatibility

---

## 📞 Quick Links

- [TensorFlow Docs](https://www.tensorflow.org/api_docs)
- [scikit-learn Docs](https://scikit-learn.org/stable/)
- [Pandas Docs](https://pandas.pydata.org/docs/)
- [WFDB Docs](https://wfdb.readthedocs.io/)
- [UCI Dataset](https://archive.ics.uci.edu/ml/datasets/heart+disease)
- [MIT-BIH Database](https://physionet.org/content/mitdb/1.0.0/)

---

**Last Updated:** October 28, 2025  
**Current Status:** Day 1 Complete ✅
