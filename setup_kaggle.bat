@echo off
echo ============================================
echo    KAGGLE API SETUP - QUICK GUIDE
echo ============================================
echo.
echo OPTION 1: Get Your API Credentials (30 seconds)
echo.
echo 1. Open in browser: https://www.kaggle.com/settings/account
echo 2. Scroll to "API" section
echo 3. Click "Create New Token"
echo 4. This downloads "kaggle.json" file
echo 5. Move it to: C:\Users\%USERNAME%\.kaggle\kaggle.json
echo.
echo Then run: python download_kaggle_dataset.py
echo.
echo ============================================
echo.
echo OPTION 2: Manual Download (EASIEST - 2 minutes)
echo.
echo 1. Open: https://www.kaggle.com/datasets/oktayrdeki/heart-disease
echo 2. Click "Download" button
echo 3. Extract heart_disease.csv
echo 4. Move to: %cd%\datasets\kaggle\heart_disease.csv
echo.
echo Then run: python integrate_kaggle_dataset.py
echo.
echo ============================================
echo.
echo Which option do you prefer?
echo.
set /p choice="Press 1 for API setup, 2 to open download page, or X to exit: "

if /i "%choice%"=="1" (
    echo.
    echo Opening Kaggle Account Settings...
    start https://www.kaggle.com/settings/account
    echo.
    echo After downloading kaggle.json:
    echo 1. Create folder: C:\Users\%USERNAME%\.kaggle
    echo 2. Move kaggle.json there
    echo 3. Run: python download_kaggle_dataset.py
    echo.
    pause
) else if /i "%choice%"=="2" (
    echo.
    echo Opening Kaggle dataset page...
    start https://www.kaggle.com/datasets/oktayrdeki/heart-disease
    echo.
    echo After downloading:
    echo 1. Extract heart_disease.csv
    echo 2. Move to: %cd%\datasets\kaggle\
    echo 3. Run: python integrate_kaggle_dataset.py
    echo.
    pause
) else (
    echo.
    echo No problem! Run this script again when ready.
    echo.
    pause
)
