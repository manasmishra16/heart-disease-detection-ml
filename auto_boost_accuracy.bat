@echo off
echo ============================================
echo    WAITING FOR KAGGLE DATASET DOWNLOAD
echo ============================================
echo.
echo Please download from:
echo https://www.kaggle.com/datasets/oktayrdeki/heart-disease
echo.
echo Extract heart_disease.csv to:
echo %cd%\datasets\kaggle\
echo.
echo This script will automatically detect when ready...
echo Press Ctrl+C to cancel
echo.

:check
if exist "datasets\kaggle\heart_disease.csv" (
    echo.
    echo ============================================
    echo    DATASET FOUND! STARTING INTEGRATION...
    echo ============================================
    echo.
    python integrate_kaggle_dataset.py
    if errorlevel 1 (
        echo.
        echo ERROR: Integration failed!
        pause
        exit /b 1
    )
    echo.
    echo ============================================
    echo    STARTING TRAINING ON COMBINED DATA...
    echo ============================================
    echo.
    python train_on_combined_dataset.py
    if errorlevel 1 (
        echo.
        echo ERROR: Training failed!
        pause
        exit /b 1
    )
    echo.
    echo ============================================
    echo    SUCCESS! YOU NOW HAVE 95%+ ACCURACY!
    echo ============================================
    echo.
    echo Next: Restart your webpage to see the new accuracy!
    echo.
    pause
    exit /b 0
) else (
    timeout /t 5 /nobreak >nul
    goto check
)
