@echo off
REM BERT Training Script - Run all three models sequentially
REM =========================================================

echo ============================================================
echo BERT Training Pipeline
echo ============================================================
echo.
echo This will train three BERT models:
echo   1. Elon Musk Tweet Model
echo   2. NBC Tesla Article Model  
echo   3. NBC Market News Model
echo.
echo Each model will output F1 and ACC metrics.
echo ============================================================
echo.

cd /d "%~dp0"

REM Activate torch-gpu conda environment
echo Activating torch-gpu conda environment...
call conda activate torch-gpu
if errorlevel 1 (
    echo [ERROR] Failed to activate torch-gpu environment.
    echo [INFO] Please make sure conda is installed and torch-gpu environment exists.
    pause
    exit /b 1
)
echo [OK] torch-gpu environment activated
echo.

REM Check if Python is available
python --version >nul 2>&1
if errorlevel 1 (
    echo [ERROR] Python not found. Please install Python or activate your conda environment.
    pause
    exit /b 1
)

echo.
echo ============================================================
echo [1/3] Training Elon Musk Tweet Model
echo ============================================================
echo.
python train_elon_musk_tweet_model.py
if errorlevel 1 (
    echo [ERROR] Tweet model training failed!
    pause
    exit /b 1
)

echo.
echo ============================================================
echo [2/3] Training NBC Tesla Article Model
echo ============================================================
echo.
python train_nbc_tesla_article_model.py
if errorlevel 1 (
    echo [ERROR] Tesla article model training failed!
    pause
    exit /b 1
)

echo.
echo ============================================================
echo [3/3] Training NBC Market News Model
echo ============================================================
echo.
python train_nbc_market_news_model.py
if errorlevel 1 (
    echo [ERROR] Market news model training failed!
    pause
    exit /b 1
)

echo.
echo ============================================================
echo ALL TRAINING COMPLETE!
echo ============================================================
echo.
echo All three models have been trained successfully.
echo Check the outputs/ directory for model files and predictions.
echo.
pause

