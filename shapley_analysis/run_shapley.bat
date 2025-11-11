@echo off
REM Shapley Calculation Runner - Neural Network and BERT
REM ====================================================

echo ============================================================
echo Shapley Value Computation Pipeline
echo ============================================================
echo.
echo This will compute SHAP values for:
echo   1. Neural network models (market, tesla, tweet)
echo   2. BERT models (tweet, tesla article, market news)
echo.
echo ============================================================
echo.

cd /d "%~dp0"

echo Activating torch-gpu conda environment...
call conda activate torch-gpu
if errorlevel 1 (
    echo [ERROR] Failed to activate torch-gpu environment.
    echo [INFO] Ensure conda is installed and the environment exists.
    pause
    exit /b 1
)
echo [OK] torch-gpu environment activated
echo.

echo ============================================================
echo [1/2] Computing SHAP for Neural Network Models
echo ============================================================
echo.
python compute_shapley_nn.py
if errorlevel 1 (
    echo [ERROR] Neural network SHAP computation failed!
    pause
    exit /b 1
)

echo.
echo ============================================================
echo [2/2] Computing SHAP for BERT Models
echo ============================================================
echo.
python compute_shapley_bert.py
if errorlevel 1 (
    echo [ERROR] BERT SHAP computation failed!
    pause
    exit /b 1
)

echo.
echo ============================================================
echo SHAP computations complete!
echo ============================================================
echo Results saved under appropriate outputs\shapley\ directories.
echo.
pause

