@echo off
REM TSLA Project - Virtual Environment Setup Script
REM This script creates a virtual environment and installs requirements if needed

set VENV_NAME=tesla_env
set REQUIREMENTS_FILE=requirements.txt

echo ========================================
echo TSLA Project Environment Setup
echo ========================================

REM Check if virtual environment exists
if exist "%VENV_NAME%" (
    echo Virtual environment '%VENV_NAME%' already exists.
    echo Activating existing environment...
    call %VENV_NAME%\Scripts\activate.bat
    echo Environment activated successfully!
    
    REM Check if requirements.txt exists
    if exist "%REQUIREMENTS_FILE%" (
        echo Checking for package updates...
        pip install -r %REQUIREMENTS_FILE% --upgrade
        echo Requirements updated successfully!
    ) else (
        echo Warning: %REQUIREMENTS_FILE% not found. Skipping package installation.
    )
) else (
    echo Creating new virtual environment '%VENV_NAME%'...
    python -m venv %VENV_NAME%
    
    if errorlevel 1 (
        echo Error: Failed to create virtual environment. Please ensure Python is installed.
        pause
        exit /b 1
    )
    
    echo Activating new environment...
    call %VENV_NAME%\Scripts\activate.bat
    
    REM Upgrade pip
    echo Upgrading pip...
    python -m pip install --upgrade pip
    
    REM Install requirements if file exists
    if exist "%REQUIREMENTS_FILE%" (
        echo Installing requirements from %REQUIREMENTS_FILE%...
        pip install -r %REQUIREMENTS_FILE%
        if errorlevel 1 (
            echo Error: Failed to install some requirements. Please check the error messages above.
            pause
            exit /b 1
        )
        echo Requirements installed successfully!
    ) else (
        echo Warning: %REQUIREMENTS_FILE% not found. Skipping package installation.
    )
)

echo.
echo ========================================
echo Setup Complete!
echo ========================================
echo.
echo To activate this environment in the future, run:
echo   %VENV_NAME%\Scripts\activate.bat
echo.
echo To deactivate the environment, run:
echo   deactivate
echo.
echo To start Jupyter notebook, run:
echo   jupyter notebook
echo.
pause
