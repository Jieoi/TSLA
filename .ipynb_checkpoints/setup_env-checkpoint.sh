#!/bin/bash
# TSLA Project - Virtual Environment Setup Script (Mac/Linux)
# This script creates a virtual environment and installs requirements if needed

VENV_NAME="tesla_env"
REQUIREMENTS_FILE="requirements.txt"

echo "========================================"
echo "TSLA Project Environment Setup"
echo "========================================"

# Check if virtual environment exists
if [ -d "$VENV_NAME" ]; then
    echo "Virtual environment '$VENV_NAME' already exists."
    echo "Activating existing environment..."
    source $VENV_NAME/bin/activate
    echo "Environment activated successfully!"
    
    # Check if requirements.txt exists
    if [ -f "$REQUIREMENTS_FILE" ]; then
        echo "Checking for package updates..."
        pip install -r $REQUIREMENTS_FILE --upgrade
        echo "Requirements updated successfully!"
    else
        echo "Warning: $REQUIREMENTS_FILE not found. Skipping package installation."
    fi
else
    echo "Creating new virtual environment '$VENV_NAME'..."
    python3 -m venv $VENV_NAME
    
    if [ $? -ne 0 ]; then
        echo "Error: Failed to create virtual environment. Please ensure Python 3 is installed."
        echo "You can install Python 3 using: brew install python3 (on Mac) or your system package manager (on Linux)"
        exit 1
    fi
    
    echo "Activating new environment..."
    source $VENV_NAME/bin/activate
    
    # Upgrade pip
    echo "Upgrading pip..."
    python -m pip install --upgrade pip
    
    # Install requirements if file exists
    if [ -f "$REQUIREMENTS_FILE" ]; then
        echo "Installing requirements from $REQUIREMENTS_FILE..."
        pip install -r $REQUIREMENTS_FILE
        if [ $? -ne 0 ]; then
            echo "Error: Failed to install some requirements. Please check the error messages above."
            exit 1
        fi
        echo "Requirements installed successfully!"
    else
        echo "Warning: $REQUIREMENTS_FILE not found. Skipping package installation."
    fi
fi

echo ""
echo "========================================"
echo "Setup Complete!"
echo "========================================"
echo ""
echo "To activate this environment in the future, run:"
echo "  source $VENV_NAME/bin/activate"
echo ""
echo "To deactivate the environment, run:"
echo "  deactivate"
echo ""
echo "To start Jupyter notebook, run:"
echo "  jupyter notebook"
echo ""
echo "Press any key to continue..."
read -n 1 -s
