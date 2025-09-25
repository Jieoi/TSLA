# Text-Encoded Sentiment for Long/short Alpha (TESLA)

## Overview

This project investigates the comparative influence of different text sources in supporting investment decision-making. Specifically, we will examine formal news articles from NBC that mentioned Tesla Inc. or its founder, Elon Musk, alongside informal content from Musk's X (formerly Twitter) account (@elonmusk).

Text data, especially financial and social media text data, is inherently noisy due to a myriad of possible reasons, from irrelevant text to sensationalised headlines meant to draw views and readership. This challenge is further exacerbated by the high volume of text data present in social media sources like X, which further reduces the signal-to-noise ratio (Kolajo et al., 2022). Given the heavy reliance on accurate input data sources for model accuracy, each of this project's five sections: 
(1) Data Collection; 
(2) Data Preparation; 
(3) Exploratory Data Analysis (EDA) & Feature Engineering; 
(4) Model Building;
(5) Model Evaluation, would seek to reduce the associated noise and/or amplify the relevant signals for subsequent steps 

![Project_flow](meta/flow.png)

## 🚀 Quick Start

### Prerequisites
- Python 3.7 or higher
- pip (Python package installer)

### Environment Setup

This project includes automated setup scripts for both Windows and Mac/Linux systems.

#### For Windows Users:
1. **Double-click** `setup_env.bat` in the project folder
2. The script will automatically:
   - Create a virtual environment named `tesla_env`
   - Install all required packages from `requirements.txt`
   - Activate the environment

#### For Mac/Linux Users:
1. **Open Terminal** and navigate to the project folder
2. **Run the setup script:**
   ```bash
   ./setup_env.sh
   ```
   If you get a permission error, first make it executable:
   ```bash
   chmod +x setup_env.sh
   ./setup_env.sh
   ```

#### Manual Setup (Alternative):
If you prefer manual setup or the scripts don't work:

1. **Create virtual environment:**
   ```bash
   # Windows
   python -m venv tesla_env
   
   # Mac/Linux
   python3 -m venv tesla_env
   ```

2. **Activate environment:**
   ```bash
   # Windows
   tesla_env\Scripts\activate
   
   # Mac/Linux
   source tesla_env/bin/activate
   ```

3. **Install requirements:**
   ```bash
   pip install -r requirements.txt
   ```

### Running the Project

1. **Activate the environment** (if not already active):
   ```bash
   # Windows
   tesla_env\Scripts\activate
   
   # Mac/Linux
   source tesla_env/bin/activate
   ```

2. **Start Jupyter Notebook:**
   ```bash
   jupyter notebook
   ```

3. **Open** `demo.ipynb` to begin working with the project

### Deactivating the Environment
When you're done working:
```bash
deactivate
```

##  Dependencies

The project uses the following key libraries (see `requirements.txt` for complete list):
- **Data Processing**: pandas, numpy, scipy
- **Natural Language Processing**: nltk
- **Machine Learning**: scikit-learn
- **Visualization**: matplotlib, seaborn
- **Web Scraping**: requests, beautifulsoup4
- **Development**: jupyter, ipykernel

## Problem Statement

## Dataset

## Approach

### Text data preprocessing library
For standardisation, a library is created at `data_cleaning.py` for all the comment text processing tasks across the three data sources.
