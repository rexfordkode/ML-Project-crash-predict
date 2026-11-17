# ML-Project-crash-predict

Predicting pedestrian and cyclist crash severity using US CRSS (Crash Report Sampling System) data.

## 🚀 Getting Started for Team Members (Google Colab Setup)

Follow these steps to set up the environment, download/upload raw data, preprocess, and run training models.

### 1. Open in Google Colab
Simply copy and paste this notebook's link or upload the project notebook directly into [Google Colab](https://colab.research.google.com/).

### 2. Clone the Repository
Run the following command in a cell to clone your branch and navigate inside the project folder:
```python
!git clone https://github.com/rexfordkode/ML-Project-crash-predict.git
%cd ML-Project-crash-predict
```

### 3. Setting Up Git Credentials (To Push Changes)
Before making commits, configure your Git username and email:
```python
!git config --global user.name "Your GitHub Username"
!git config --global user.email "your_email@example.com"
```

### 4. Uploading & Extracting Data
1. Place your `data.zip` file directly into `/content/ML-Project-crash-predict/data/raw/`.
2. Extract it using:
```python
import zipfile
import os

os.makedirs('data/raw', exist_ok=True)
with zipfile.ZipFile('data/raw/data.zip', 'r') as zip_ref:
    zip_ref.extractall('data/raw')
```

### 5. Running Preprocessing
Execute the modular preprocessing script to ingest the raw CSVs and generate the cleaned master dataset:
```python
!python src/preprocess.py
```
This generates a clean dataset ready for training at `data/processed/clean_ped_cyc_crashes.csv`.

## 📂 Folder Structure
```
├── data/
│   ├── raw/           # Raw crash CSVs & data.zip (Git ignored)
│   └── processed/     # Cleaned model-ready dataset (Committed to Git)
├── src/
│   ├── __init__.py    # Empty package identifier
│   ├── ingest.py      # Raw dataset ingestion rules
│   ├── preprocess.py  # Data cleaning, filtering and feature engineering
│   ├── train.py       # Model training scripts
│   └── evaluate.py    # Model evaluation metrics
└── README.md
```

## ⚙️ Development & Pull Requests
* Make sure you checkout to your custom feature branch before committing:
  `!git checkout -b feat/your-feature-name`
* Stage and commit your script updates, leaving out heavy files (raw CSVs and `.zip` are automatically ignored by our `.gitignore`): 
  `!git add src/ .gitignore README.md`
  `!git commit -m "Describe your updates"`
  `!git push origin feat/your-feature-name`
