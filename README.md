# HackBio - Gene Expression Analysis

This repository contains code for analyzing and modeling gene expression data, with a focus on predicting drug responses based on gene expression patterns.

## Repository Structure

```
HackBio/
├── data/                    # Data files
│   ├── raw/                 # Raw data files
│   └── processed/           # Processed data files
├── docs/                    # Documentation
│   └── reports/             # Project reports and findings
├── logs/                    # Log files
├── notebooks/               # Jupyter notebooks
├── results/                 # Results from model runs
│   ├── advanced/            # Results from advanced model
│   ├── enhanced/            # Results from enhanced model
│   ├── optimized/           # Results from optimized model
│   └── svd/                 # SVD analysis results
├── src/                     # Source code
│   ├── models/              # Model implementations
│   ├── preprocessing/       # Data preprocessing code
│   ├── utils/               # Utility functions
│   └── visualization/       # Data visualization code
└── tests/                   # Test files
```

## Installation

```bash
# Create  virtual environment
python -m venv venv

# Activate the virtual environment
source venv/bin/activate  # Linux/Mac
venv\Scripts\activate     # Windows

# Install dependencies
pip install -r requirements.txt
```

## Usage

1. Preprocess data:
```bash
python src/preprocessing/svd_optimization.py
```

2. Train and evaluate models:
```bash
python src/models/advanced_model.py
python src/models/enhanced_cnn.py
python src/models/optimized_cnn.py
```

3. View results in the `results/` directory

## Documentation

See the `docs/` directory for detailed reports and findings. 