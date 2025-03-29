# Getting Started with Single-Cell Analysis

## Prerequisites

Before you begin, make sure you have:
1. Python 3.7 or higher installed
2. Basic understanding of Python programming
3. Git installed on your computer

## Quick Start Guide

### 1. Setting Up Your Environment

```bash
# Clone the repository
git clone https://github.com/jaffarkeikei/HackBio.git
cd HackBio

# Create a virtual environment
python -m venv venv

# Activate the virtual environment
# On Windows:
venv\Scripts\activate
# On macOS/Linux:
source venv/bin/activate

# Install required packages
pip install -r requirements.txt
```

### 2. Understanding the Data

Our dataset contains:
```
📊 Dataset Overview
├── 501 samples
├── 119 unique drugs
├── 6 cell types
└── 18,211 genes measured
```

### 3. Running Your First Analysis

```python
# Example code to load and view data
import pandas as pd

# Load the dataset
data = pd.read_parquet('dataset/de_train_split.parquet')

# View basic information
print(data.info())

# See first few rows
print(data.head())
```

### 4. Creating Your First Visualization

```python
# Example visualization code
import matplotlib.pyplot as plt
import seaborn as sns

# Create a simple visualization
plt.figure(figsize=(10, 6))
sns.countplot(data=data, x='cell_type')
plt.title('Distribution of Cell Types')
plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig('my_first_plot.png')
```

### 5. Training a Model (New)

```python
# Example model training code
from drug_response_model import DrugResponsePredictor

# Separate metadata and gene columns
metadata_cols = ['cell_type', 'sm_name', 'sm_lincs_id', 'SMILES', 'control']
gene_cols = [col for col in data.columns if col not in metadata_cols]

# Prepare data
X = data[['cell_type', 'sm_name']]  # Categorical features
y = data[gene_cols]  # Gene expression to predict

# Create and train model
model = DrugResponsePredictor(n_estimators=100)
model.fit(X, y, top_n_genes=100)  # Train on top 100 most variable genes

# Visualize results
model.visualize_predictions(n_genes=3)
```

### 6. Analyzing Molecular Structures (New)

```python
# Example molecular analysis code
from molecular_analysis import MolecularAnalyzer

# Create analyzer object
analyzer = MolecularAnalyzer(data['SMILES'])

# Calculate molecular descriptors
descriptors = analyzer.calculate_descriptors()
print(descriptors.head())

# Analyze drug-likeness
drug_likeness = analyzer.analyze_drug_likeness()
```

## Common Operations

### 1. Data Loading
```python
# Load training data
train_data = pd.read_parquet('dataset/de_train_split.parquet')

# Load test data
test_data = pd.read_parquet('dataset/de_test_split.parquet')
```

### 2. Basic Analysis
```python
# Get summary statistics
summary = train_data.describe()

# Count unique values
unique_drugs = train_data['sm_name'].nunique()
unique_cells = train_data['cell_type'].nunique()
```

### 3. Visualization Examples
```python
# Create heatmap
plt.figure(figsize=(12, 8))
sns.heatmap(data.corr(), cmap='viridis')
plt.title('Correlation Heatmap')
plt.savefig('correlation_heatmap.png')
```

### 4. Making Predictions (New)
```python
# Using the trained model for predictions
import pickle

# Load saved model
with open('models/drug_response_model.pkl', 'rb') as f:
    model = pickle.load(f)

# Example data
example_data = {
    'drug_name': ['Clotrimazole', 'Triptolide'],
    'cell_type': ['NK cells', 'T cells CD4+']
}
example_df = pd.DataFrame(example_data)

# Make predictions
predictions = model.predict(example_df)
```

## Running the Complete Pipeline (New)

For a complete analysis and model training workflow:

```bash
# Run the step-by-step pipeline (using complete_analysis.py)
python complete_analysis.py --step 1  # Start with molecular analysis
python complete_analysis.py --step 2  # Continue with data preprocessing
python complete_analysis.py --step 3  # Run model training

# For the fixed implementation of multi-output regression
python fixed_analysis.py

# For the final optimized solution with all fixes
python final_analysis.py

# For the advanced neural network models with SVD (following NeurIPS 2023 winners)
python advanced_model.py --svd_components 100  # Default: 100 components
python advanced_model.py --svd_components 50   # Try with fewer components
python advanced_model.py --epochs 50 --batch_size 64  # Customize training parameters
```

The **final_analysis.py** script is the most robust solution with:
- Complete categorical feature encoding
- XGBoost feature name compatibility
- Multi-output regression handling
- Comprehensive error handling
- Performance evaluation metrics
- Visualization generation

The **advanced_model.py** script implements the winning approaches from NeurIPS 2023:
- SVD dimensionality reduction
- 1D CNN, LSTM, and GRU neural network models
- Model ensembling
- Early stopping and performance evaluation

## Next Steps

After completing this tutorial, you can:
1. Explore the `exploration_notebook.ipynb` notebook for deeper analysis
2. Try different visualizations in `visualize_data.py`
3. Modify model parameters in `drug_response_model.py`
4. Analyze drug structures with the `molecular_analysis.py` module

## Troubleshooting

Common Issues and Solutions:

1. **Data Loading Error**
   ```
   Solution: Make sure you're in the correct directory
   and have downloaded the dataset files
   ```

2. **Memory Issues**
   ```
   Solution: Use data chunking or reduce dataset size
   data = pd.read_parquet('file.parquet', columns=['specific_columns'])
   ```

3. **Visualization Problems**
   ```
   Solution: Check matplotlib backend
   import matplotlib
   matplotlib.use('Agg')  # For systems without display
   ```

4. **RDKit Import Error** (New)
   ```
   Solution: Install RDKit separately if pip install fails
   conda install -c conda-forge rdkit
   ```

## Getting Help

- Check the documentation in the `docs/` directory
- Review example notebooks
- Create an issue on GitHub
- Contact project maintainers 