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
pip install pandas numpy matplotlib seaborn scikit-learn
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

## Next Steps

After completing this tutorial, you can:
1. Explore the `exploration.ipynb` notebook
2. Try different visualizations in `visualize_data.py`
3. Modify analysis parameters in `analyze_data.py`

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

## Getting Help

- Check the documentation in the `docs/` directory
- Review example notebooks
- Create an issue on GitHub
- Contact project maintainers 