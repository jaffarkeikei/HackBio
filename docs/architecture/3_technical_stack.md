# Technical Stack Architecture

This document details the technical stack used in the Single-Cell Perturbations Analysis project, including core libraries, frameworks, and tools.

## Technology Stack Overview

```mermaid
graph TD
    A[Python Ecosystem] --> B[Data Processing]
    A --> C[Analysis & Modeling]
    A --> D[Visualization]
    A --> E[Molecular Analysis]
    A --> F[Development Tools]
    
    B --> B1[Pandas]
    B --> B2[NumPy]
    B --> B3[PyArrow]
    
    C --> C1[Scikit-learn]
    C --> C2[SciPy]
    C --> C3[StatsModels]
    
    D --> D1[Matplotlib]
    D --> D2[Seaborn]
    
    E --> E1[RDKit]
    
    F --> F1[Jupyter]
    F --> F2[Git/GitHub]
```

## Core Technologies

### Python Ecosystem

Python serves as the foundation of our technical stack due to its rich ecosystem for data science and bioinformatics:

```
┌─────────────────────────────────────────────────────────┐
│                     Python 3.7+                         │
├─────────────────────────────────────────────────────────┤
│ ┌─────────────┐ ┌─────────────┐ ┌─────────────────────┐ │
│ │ Data        │ │ Scientific  │ │ Machine Learning    │ │
│ │ Manipulation│ │ Computing   │ │ & Statistics        │ │
│ └─────────────┘ └─────────────┘ └─────────────────────┘ │
└─────────────────────────────────────────────────────────┘
```

### Dependency Management

```mermaid
graph TD
    A[Requirements.txt] --> B[Virtual Environment]
    B --> C[Package Installations]
    C --> D1[Core Libraries]
    C --> D2[Analysis Libraries]
    C --> D3[Visualization Libraries]
    C --> D4[Specialized Libraries]
```

## Data Processing Layer

### Pandas & NumPy

```
┌───────────────────────────────────────────┐
│               Pandas                      │
│  ┌────────────────────────────────────┐   │
│  │           DataFrame                │   │
│  │  ┌──────┬──────┬──────┬───────┐    │   │
│  │  │Index │Col 1 │Col 2 │ ...   │    │   │
│  │  ├──────┼──────┼──────┼───────┤    │   │
│  │  │  0   │ val  │ val  │ ...   │    │   │
│  │  ├──────┼──────┼──────┼───────┤    │   │
│  │  │  1   │ val  │ val  │ ...   │    │   │
│  │  └──────┴──────┴──────┴───────┘    │   │
│  └────────────────────────────────────┘   │
│                   ▲                       │
│                   │                       │
│  ┌────────────────────────────────────┐   │
│  │             NumPy                  │   │
│  │  ┌──────────────────────────────┐  │   │
│  │  │         ndarray              │  │   │
│  │  └──────────────────────────────┘  │   │
│  └────────────────────────────────────┘   │
└───────────────────────────────────────────┘
```

Key capabilities:
- **Pandas**: Data manipulation, cleaning, and analysis
- **NumPy**: Numerical computing and array operations
- **PyArrow**: Efficient Parquet file handling

## Analysis and Modeling Layer

### Machine Learning with Scikit-learn

```mermaid
flowchart TD
    A[Raw Data] --> B[Preprocessing]
    B --> C[Feature Engineering]
    C --> D[Model Training]
    D --> E[Evaluation]
    
    subgraph Scikit-learn
    B
    C
    D
    E
    end
    
    B --> B1[StandardScaler]
    B --> B2[OneHotEncoder]
    C --> C1[VarianceThreshold]
    C --> C2[PCA]
    D --> D1[RandomForestRegressor]
    D --> D2[RandomForestClassifier]
    E --> E1[Cross-Validation]
    E --> E2[Metrics]
```

### Statistical Analysis with SciPy and StatsModels

These libraries provide statistical functions for hypothesis testing, feature selection, and more:

```
┌───────────────────────────────────────────────┐
│                SciPy & StatsModels            │
├───────────────────────────────────────────────┤
│ ┌─────────────────┐ ┌───────────────────────┐ │
│ │ Hypothesis Tests│ │ Statistical Models    │ │
│ │ - t-test        │ │ - Linear Regression   │ │
│ │ - ANOVA         │ │ - Logistic Regression │ │
│ │ - Chi-square    │ │ - Time Series         │ │
│ └─────────────────┘ └───────────────────────┘ │
│ ┌─────────────────┐ ┌───────────────────────┐ │
│ │ Multiple Testing│ │ Descriptive Statistics│ │
│ │ - FDR Correction│ │ - Summary Statistics  │ │
│ │ - Bonferroni    │ │ - Distribution Tests  │ │
│ └─────────────────┘ └───────────────────────┘ │
└───────────────────────────────────────────────┘
```

## Visualization Layer

### Data Visualization with Matplotlib and Seaborn

Visualization capabilities are critical for interpreting the complex data in our project:

```python
# Sample Visualization Code
import matplotlib.pyplot as plt
import seaborn as sns

# Create plot
plt.figure(figsize=(10, 6))
sns.scatterplot(x='PC1', y='PC2', hue='cell_type', data=pca_results)
plt.title('PCA of Gene Expression by Cell Type')
plt.savefig('pca_visualization.png')
```

Visualization hierarchy:

```
┌───────────────────────────────────────────────┐
│               Visualization Layer             │
│                                               │
│  ┌─────────────────────┐  ┌────────────────┐  │
│  │    Matplotlib       │  │    Seaborn     │  │
│  │  (Low-level API)    │  │(High-level API)│  │
│  └─────────────────────┘  └────────────────┘  │
│  ┌────────────────────────────────────────┐   │
│  │           Output Formats               │   │
│  │    PNG    │    PDF    │    SVG         │   │
│  └────────────────────────────────────────┘   │
└───────────────────────────────────────────────┘
```

### Common Plot Types

```
┌───────────────┐ ┌───────────────┐ ┌───────────────┐
│  Heatmap      │ │  PCA/t-SNE    │ │  Bar Charts   │
│               │ │               │ │               │
│  █████████    │ │      •        │ │   █    █      │
│  █████████    │ │    • • •      │ │   █    █      │
│  █████████    │ │   •  •    •   │ │   █  █ █ █    │
│               │ │  •   •     •  │ │   █  █ █ █    │
└───────────────┘ └───────────────┘ └───────────────┘

┌───────────────┐ ┌───────────────┐ ┌───────────────┐
│  Boxplots     │ │ Volcano Plots │ │ Molecule Viz  │
│               │ │               │ │               │
│    ┬─┐        │ │      •        │ │    O═C−N      │
│    │ │        │ │    •   •      │ │    /   \      │
│    ├─┤        │ │  •       •    │ │  •     •      │
│    └─┘        │ │ •  •  •  •  • │ │    \   /      │
│               │ │               │ │     C═C       │
└───────────────┘ └───────────────┘ └───────────────┘
```

## Molecular Analysis Layer

### Cheminformatics with RDKit

RDKit is used for analyzing drug structures from SMILES strings:

```mermaid
graph LR
    A[SMILES String] --> B[RDKit Molecule]
    B --> C1[Molecular Descriptors]
    B --> C2[Molecular Fingerprints]
    B --> C3[Structural Visualization]
    C1 --> D1[Physiochemical Properties]
    C2 --> D2[Similarity Analysis]
    C3 --> D3[2D/3D Rendering]
```

Key RDKit capabilities:
- SMILES parsing and manipulation
- Molecular descriptor calculation
- Chemical structure visualization
- Similarity searching
- Molecular fingerprinting

## Development and Collaboration Tools

### Interactive Development with Jupyter

```
┌───────────────────────────────────────────────┐
│               Jupyter Notebook                │
├───────────────────────────────────────────────┤
│ ┌─────────────────────────────────────────┐   │
│ │ # Markdown Cell                         │   │
│ │ This is explanatory text                │   │
│ └─────────────────────────────────────────┘   │
│ ┌─────────────────────────────────────────┐   │
│ │ # Code Cell                             │   │
│ │ import pandas as pd                     │   │
│ │ df = pd.read_parquet('data.parquet')    │   │
│ │ df.head()                               │   │
│ └─────────────────────────────────────────┘   │
│ ┌─────────────────────────────────────────┐   │
│ │ # Output Cell                           │   │
│ │ [Table output from above command]       │   │
│ └─────────────────────────────────────────┘   │
└───────────────────────────────────────────────┘
```

### Version Control with Git and GitHub

```mermaid
gitGraph
    commit
    branch feature
    checkout feature
    commit
    commit
    checkout main
    merge feature
    commit
```

## System Integration

### Component Interaction

```mermaid
sequenceDiagram
    participant U as User
    participant DL as Data Layer
    participant ML as ML Layer
    participant VL as Viz Layer
    participant MA as Molecular Analysis
    
    U->>DL: Load Data
    DL->>ML: Processed Data
    ML->>DL: Predictions
    DL->>VL: Data for Visualization
    DL->>MA: SMILES Strings
    MA->>DL: Molecular Descriptors
    VL->>U: Visualizations
    ML->>U: Model Results
    MA->>U: Molecular Insights
```

### Architecture Layers

```
┌───────────────────────────────────────────────┐
│             Application Layer                 │
│  ┌────────────────────────────────────────┐   │
│  │   Notebooks │ Scripts │ Models │ Data  │   │
│  └────────────────────────────────────────┘   │
└───────────────────────────────────────────────┘
                     ▲
                     │
┌───────────────────────────────────────────────┐
│              Framework Layer                  │
│  ┌──────────┐ ┌─────────┐ ┌───────────────┐   │
│  │ Pandas   │ │ Sklearn │ │ Matplotlib    │   │
│  │ NumPy    │ │ SciPy   │ │ Seaborn       │   │
│  │ PyArrow  │ │ RDKit   │ │ IPython       │   │
│  └──────────┘ └─────────┘ └───────────────┘   │
└───────────────────────────────────────────────┘
                     ▲
                     │
┌───────────────────────────────────────────────┐
│              Python Runtime                   │
│  ┌────────────────────────────────────────┐   │
│  │       Python 3.7+ Interpreter          │   │
│  └────────────────────────────────────────┘   │
└───────────────────────────────────────────────┘
                     ▲
                     │
┌───────────────────────────────────────────────┐
│               Operating System                │
└───────────────────────────────────────────────┘
```

## Performance Considerations

### Memory Management

For handling the large gene expression dataset:

```python
# Efficient loading with column selection
specific_genes = ['gene1', 'gene2', '...']
df = pd.read_parquet('data.parquet', columns=specific_genes)

# Chunking for large operations
chunk_size = 100
for chunk in pd.read_parquet('data.parquet', chunksize=chunk_size):
    # Process each chunk
    process_chunk(chunk)
```

### Computation Optimization

```python
# Parallel processing for model training
from sklearn.ensemble import RandomForestClassifier

# Use all available cores
model = RandomForestClassifier(n_jobs=-1)
model.fit(X_train, y_train)
```

## Deployment and Execution

### Running Scripts and Notebooks

```bash
# Running analysis scripts
python analyze_data.py

# Running model training
python train_model.py

# Making predictions
python predict.py --test
```

### Environment Reproducibility

```bash
# Creating a virtual environment
python -m venv venv

# Installing dependencies
pip install -r requirements.txt
``` 