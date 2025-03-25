# HackBio: Single-Cell Perturbations Analysis

## Overview 
This repository contains a comprehensive toolkit for analyzing single-cell perturbation data from the NeurIPS 2023 Competition. The project focuses on understanding how different drugs (small molecules) affect various cell types at the gene expression level.

## Project Structure
```
.
├── dataset/              # Contains the main dataset files
├── docs/                 # Documentation and additional resources
├── plots/                # Generated visualizations and plots
├── analyze_data.py       # Core data analysis functions
├── visualize_data.py     # Data visualization utilities
├── dataset.ipynb         # Main notebook for data processing
├── exploration.ipynb     # Interactive data exploration notebook
└── data_summary.txt     # Dataset statistics and metadata
```

## Dataset Overview
The project works with a rich single-cell perturbation dataset that includes:
- 501 total samples
- 119 unique drugs
- 6 unique cell types
- 18,211 genes measured
- 12 control samples and 489 treatment samples

The main dataset is stored in `dataset/de_train_split.parquet` and contains gene expression data after drug perturbations.

## Key Components

### Data Processing
- `dataset.ipynb`: A comprehensive notebook that demonstrates:
  - Loading and parsing the parquet dataset
  - Converting data to various formats (CSV, Tensors)
  - Basic data preprocessing steps

### Analysis Tools
- `analyze_data.py`: Contains functions for:
  - Statistical analysis of gene expression
  - Drug response analysis
  - Cell type-specific effects

### Visualization
- `visualize_data.py`: Provides tools for:
  - Gene expression visualization
  - Drug response plots
  - Cell type comparison charts
- Generated plots are saved in the `plots/` directory

### Exploration
- `exploration.ipynb`: An interactive notebook for:
  - Data exploration
  - Hypothesis testing
  - Quick visualizations

## Getting Started

1. Clone the repository:
```bash
git clone https://github.com/jaffarkeikei/HackBio.git
cd HackBio
```

2. Install required dependencies (recommended to use a virtual environment):
```bash
pip install pandas numpy matplotlib seaborn scikit-learn
```

3. Start with `dataset.ipynb` to understand the data structure and basic processing steps.

## Contributing
Feel free to contribute to this project by:
- Adding new analysis methods
- Improving visualizations
- Enhancing documentation
- Reporting issues

## License
This project is open-source and available under the MIT License.
