# HackBio: Single-Cell Perturbations Analysis

## Overview 
This repository contains a comprehensive toolkit for analyzing single-cell perturbation data from the NeurIPS 2023 Competition. The project focuses on understanding how different drugs (small molecules) affect various cell types at the gene expression level.

## Architecture
The project follows a modular architecture designed for analyzing single-cell perturbation data:

- **System Architecture**: The architecture diagrams can be found in the [technical documentation](/docs/architecture/technical_architecture.md#system-overview)
- **Data Flow**: View the data flow in the [technical documentation](/docs/architecture/technical_architecture.md#data-flow-architecture)
- **Analysis Pipeline**: Review the complete workflow in the [technical documentation](/docs/architecture/technical_architecture.md#analysis-pipeline)

The documentation includes interactive diagrams that show how the different components of the system work together.

## Project Structure
```
.
├── dataset/                 # Contains the main dataset files
├── docs/                    # Documentation and additional resources
├── plots/                   # Generated visualizations and plots
├── models/                  # Trained ML models
├── results/                 # Analysis results
├── analyze_data.py          # Core data analysis functions
├── visualize_data.py        # Data visualization utilities
├── molecular_analysis.py    # Analysis of drug molecular structures
├── correlation_analysis.py  # Correlations between drug properties and gene expressions
├── model_interpreter.py     # Tools for interpreting predictive models
├── train_model.py           # Train gene expression prediction models
├── predict.py               # Make predictions with trained models
├── run_analysis.py          # Full analysis pipeline
├── dataset.ipynb            # Main notebook for data processing
├── exploration_notebook.ipynb # Interactive data exploration notebook
└── data_summary.txt         # Dataset statistics and metadata
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

### Data Analysis
- `molecular_analysis.py`: Analyzes drug structures using RDKit
  - Calculates molecular descriptors
  - Generates molecular fingerprints
  - Clusters drugs based on structural similarity
  - Visualizes molecular structures
  
- `correlation_analysis.py`: Identifies relationships between drug properties and gene expression
  - Calculates correlations between molecular properties and gene responses
  - Identifies genes most affected by specific drug properties
  - Visualizes property-gene correlation patterns
  - Clusters properties by their expression effects

### Model Training & Prediction
- `train_model.py`: Trains models for:
  - Cell type classification from gene expression
  - Drug response prediction based on drug and cell type
  
- `predict.py`: Uses trained models to predict:
  - Cell types from gene expression
  - Gene expression changes in response to drug treatments
  - Includes a command-line interface for batch predictions

### Model Interpretation
- `model_interpreter.py`: Tools for understanding model behavior
  - Evaluates model performance across genes, cell types, and drugs
  - Calculates feature importance
  - Generates SHAP explanations
  - Visualizes prediction errors and model performance

### Complete Pipeline
- `run_analysis.py`: Brings together all components
  - Runs the full analysis pipeline including molecular analysis, model training, correlation analysis, and model interpretation
  - Allows skipping specific steps with command-line arguments
  - Generates comprehensive reports and visualizations

## Getting Started

### Prerequisites
- Python 3.8+
- Required packages listed in `requirements.txt`

### Installation

1. Clone the repository:
```bash
git clone https://github.com/jaffarkeikei/HackBio.git
cd HackBio
```

2. Install required dependencies (recommended to use a virtual environment):
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
pip install -r requirements.txt
```

3. Download the dataset and place it in the `dataset/` directory.

### Running the Analysis

#### Full Pipeline
To run the complete analysis pipeline:
```bash
python run_analysis.py
```

#### Skip Model Training
If you already have trained models:
```bash
python run_analysis.py --no-train
```

#### Model Interpretation Only
To only interpret existing models:
```bash
python run_analysis.py --interpret-only
```

#### Make Predictions
To use trained models for prediction:
```bash
# Predict on test data
python predict.py --test

# Predict on custom data
python predict.py --input custom_data.csv --output predictions.csv
```

### Exploring the Data
Start with `exploration_notebook.ipynb` to interactively explore:
- Dataset structure and basic statistics
- Drug molecular property analysis
- Model predictions and evaluations
- Correlation patterns between drug properties and gene expression

## Key Analyses

### Molecular Structure Analysis
- Calculation of drug-like properties based on Lipinski's Rule of Five
- Clustering of drugs based on structural similarity
- Visualization of molecular structures and property distributions

### Gene Expression Prediction
- Random Forest models to predict gene expression changes in response to drugs
- Cell type classification from gene expression profiles
- Evaluations of model performance by gene, cell type, and drug

### Structure-Activity Relationships
- Correlation analysis between drug molecular properties and gene expression changes
- Identification of genes most affected by specific molecular properties
- Clustering of molecular properties with similar gene expression effects

## Results
Analysis results are saved in the following locations:
- `plots/`: Visualizations and figures
- `models/`: Trained machine learning models
- `results/`: Numerical results and data tables

## Contributing
Feel free to contribute to this project by:
- Adding new analysis methods
- Improving visualizations
- Enhancing documentation
- Reporting issues

## License
This project is open-source and available under the MIT License.
