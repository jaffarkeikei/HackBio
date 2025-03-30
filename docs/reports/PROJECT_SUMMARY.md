# HackBio Project Summary

## Project Overview

This project involves creating a machine learning pipeline for analyzing gene expression data in response to various small molecules. The goal is to predict gene expression changes based on molecular properties and cell types, which can help identify potential therapeutic compounds.

## Pipeline Steps

The final analysis pipeline includes the following steps:

1. **Molecular Analysis**: Processing SMILES strings to extract molecular descriptors
2. **Data Preprocessing**: Cleaning, normalization, and feature engineering
3. **Model Training**: 
   - Ensemble approach combining Random Forest and XGBoost
   - Advanced neural network models (1D CNN, LSTM, GRU)
4. **Dimensionality Reduction**: SVD for reducing gene expression dimensions
5. **Evaluation**: Performance assessment using R² and MSE metrics
6. **Visualization**: Creating plots for model performance and predictions

## Key Challenges Addressed

Throughout the development process, we encountered and resolved several challenges:

1. **Data Structure**: Handling both categorical and numerical data required proper encoding
2. **Feature Names**: Fixed compatibility issues with XGBoost by cleaning column names
3. **Multi-output Regression**: Implemented proper handling for predicting multiple gene expressions
4. **Error Handling**: Added robust logging and error tracking
5. **High Dimensionality**: Implemented SVD following the NeurIPS 2023 winning approach

## Implemented Approaches

### Traditional Machine Learning (final_analysis.py)
- Random Forest and XGBoost models with multi-output regression
- One-hot encoding of categorical features
- Statistical feature engineering
- Performance: R² ≈ 0.33

### Advanced Neural Networks (advanced_model.py)
- SVD dimensionality reduction (following 2nd place NeurIPS 2023 winner)
- 1D CNN model for sequence modeling
- LSTM and GRU models (following 1st place NeurIPS 2023 winner)
- Model ensembling for improved performance

## Results

The pipeline successfully:

- Processes 501 compounds with their SMILES strings
- Encodes 4 categorical features into 363 binary features
- Trains models on the top variable genes
- Combines predictions from multiple models
- Implements dimensionality reduction for efficient computation

## Visualizations

The analysis produces several key visualizations:

- Gene expression distributions
- Model performance comparisons
- Prediction accuracy plots
- Feature importance charts
- R² score distributions
- Training and validation loss curves

## Future Work

To further enhance this project, consider:

1. **Hyperparameter Optimization**: Fine-tune model parameters using grid search
2. **Feature Selection**: Identify most predictive molecular descriptors
3. **Cross-validation**: Implement k-fold cross-validation as used in winning solutions
4. **Transfer Learning**: Pre-train models on related biological datasets
5. **Alternative Architectures**: Try transformer-based approaches (2nd place winner)

## Conclusion

This project demonstrates multiple effective approaches to predicting gene expression from molecular structures and cell types. Following the winning approaches from NeurIPS 2023, we implemented both traditional machine learning and neural network solutions, providing a comprehensive framework for gene expression prediction.

## Usage

To run the final analysis:

```bash
python final_analysis.py          # Traditional ML approach
python advanced_model.py          # Neural network approach
```

This will generate models, visualizations, and performance metrics in their respective directories. 