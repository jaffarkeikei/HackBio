import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
from train_model import CellTypeClassifier, DrugResponsePredictor
import seaborn as sns
import pickle

def generate_synthetic_data(n_samples=500, n_genes=100, n_cell_types=3, n_drugs=20):
    """
    Generate synthetic data for testing.
    
    Parameters:
    -----------
    n_samples : int
        Number of samples to generate
    n_genes : int
        Number of genes
    n_cell_types : int
        Number of cell types
    n_drugs : int
        Number of drugs
        
    Returns:
    --------
    pandas DataFrame
        Synthetic dataset
    """
    # Generate gene names
    gene_cols = [f"gene_{i}" for i in range(1, n_genes + 1)]
    
    # Generate cell types
    cell_types = [f"cell_type_{i}" for i in range(1, n_cell_types + 1)]
    
    # Generate drug names
    drugs = [f"drug_{i}" for i in range(1, n_drugs + 1)]
    
    # Create dataframe
    df = pd.DataFrame()
    
    # Assign random cell types and drugs
    df['cell_type'] = np.random.choice(cell_types, size=n_samples)
    df['sm_name'] = np.random.choice(drugs, size=n_samples)
    
    # Add control column (10% of samples are controls)
    df['control'] = np.random.choice([0, 1], size=n_samples, p=[0.9, 0.1])
    
    # Generate gene expression data
    # We'll create a base expression for each cell type
    cell_type_base = {}
    for cell in cell_types:
        cell_type_base[cell] = np.random.normal(0, 1, size=n_genes)
    
    # Generate drug effects
    drug_effects = {}
    for drug in drugs:
        drug_effects[drug] = np.random.normal(0, 0.5, size=n_genes)
    
    # Generate expression data
    for i in range(n_samples):
        cell = df.iloc[i]['cell_type']
        drug = df.iloc[i]['sm_name']
        is_control = df.iloc[i]['control']
        
        # Base expression for cell type
        base_expr = cell_type_base[cell]
        
        # Add drug effect if not control
        if not is_control:
            drug_effect = drug_effects[drug]
            expr = base_expr + drug_effect
        else:
            expr = base_expr
        
        # Add noise
        expr += np.random.normal(0, 0.2, size=n_genes)
        
        # Add to dataframe
        df.loc[i, gene_cols] = expr
    
    return df

def test_cell_type_classifier(df):
    """
    Test the cell type classifier.
    
    Parameters:
    -----------
    df : pandas DataFrame
        Input dataset
    """
    print("\n=== Testing Cell Type Classifier ===")
    classifier = CellTypeClassifier(n_top_genes=50, model_type='rf')
    
    # Preprocess data
    X, y = classifier.preprocess_data(df)
    
    # Split data for demonstration purposes
    # In the main train_model.py, this is handled inside train()
    from sklearn.model_selection import train_test_split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Train model (with a small CV to make it faster)
    classifier.train(X_train, y_train, cv=3)
    
    # Evaluate
    results = classifier.evaluate(X_test, y_test)
    
    # Plot confusion matrix
    fig = classifier.plot_confusion_matrix(results)
    
    # Plot feature importance
    classifier.plot_feature_importance(top_n=10)
    
    # Save model
    os.makedirs('models', exist_ok=True)
    classifier.save_model('models/test_cell_type_classifier.pkl')
    
    print(f"Cell type classification accuracy: {results['accuracy']:.4f}")
    return classifier

def test_drug_response_predictor(df):
    """
    Test the drug response predictor.
    
    Parameters:
    -----------
    df : pandas DataFrame
        Input dataset
    """
    print("\n=== Testing Drug Response Predictor ===")
    predictor = DrugResponsePredictor(n_top_genes=50, model_type='rf')
    
    # Preprocess data
    X_train, X_test, y_train, y_test = predictor.preprocess_data(df, test_size=0.2)
    
    # Train model
    predictor.train(X_train, y_train)
    
    # Evaluate
    results = predictor.evaluate(X_test, y_test)
    
    # Plot prediction performance
    predictor.plot_prediction_performance(results, n_genes=3)
    
    # Save model
    os.makedirs('models', exist_ok=True)
    predictor.save_model('models/test_drug_response_predictor.pkl')
    
    print(f"Drug response prediction R²: {results['r2']:.4f}")
    return predictor

def main():
    """
    Main function to run the tests.
    """
    # Create output directories
    os.makedirs('plots', exist_ok=True)
    os.makedirs('models', exist_ok=True)
    
    # Generate synthetic data
    print("Generating synthetic data...")
    df = generate_synthetic_data(n_samples=500, n_genes=100, n_cell_types=3, n_drugs=20)
    print(f"Generated dataset with shape: {df.shape}")
    
    # Save synthetic data for reference
    df.to_csv('test_synthetic_data.csv', index=False)
    print("Saved synthetic data to test_synthetic_data.csv")
    
    # Test cell type classifier
    classifier = test_cell_type_classifier(df)
    
    # Test drug response predictor
    predictor = test_drug_response_predictor(df)
    
    print("\nAll tests completed successfully!")
    print("\nGenerated files:")
    print("- models/test_cell_type_classifier.pkl")
    print("- models/test_drug_response_predictor.pkl")
    print("- plots/cell_type_confusion_matrix.png")
    print("- plots/cell_type_feature_importance.png")
    print("- plots/drug_response_predictions.png")
    
    return df, classifier, predictor

if __name__ == "__main__":
    df, classifier, predictor = main() 