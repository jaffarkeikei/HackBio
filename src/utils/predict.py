import pandas as pd
import numpy as np
import pickle
import os
import argparse
from rdkit import Chem
from rdkit.Chem import Descriptors

def load_model(model_path):
    """Load a pickle model file"""
    with open(model_path, 'rb') as f:
        model = pickle.load(f)
    return model

def predict_cell_type(gene_expression, model):
    """
    Predict cell type from gene expression data
    
    Parameters:
    -----------
    gene_expression : pandas DataFrame
        Gene expression data (genes in columns)
    model : sklearn model
        Trained cell type classification model
        
    Returns:
    --------
    list
        Predicted cell types
    """
    # Make prediction
    cell_types = model.predict(gene_expression)
    return cell_types

def predict_drug_response(drug_names, cell_types, model):
    """
    Predict gene expression response to drugs
    
    Parameters:
    -----------
    drug_names : list
        Names of drugs
    cell_types : list
        Cell types
    model : DrugResponsePredictor
        Trained drug response model
        
    Returns:
    --------
    pandas DataFrame
        Predicted gene expression responses
    """
    # Create input data
    X = pd.DataFrame({
        'cell_type': cell_types,
        'sm_name': drug_names
    })
    
    # Make prediction
    predictions = model.predict(X)
    return predictions

def main():
    # Parse command-line arguments
    parser = argparse.ArgumentParser(description='Predict drug responses')
    parser.add_argument('--test', action='store_true', help='Run predictions on test data')
    parser.add_argument('--input', type=str, help='Path to input data CSV file')
    parser.add_argument('--output', type=str, default='predictions.csv', help='Path to output CSV file')
    args = parser.parse_args()
    
    # Check if models directory exists
    if not os.path.exists('models'):
        print("Error: 'models' directory not found. Please run train_model.py first.")
        return
    
    # Load trained models
    print("Loading models...")
    try:
        cell_type_model = load_model('models/cell_type_classifier.pkl')
        drug_response_model = load_model('models/drug_response_model.pkl')
        with open('models/top_variable_genes.pkl', 'rb') as f:
            top_genes = pickle.load(f)
    except FileNotFoundError as e:
        print(f"Error: {e}")
        print("Please run train_model.py to generate models first.")
        return
    
    # If running on test data
    if args.test:
        print("Loading test data...")
        try:
            test_df = pd.read_parquet('dataset/de_test_split.parquet')
            print(f"Test data loaded with shape: {test_df.shape}")
        except FileNotFoundError:
            print("Error: Test data not found at 'dataset/de_test_split.parquet'")
            return
        
        # Extract features
        metadata_cols = ['cell_type', 'sm_name', 'sm_lincs_id', 'SMILES', 'control']
        gene_cols = [col for col in test_df.columns if col not in metadata_cols]
        
        # Handle case if not all top genes are in test data
        valid_genes = [gene for gene in top_genes if gene in gene_cols]
        
        # Predict cell types
        print("Predicting cell types...")
        cell_type_predictions = predict_cell_type(test_df[valid_genes], cell_type_model)
        
        # Predict drug responses
        print("Predicting drug responses...")
        response_predictions = predict_drug_response(
            test_df['sm_name'].tolist(),
            cell_type_predictions,
            drug_response_model
        )
        
        # Create output
        output_df = pd.DataFrame({
            'sample_id': range(len(test_df)),
            'drug': test_df['sm_name'],
            'actual_cell_type': test_df['cell_type'],
            'predicted_cell_type': cell_type_predictions
        })
        
        # Add gene expression predictions
        for i, gene in enumerate(drug_response_model.y_test.columns):
            output_df[f'pred_{gene}'] = response_predictions[:, i]
        
        # Save predictions
        output_df.to_csv(args.output, index=False)
        print(f"Predictions saved to {args.output}")
        
        # Calculate accuracy
        accuracy = (output_df['actual_cell_type'] == output_df['predicted_cell_type']).mean()
        print(f"Cell type prediction accuracy: {accuracy:.4f}")
    
    # If using custom input data
    elif args.input:
        try:
            input_df = pd.read_csv(args.input)
            print(f"Input data loaded with shape: {input_df.shape}")
        except FileNotFoundError:
            print(f"Error: Input file not found at '{args.input}'")
            return
        
        # TODO: Implement custom prediction logic based on input format
        print("Custom input prediction not yet implemented")
    
    else:
        # If no input is provided, run an example prediction
        print("Running example prediction...")
        
        # Create example data
        example_data = {
            'drug_name': ['Clotrimazole', 'Triptolide', 'Mitoxantrone'],
            'cell_type': ['NK cells', 'T cells CD4+', 'B cells']
        }
        example_df = pd.DataFrame(example_data)
        
        # Predict drug responses
        response_predictions = predict_drug_response(
            example_df['drug_name'].tolist(),
            example_df['cell_type'].tolist(),
            drug_response_model
        )
        
        # Create output
        output_df = example_df.copy()
        for i, gene in enumerate(drug_response_model.y_test.columns[:5]):  # Show first 5 genes only
            output_df[f'pred_{gene}'] = response_predictions[:, i]
        
        # Display predictions
        print("\nExample predictions:")
        print(output_df)
        
        # Save predictions
        output_df.to_csv(args.output, index=False)
        print(f"Example predictions saved to {args.output}")

if __name__ == "__main__":
    main() 