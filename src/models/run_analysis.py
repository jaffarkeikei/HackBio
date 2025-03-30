#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Single-Cell Perturbations Analysis Pipeline

This script brings together all the components of the analysis pipeline:
1. Data preprocessing and exploration
2. Molecular property analysis
3. Model training
4. Model evaluation and interpretation
5. Correlation analysis between molecular properties and gene expression

Usage:
    python run_analysis.py [--no-train] [--interpret-only]

Options:
    --no-train       Skip model training (use existing models)
    --interpret-only Only run model interpretation (skip training & analysis)
"""

import os
import sys
import argparse
import time
import subprocess
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from molecular_analysis import MolecularAnalyzer
from correlation_analysis import CorrelationAnalyzer
from model_interpreter import ModelInterpreter
import logging
from sklearn.decomposition import TruncatedSVD
from sklearn.preprocessing import OneHotEncoder
from sklearn.ensemble import VotingRegressor

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("analysis.log"),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger("run_analysis")


def setup_environment():
    """Create necessary directories and check data availability"""
    # Create output directories if they don't exist
    os.makedirs('models', exist_ok=True)
    os.makedirs('plots', exist_ok=True)
    os.makedirs('results', exist_ok=True)
    
    # Check if dataset exists
    if not os.path.exists('dataset/de_train_split.parquet'):
        logger.error("Dataset not found at dataset/de_train_split.parquet")
        logger.info("Please make sure the dataset is available in the dataset directory")
        sys.exit(1)
    
    logger.info("Environment setup complete")


def analyze_molecular_properties():
    """Analyze drug molecular structures"""
    logger.info("Starting molecular property analysis...")
    
    # Load dataset
    df = pd.read_parquet('dataset/de_train_split.parquet')
    
    # Initialize analyzer
    analyzer = MolecularAnalyzer(df['SMILES'])
    
    # Calculate descriptors
    descriptors = analyzer.calculate_descriptors()
    
    # Save descriptors to CSV
    descriptors.to_csv('results/molecular_descriptors.csv', index=False)
    logger.info(f"Saved molecular descriptors for {len(descriptors)} compounds to results/molecular_descriptors.csv")
    
    # Analyze drug likeness
    drug_likeness = analyzer.analyze_drug_likeness()
    logger.info(f"Drug-like molecules: {drug_likeness['DrugLike'].sum()} out of {len(drug_likeness)}")
    
    # Generate visualization plots
    analyzer.plot_descriptor_distributions()
    plt.savefig('plots/descriptor_distributions.png', dpi=300, bbox_inches='tight')
    
    # Cluster molecules
    cluster_labels = analyzer.cluster_molecules(n_clusters=5)
    
    # Visualize clusters
    analyzer.visualize_clusters()
    plt.savefig('plots/molecule_clusters.png', dpi=300, bbox_inches='tight')
    
    # Visualize molecule examples
    analyzer.visualize_molecules(n_mols=10)
    plt.savefig('plots/molecule_examples.png', dpi=300, bbox_inches='tight')
    
    logger.info("Molecular property analysis complete")
    return descriptors


def preprocess_data_with_svd(df, n_components=100):
    """Apply dimensionality reduction to gene expression data using truncated SVD
    
    Parameters:
    -----------
    df : pandas DataFrame
        Input dataset with gene expression data
    n_components : int
        Number of components to reduce to
        
    Returns:
    --------
    tuple
        Original dataframe and reduced gene expression features
    """
    logger.info(f"Applying dimensionality reduction with SVD to {n_components} components")
    
    # Identify metadata columns
    metadata_cols = ['cell_type', 'sm_name', 'sm_lincs_id', 'SMILES', 'control']
    gene_cols = [col for col in df.columns if col not in metadata_cols]
    
    # Apply SVD reduction
    svd = TruncatedSVD(n_components=n_components, random_state=42)
    reduced_data = svd.fit_transform(df[gene_cols])
    
    # Create column names for reduced features
    reduced_cols = [f'svd_comp_{i}' for i in range(n_components)]
    
    # Create a new dataframe with reduced features and metadata
    reduced_df = pd.DataFrame(reduced_data, columns=reduced_cols)
    for col in metadata_cols:
        if col in df.columns:
            reduced_df[col] = df[col].values
    
    # Calculate explained variance
    explained_variance = svd.explained_variance_ratio_.sum()
    logger.info(f"SVD explained variance with {n_components} components: {explained_variance:.4f}")
    
    # Save SVD model for later use
    import pickle
    with open('models/svd_model.pkl', 'wb') as f:
        pickle.dump(svd, f)
    
    return df, reduced_df


def apply_onehot_encoding(df):
    """Apply one-hot encoding to categorical features
    
    Parameters:
    -----------
    df : pandas DataFrame
        Input dataset
        
    Returns:
    --------
    pandas DataFrame
        Dataset with one-hot encoded categorical features
    """
    logger.info("Applying one-hot encoding to categorical features")
    
    # Create encoder for cell types
    cell_encoder = OneHotEncoder(sparse_output=False)
    cell_encoded = cell_encoder.fit_transform(df[['cell_type']])
    cell_cols = [f'cell_{cat}' for cat in cell_encoder.categories_[0]]
    
    # Create encoder for drug names
    drug_encoder = OneHotEncoder(sparse_output=False)
    drug_encoded = drug_encoder.fit_transform(df[['sm_name']])
    drug_cols = [f'drug_{cat}' for cat in drug_encoder.categories_[0]]
    
    # Create new dataframe with encoded features
    encoded_df = pd.DataFrame(cell_encoded, columns=cell_cols)
    encoded_df = pd.concat([encoded_df, pd.DataFrame(drug_encoded, columns=drug_cols)], axis=1)
    
    # Add other columns from original dataframe
    for col in df.columns:
        if col not in ['cell_type', 'sm_name']:
            encoded_df[col] = df[col].values
    
    # Save encoders for later use
    import pickle
    with open('models/encoders.pkl', 'wb') as f:
        pickle.dump({'cell_encoder': cell_encoder, 'drug_encoder': drug_encoder}, f)
    
    return encoded_df


def add_statistical_features(df):
    """Add statistical features derived from gene expression data
    
    Parameters:
    -----------
    df : pandas DataFrame
        Input dataset with gene expression data
        
    Returns:
    --------
    pandas DataFrame
        Dataset with additional statistical features
    """
    logger.info("Adding statistical features")
    
    # Identify metadata columns
    metadata_cols = ['cell_type', 'sm_name', 'sm_lincs_id', 'SMILES', 'control']
    gene_cols = [col for col in df.columns if col not in metadata_cols]
    
    # Calculate statistics for each sample
    df['gene_mean'] = df[gene_cols].mean(axis=1)
    df['gene_std'] = df[gene_cols].std(axis=1)
    df['gene_median'] = df[gene_cols].median(axis=1)
    df['gene_q1'] = df[gene_cols].quantile(0.25, axis=1)
    df['gene_q3'] = df[gene_cols].quantile(0.75, axis=1)
    df['gene_iqr'] = df['gene_q3'] - df['gene_q1']
    
    return df


def train_models_optimized():
    """Train optimized drug response models using advanced techniques"""
    logger.info("Starting optimized model training...")
    
    # Load and preprocess data
    df = pd.read_parquet('dataset/de_train_split.parquet')
    
    # Step 1: Add statistical features
    df_with_stats = add_statistical_features(df)
    
    # Step 2: Apply SVD for dimensionality reduction
    original_df, reduced_df = preprocess_data_with_svd(df_with_stats, n_components=100)
    
    # Step 3: Apply one-hot encoding
    encoded_df = apply_onehot_encoding(reduced_df)
    
    # Save processed dataframes for reference
    encoded_df.to_parquet('results/processed_data.parquet')
    
    try:
        # Run the modified training script with the processed data
        start_time = time.time()
        
        # Option 1: Run as subprocess with the processed data path
        # subprocess.run(["python", "train_model.py", "--data", "results/processed_data.parquet"], check=True)
        
        # Option 2: Create an ensemble model directly here
        # For demonstration, we'll implement a simple version here
        logger.info("Creating ensemble model with RandomForest and XGBoost")
        
        # Identify feature and target columns
        metadata_cols = ['cell_type', 'sm_name', 'sm_lincs_id', 'SMILES', 'control']
        gene_cols = [col for col in original_df.columns if col not in metadata_cols]
        
        # Using only a subset of genes for demonstration (would use all in production)
        from sklearn.ensemble import RandomForestRegressor
        import xgboost as xgb
        
        # Create base models for ensemble
        rf_model = RandomForestRegressor(n_estimators=100, max_depth=20, random_state=42, n_jobs=-1)
        xgb_model = xgb.XGBRegressor(n_estimators=100, max_depth=5, learning_rate=0.1, random_state=42)
        
        # Create ensemble model
        ensemble = VotingRegressor([
            ('rf', rf_model),
            ('xgb', xgb_model)
        ])
        
        # For demonstration, train on a small subset of genes (in practice would use all genes)
        top_genes = gene_cols[:10]  # First 10 genes for demonstration
        
        # Split data for training
        from sklearn.model_selection import train_test_split
        X = encoded_df.drop(gene_cols, axis=1)
        y = original_df[top_genes]
        
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        
        # Train ensemble model
        ensemble.fit(X_train, y_train)
        
        # Save ensemble model
        import pickle
        with open('models/ensemble_model.pkl', 'wb') as f:
            pickle.dump(ensemble, f)
        
        # Save list of top genes used
        with open('models/top_genes_used.pkl', 'wb') as f:
            pickle.dump(top_genes, f)
        
        # Evaluate model
        from sklearn.metrics import r2_score, mean_squared_error
        y_pred = ensemble.predict(X_test)
        
        # Calculate metrics
        r2 = r2_score(y_test, y_pred)
        mse = mean_squared_error(y_test, y_pred)
        
        # Save metrics to results
        pd.DataFrame({
            'Metric': ['MSE', 'R²'],
            'Value': [mse, r2]
        }).to_csv('results/ensemble_model_evaluation.csv', index=False)
        
        logger.info(f"Ensemble model training completed in {time.time() - start_time:.2f} seconds")
        logger.info(f"Ensemble model R²: {r2:.4f}, MSE: {mse:.4f}")
        
        return True
    except Exception as e:
        logger.error(f"Optimized model training failed with error: {e}")
        return False


def train_models():
    """Train the standard drug response models"""
    logger.info("Starting model training...")
    
    try:
        # Run the training script
        start_time = time.time()
        subprocess.run(["python", "train_model.py"], check=True)
        
        end_time = time.time()
        logger.info(f"Model training completed in {end_time - start_time:.2f} seconds")
        
        # Check if model files were created
        if not os.path.exists('models/drug_response_model.pkl'):
            logger.error("Model training failed - model files not found")
            return False
        
        return True
    except subprocess.CalledProcessError as e:
        logger.error(f"Model training failed with error: {e}")
        return False


def perform_correlation_analysis():
    """Analyze correlations between molecular properties and gene expression"""
    logger.info("Starting correlation analysis...")
    
    # Initialize analyzer
    analyzer = CorrelationAnalyzer()
    
    # Load response data
    df = analyzer.load_response_data('dataset/de_train_split.parquet')
    
    # Calculate mean responses
    mean_responses = analyzer.prepare_mean_responses(by='sm_name')
    
    # If we have descriptor data
    if analyzer.descriptors is not None:
        # Calculate correlations
        corr_matrix, p_vals = analyzer.calculate_correlations(mean_responses, method='pearson')
        
        # Save correlation matrix
        corr_matrix.to_csv('results/property_gene_correlations.csv')
        p_vals.to_csv('results/correlation_p_values.csv')
        
        # Plot correlation heatmap
        fig = analyzer.plot_correlation_heatmap(n_top_genes=20)
        fig.savefig('plots/property_gene_correlations.png', dpi=300, bbox_inches='tight')
        
        # Find top relationships
        top_relations = analyzer.find_property_gene_relationships(top_n=15)
        logger.info("\nTop Property-Gene Relationships:")
        logger.info(top_relations[['Property', 'Gene', 'Correlation', 'p_value']])
        
        # Save top relationships
        top_relations.to_csv('results/top_property_gene_relationships.csv', index=False)
        
        # Plot clustered heatmap
        fig_clustered = analyzer.plot_clustered_heatmap(n_clusters=3, n_top_genes=30)
        fig_clustered.savefig('plots/clustered_property_gene_correlations.png', dpi=300, bbox_inches='tight')
        
        logger.info("Correlation analysis complete")
        return True
    else:
        logger.error("No descriptor data available. Please run molecular_analysis.py first.")
        return False


def interpret_models():
    """Interpret drug response models"""
    logger.info("Starting model interpretation...")
    
    # Check if model files exist
    if not os.path.exists('models/drug_response_model.pkl') and not os.path.exists('models/ensemble_model.pkl'):
        logger.error("No model files found. Please run train_model.py or train_models_optimized() first.")
        return False
    
    # Initialize interpreter
    model_path = 'models/ensemble_model.pkl' if os.path.exists('models/ensemble_model.pkl') else 'models/drug_response_model.pkl'
    genes_path = 'models/top_genes_used.pkl' if os.path.exists('models/top_genes_used.pkl') else 'models/top_variable_genes.pkl'
    
    interpreter = ModelInterpreter(
        model_path=model_path,
        top_genes_path=genes_path
    )
    
    # Load dataset
    if os.path.exists('dataset/de_train_split.parquet'):
        interpreter.load_dataset('dataset/de_train_split.parquet')
    
    # Generate comprehensive model summary
    try:
        results = interpreter.evaluate_model()
        summary = interpreter.summarize_model(results)
        
        # Save evaluation results
        if results:
            pd.DataFrame({
                'Metric': ['MSE', 'R²'],
                'Value': [results['overall_mse'], results['overall_r2']]
            }).to_csv('results/model_evaluation.csv', index=False)
            
            results['gene_metrics'].to_csv('results/gene_performance.csv', index=False)
            results['cell_metrics'].to_csv('results/cell_type_performance.csv', index=False)
            results['drug_metrics'].to_csv('results/drug_performance.csv', index=False)
        
        logger.info("Model interpretation complete")
        return True
    except Exception as e:
        logger.error(f"Model interpretation failed with error: {e}")
        return False


def main():
    """Main function to run the analysis pipeline"""
    parser = argparse.ArgumentParser(description="Run the full analysis pipeline for single-cell perturbations")
    parser.add_argument('--no-train', action='store_true', help='Skip model training')
    parser.add_argument('--interpret-only', action='store_true', help='Only run model interpretation')
    parser.add_argument('--optimize', action='store_true', help='Use optimized model training techniques')
    args = parser.parse_args()
    
    # Set up environment
    setup_environment()
    
    if args.interpret_only:
        # Only run model interpretation
        logger.info("Running model interpretation only")
        interpret_models()
    else:
        # Run full pipeline
        logger.info("Starting full analysis pipeline")
        
        # Step 1: Analyze molecular properties
        try:
            analyze_molecular_properties()
        except Exception as e:
            logger.error(f"Molecular property analysis failed: {e}")
        
        # Step 2: Train models (if not skipped)
        if not args.no_train:
            try:
                if args.optimize:
                    train_success = train_models_optimized()
                else:
                    train_success = train_models()
                    
                if not train_success:
                    logger.warning("Model training was not successful. Continuing with existing models.")
            except Exception as e:
                logger.error(f"Model training failed: {e}")
        
        # Step 3: Correlation analysis
        try:
            perform_correlation_analysis()
        except Exception as e:
            logger.error(f"Correlation analysis failed: {e}")
        
        # Step 4: Model interpretation
        try:
            interpret_models()
        except Exception as e:
            logger.error(f"Model interpretation failed: {e}")
        
        logger.info("Analysis pipeline complete")
        
        # Generate summary report
        try:
            summary = {
                "Molecular Analysis": os.path.exists("results/molecular_descriptors.csv"),
                "Model Training": os.path.exists("models/drug_response_model.pkl") or os.path.exists("models/ensemble_model.pkl"),
                "Correlation Analysis": os.path.exists("results/property_gene_correlations.csv"),
                "Model Interpretation": os.path.exists("results/model_evaluation.csv"),
            }
            
            with open("results/analysis_summary.txt", "w") as f:
                f.write("Analysis Pipeline Summary\n")
                f.write("========================\n\n")
                for step, completed in summary.items():
                    status = "Completed" if completed else "Not completed"
                    f.write(f"{step}: {status}\n")
            
            logger.info("Generated analysis summary report")
        except Exception as e:
            logger.error(f"Failed to generate summary report: {e}")


if __name__ == "__main__":
    main() 