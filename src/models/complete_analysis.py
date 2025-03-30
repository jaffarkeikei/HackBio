#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Complete Analysis Pipeline Script

This script breaks down the analysis pipeline into smaller steps with enhanced error handling.
Each step will be executed separately and results will be saved after each step.

Usage:
    python complete_analysis.py [--step STEP_NUMBER]

Options:
    --step STEP_NUMBER    Start from specific step (1: molecular analysis, 2: data preprocessing, 
                          3: model training, 4: correlation analysis, 5: model interpretation)
"""

import os
import sys
import argparse
import time
import subprocess
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import TruncatedSVD
from sklearn.preprocessing import OneHotEncoder
from sklearn.ensemble import VotingRegressor, RandomForestRegressor
import logging
import pickle
import traceback
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_squared_error

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("complete_analysis.log"),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger("complete_analysis")


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


def step1_molecular_analysis():
    """Step 1: Analyze molecular properties of drugs"""
    logger.info("STEP 1: Starting molecular property analysis...")
    
    try:
        # Load dataset
        df = pd.read_parquet('dataset/de_train_split.parquet')
        
        # Import MolecularAnalyzer here to handle import errors gracefully
        from molecular_analysis import MolecularAnalyzer
        
        # Initialize analyzer
        analyzer = MolecularAnalyzer(df['SMILES'])
        
        # Calculate descriptors
        descriptors = analyzer.calculate_descriptors()
        
        # Save descriptors to CSV
        descriptors.to_csv('results/molecular_descriptors.csv', index=False)
        logger.info(f"Saved molecular descriptors for {len(descriptors)} compounds to results/molecular_descriptors.csv")
        
        # Analyze drug likeness
        drug_likeness = analyzer.analyze_drug_likeness()
        drug_likeness.to_csv('results/drug_likeness.csv', index=False)
        logger.info(f"Drug-like molecules: {drug_likeness['DrugLike'].sum()} out of {len(drug_likeness)}")
        
        # Generate visualization plots
        try:
            analyzer.plot_descriptor_distributions()
            plt.savefig('plots/descriptor_distributions.png', dpi=300, bbox_inches='tight')
            plt.close()
            
            # Cluster molecules
            cluster_labels = analyzer.cluster_molecules(n_clusters=5)
            
            # Visualize clusters
            analyzer.visualize_clusters()
            plt.savefig('plots/molecule_clusters.png', dpi=300, bbox_inches='tight')
            plt.close()
            
            # Visualize molecule examples
            analyzer.visualize_molecules(n_mols=10)
            plt.savefig('plots/molecule_examples.png', dpi=300, bbox_inches='tight')
            plt.close()
        except Exception as viz_error:
            logger.error(f"Error in visualization: {viz_error}")
            logger.error(traceback.format_exc())
        
        logger.info("STEP 1: Molecular property analysis complete")
        return True, descriptors
    
    except Exception as e:
        logger.error(f"STEP 1 FAILED: Error in molecular analysis: {e}")
        logger.error(traceback.format_exc())
        return False, None


def step2_data_preprocessing():
    """Step 2: Preprocess data including dimensionality reduction and encoding"""
    logger.info("STEP 2: Starting data preprocessing...")
    
    try:
        # Load dataset
        df = pd.read_parquet('dataset/de_train_split.parquet')
        
        # Add statistical features
        df_with_stats = add_statistical_features(df)
        logger.info("Added statistical features to the dataset")
        
        # Apply SVD for dimensionality reduction
        original_df, reduced_df = apply_svd_reduction(df_with_stats, n_components=100)
        logger.info("Applied SVD dimensionality reduction")
        
        # Apply one-hot encoding
        encoded_df = apply_onehot_encoding(reduced_df)
        logger.info("Applied one-hot encoding to categorical features")
        
        # Save processed data
        encoded_df.to_parquet('results/processed_data.parquet')
        logger.info("Saved preprocessed data to results/processed_data.parquet")
        
        logger.info("STEP 2: Data preprocessing complete")
        return True, original_df, encoded_df
    
    except Exception as e:
        logger.error(f"STEP 2 FAILED: Error in data preprocessing: {e}")
        logger.error(traceback.format_exc())
        return False, None, None


def step3_train_models(original_df=None, encoded_df=None):
    """Step 3: Train ensemble models"""
    logger.info("STEP 3: Starting ensemble model training...")
    
    try:
        # Load data if not provided
        if original_df is None or encoded_df is None:
            if os.path.exists('results/processed_data.parquet'):
                logger.info("Loading preprocessed data from file")
                encoded_df = pd.read_parquet('results/processed_data.parquet')
                original_df = pd.read_parquet('dataset/de_train_split.parquet')
                original_df = add_statistical_features(original_df)
            else:
                logger.error("No preprocessed data available. Please run step2_data_preprocessing first.")
                return False
        
        # Identify metadata and gene columns
        metadata_cols = ['cell_type', 'sm_name', 'sm_lincs_id', 'SMILES', 'control']
        original_gene_cols = [col for col in original_df.columns if col not in metadata_cols and 
                             not col.startswith('gene_') and not col.startswith('svd_comp_')]
        
        # Using a subset of genes for faster training (in production would use all or top variable genes)
        # Select a limited number of genes based on variance
        gene_variance = original_df[original_gene_cols].var().sort_values(ascending=False)
        top_genes = gene_variance.index[:20].tolist()  # Using top 20 genes for demonstration
        
        logger.info(f"Training model on {len(top_genes)} top variable genes")
        
        # Create base models for ensemble
        rf_model = RandomForestRegressor(n_estimators=100, max_depth=20, random_state=42, n_jobs=-1)
        xgb_model = xgb.XGBRegressor(n_estimators=100, max_depth=5, learning_rate=0.1, random_state=42)
        
        # Create ensemble model
        ensemble = VotingRegressor([
            ('rf', rf_model),
            ('xgb', xgb_model)
        ])
        
        # Split data for training
        X = encoded_df.drop(original_gene_cols, axis=1, errors='ignore')
        y = original_df[top_genes]
        
        logger.info(f"Feature matrix shape: {X.shape}, Target matrix shape: {y.shape}")
        
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        
        # Train ensemble model
        start_time = time.time()
        ensemble.fit(X_train, y_train)
        training_time = time.time() - start_time
        
        # Save ensemble model
        with open('models/ensemble_model.pkl', 'wb') as f:
            pickle.dump(ensemble, f)
        
        # Save list of top genes used
        with open('models/top_genes_used.pkl', 'wb') as f:
            pickle.dump(top_genes, f)
        
        # Evaluate model
        y_pred = ensemble.predict(X_test)
        
        # Calculate metrics
        r2_scores = []
        mse_scores = []
        
        for i, gene in enumerate(top_genes):
            r2 = r2_score(y_test.iloc[:, i], y_pred[:, i])
            mse = mean_squared_error(y_test.iloc[:, i], y_pred[:, i])
            r2_scores.append(r2)
            mse_scores.append(mse)
        
        avg_r2 = np.mean(r2_scores)
        avg_mse = np.mean(mse_scores)
        
        # Create and save detailed evaluation results
        gene_metrics = pd.DataFrame({
            'Gene': top_genes,
            'R2': r2_scores,
            'MSE': mse_scores
        })
        
        gene_metrics.to_csv('results/gene_performance_metrics.csv', index=False)
        
        # Save summary metrics
        pd.DataFrame({
            'Metric': ['Average MSE', 'Average R²', 'Training Time (s)'],
            'Value': [avg_mse, avg_r2, training_time]
        }).to_csv('results/ensemble_model_evaluation.csv', index=False)
        
        logger.info(f"Ensemble model training completed in {training_time:.2f} seconds")
        logger.info(f"Average R²: {avg_r2:.4f}, Average MSE: {avg_mse:.4f}")
        
        # Optional: Plot predictions vs actual for a few genes
        try:
            plt.figure(figsize=(12, 8))
            for i, gene in enumerate(top_genes[:4]):  # Plot first 4 genes
                plt.subplot(2, 2, i+1)
                plt.scatter(y_test.iloc[:, i], y_pred[:, i], alpha=0.5)
                plt.plot([y_test.iloc[:, i].min(), y_test.iloc[:, i].max()], 
                         [y_test.iloc[:, i].min(), y_test.iloc[:, i].max()], 'r--')
                plt.title(f'Gene: {gene}, R² = {r2_scores[i]:.4f}')
                plt.xlabel('Actual Expression')
                plt.ylabel('Predicted Expression')
            
            plt.tight_layout()
            plt.savefig('plots/gene_predictions.png', dpi=300, bbox_inches='tight')
            plt.close()
        except Exception as viz_error:
            logger.error(f"Error in prediction visualization: {viz_error}")
        
        logger.info("STEP 3: Model training complete")
        return True
    
    except Exception as e:
        logger.error(f"STEP 3 FAILED: Error in model training: {e}")
        logger.error(traceback.format_exc())
        return False


def step4_correlation_analysis():
    """Step 4: Analyze correlations between molecular properties and gene expression"""
    logger.info("STEP 4: Starting correlation analysis...")
    
    try:
        # Import CorrelationAnalyzer here to handle import errors gracefully
        from correlation_analysis import CorrelationAnalyzer
        
        # Initialize analyzer
        analyzer = CorrelationAnalyzer()
        
        # Load response data
        df = analyzer.load_response_data('dataset/de_train_split.parquet')
        logger.info(f"Loaded response data with shape: {df.shape}")
        
        # Calculate mean responses
        mean_responses = analyzer.prepare_mean_responses(by='sm_name')
        logger.info(f"Prepared mean responses with shape: {mean_responses.shape}")
        
        # Check if we have descriptor data
        if analyzer.descriptors is not None:
            logger.info(f"Loaded molecular descriptors with shape: {analyzer.descriptors.shape}")
            
            # Calculate correlations
            corr_matrix, p_vals = analyzer.calculate_correlations(mean_responses, method='pearson')
            
            # Save correlation matrix
            corr_matrix.to_csv('results/property_gene_correlations.csv')
            p_vals.to_csv('results/correlation_p_values.csv')
            logger.info("Saved correlation matrices to results directory")
            
            # Plot correlation heatmap
            try:
                fig = analyzer.plot_correlation_heatmap(n_top_genes=20)
                fig.savefig('plots/property_gene_correlations.png', dpi=300, bbox_inches='tight')
                plt.close()
            except Exception as viz_error:
                logger.error(f"Error in correlation heatmap visualization: {viz_error}")
            
            # Find top relationships
            top_relations = analyzer.find_property_gene_relationships(top_n=15)
            logger.info("\nTop Property-Gene Relationships:")
            for idx, row in top_relations.iterrows():
                logger.info(f"{row['Property']} - {row['Gene']}: {row['Correlation']:.4f} (p={row['p_value']:.4f})")
            
            # Save top relationships
            top_relations.to_csv('results/top_property_gene_relationships.csv', index=False)
            
            # Plot clustered heatmap
            try:
                fig_clustered = analyzer.plot_clustered_heatmap(n_clusters=3, n_top_genes=30)
                fig_clustered.savefig('plots/clustered_property_gene_correlations.png', dpi=300, bbox_inches='tight')
                plt.close()
            except Exception as viz_error:
                logger.error(f"Error in clustered heatmap visualization: {viz_error}")
            
            logger.info("STEP 4: Correlation analysis complete")
            return True
        else:
            logger.error("No descriptor data available. Please run step1_molecular_analysis first.")
            return False
    
    except Exception as e:
        logger.error(f"STEP 4 FAILED: Error in correlation analysis: {e}")
        logger.error(traceback.format_exc())
        return False


def step5_model_interpretation():
    """Step 5: Interpret trained models"""
    logger.info("STEP 5: Starting model interpretation...")
    
    try:
        # Import ModelInterpreter here to handle import errors gracefully
        from model_interpreter import ModelInterpreter
        
        # Check if model files exist
        if not os.path.exists('models/ensemble_model.pkl'):
            logger.error("No ensemble model found. Please run step3_train_models first.")
            return False
        
        if not os.path.exists('models/top_genes_used.pkl'):
            logger.error("No gene list found. Please run step3_train_models first.")
            return False
        
        # Initialize interpreter
        interpreter = ModelInterpreter(
            model_path='models/ensemble_model.pkl',
            top_genes_path='models/top_genes_used.pkl'
        )
        
        # Load dataset
        if os.path.exists('dataset/de_train_split.parquet'):
            interpreter.load_dataset('dataset/de_train_split.parquet')
            logger.info("Loaded dataset for model interpretation")
        else:
            logger.error("Dataset not found for model interpretation")
            return False
        
        # Generate comprehensive model summary
        results = interpreter.evaluate_model()
        summary = interpreter.summarize_model(results)
        
        # Save evaluation results
        if results:
            pd.DataFrame({
                'Metric': ['MSE', 'R²'],
                'Value': [results['overall_mse'], results['overall_r2']]
            }).to_csv('results/model_evaluation.csv', index=False)
            
            if 'gene_metrics' in results:
                results['gene_metrics'].to_csv('results/gene_performance.csv', index=False)
            if 'cell_metrics' in results:
                results['cell_metrics'].to_csv('results/cell_type_performance.csv', index=False)
            if 'drug_metrics' in results:
                results['drug_metrics'].to_csv('results/drug_performance.csv', index=False)
            
            logger.info("Saved model evaluation results to results directory")
        
        logger.info("STEP 5: Model interpretation complete")
        return True
    
    except Exception as e:
        logger.error(f"STEP 5 FAILED: Error in model interpretation: {e}")
        logger.error(traceback.format_exc())
        return False


# Helper Functions
def add_statistical_features(df):
    """Add statistical features derived from gene expression data"""
    # Identify metadata columns
    metadata_cols = ['cell_type', 'sm_name', 'sm_lincs_id', 'SMILES', 'control']
    gene_cols = [col for col in df.columns if col not in metadata_cols]
    
    # Make a copy to avoid modifying the original
    df_copy = df.copy()
    
    # Calculate statistics for each sample
    df_copy['gene_mean'] = df[gene_cols].mean(axis=1)
    df_copy['gene_std'] = df[gene_cols].std(axis=1)
    df_copy['gene_median'] = df[gene_cols].median(axis=1)
    df_copy['gene_q1'] = df[gene_cols].quantile(0.25, axis=1)
    df_copy['gene_q3'] = df[gene_cols].quantile(0.75, axis=1)
    df_copy['gene_iqr'] = df_copy['gene_q3'] - df_copy['gene_q1']
    
    return df_copy


def apply_svd_reduction(df, n_components=100):
    """Apply dimensionality reduction to gene expression data using truncated SVD"""
    # Identify metadata columns
    metadata_cols = ['cell_type', 'sm_name', 'sm_lincs_id', 'SMILES', 'control']
    stats_cols = ['gene_mean', 'gene_std', 'gene_median', 'gene_q1', 'gene_q3', 'gene_iqr']
    gene_cols = [col for col in df.columns if col not in metadata_cols and col not in stats_cols]
    
    # Apply SVD reduction
    svd = TruncatedSVD(n_components=n_components, random_state=42)
    reduced_data = svd.fit_transform(df[gene_cols])
    
    # Create column names for reduced features
    reduced_cols = [f'svd_comp_{i}' for i in range(n_components)]
    
    # Create a new dataframe with reduced features and metadata
    reduced_df = pd.DataFrame(reduced_data, columns=reduced_cols)
    
    # Add metadata and statistical columns
    for col in metadata_cols + stats_cols:
        if col in df.columns:
            reduced_df[col] = df[col].values
    
    # Calculate explained variance
    explained_variance = svd.explained_variance_ratio_.sum()
    logger.info(f"SVD explained variance with {n_components} components: {explained_variance:.4f}")
    
    # Save SVD model for later use
    with open('models/svd_model.pkl', 'wb') as f:
        pickle.dump(svd, f)
    
    return df, reduced_df


def apply_onehot_encoding(df):
    """Apply one-hot encoding to categorical features"""
    # Make a copy to avoid modifying the original
    encoded_df = df.copy()
    
    if 'cell_type' in df.columns:
        # Create encoder for cell types
        cell_encoder = OneHotEncoder(sparse_output=False)
        cell_encoded = cell_encoder.fit_transform(df[['cell_type']])
        cell_cols = [f'cell_{cat}' for cat in cell_encoder.categories_[0]]
        
        # Add to dataframe
        for i, col in enumerate(cell_cols):
            encoded_df[col] = cell_encoded[:, i]
    
    if 'sm_name' in df.columns:
        # Create encoder for drug names
        drug_encoder = OneHotEncoder(sparse_output=False)
        drug_encoded = drug_encoder.fit_transform(df[['sm_name']])
        drug_cols = [f'drug_{cat}' for cat in drug_encoder.categories_[0]]
        
        # Add to dataframe
        for i, col in enumerate(drug_cols):
            encoded_df[col] = drug_encoded[:, i]
    
    # Save encoders for later use
    with open('models/encoders.pkl', 'wb') as f:
        encoders = {}
        if 'cell_type' in df.columns:
            encoders['cell_encoder'] = cell_encoder
        if 'sm_name' in df.columns:
            encoders['drug_encoder'] = drug_encoder
        pickle.dump(encoders, f)
    
    return encoded_df


def generate_summary_report():
    """Generate a summary report of all completed steps"""
    summary = {
        "1. Molecular Analysis": os.path.exists("results/molecular_descriptors.csv"),
        "2. Data Preprocessing": os.path.exists("results/processed_data.parquet"),
        "3. Model Training": os.path.exists("models/ensemble_model.pkl"),
        "4. Correlation Analysis": os.path.exists("results/property_gene_correlations.csv"),
        "5. Model Interpretation": os.path.exists("results/model_evaluation.csv"),
    }
    
    with open("results/analysis_summary.txt", "w") as f:
        f.write("Complete Analysis Pipeline Summary\n")
        f.write("================================\n\n")
        for step, completed in summary.items():
            status = "✅ Completed" if completed else "❌ Not completed"
            f.write(f"{step}: {status}\n")
    
    logger.info("Generated analysis summary report")


def main():
    """Main function to run the analysis pipeline in steps"""
    parser = argparse.ArgumentParser(description="Run the complete analysis pipeline in steps")
    parser.add_argument('--step', type=int, default=1, help='Start from specific step (1-5)')
    args = parser.parse_args()
    
    # Set up environment
    setup_environment()
    
    # Variables to store intermediate results
    descriptors = None
    original_df = None
    encoded_df = None
    
    # Execute steps based on the starting point
    if args.step <= 1:
        logger.info("Starting from Step 1: Molecular Analysis")
        success, descriptors = step1_molecular_analysis()
        if not success:
            logger.error("Failed at Step 1. Exiting.")
            return
    
    if args.step <= 2:
        logger.info("Executing Step 2: Data Preprocessing")
        success, original_df, encoded_df = step2_data_preprocessing()
        if not success:
            logger.error("Failed at Step 2. Exiting.")
            return
    
    if args.step <= 3:
        logger.info("Executing Step 3: Model Training")
        success = step3_train_models(original_df, encoded_df)
        if not success:
            logger.error("Failed at Step 3. Exiting.")
            return
    
    if args.step <= 4:
        logger.info("Executing Step 4: Correlation Analysis")
        success = step4_correlation_analysis()
        if not success:
            logger.error("Failed at Step 4. Exiting.")
            return
    
    if args.step <= 5:
        logger.info("Executing Step 5: Model Interpretation")
        success = step5_model_interpretation()
        if not success:
            logger.error("Failed at Step 5. Exiting.")
            return
    
    # Generate final summary report
    generate_summary_report()
    
    logger.info("Complete analysis pipeline finished successfully!")


if __name__ == "__main__":
    main() 