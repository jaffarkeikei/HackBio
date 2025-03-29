#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Fixed Analysis Pipeline Script

This script fixes issues in the complete_analysis.py script,
particularly addressing the multi-target regression problem in the model training step.

Usage:
    python fixed_analysis.py [--step STEP_NUMBER]

Options:
    --step STEP_NUMBER    Start from specific step (1: molecular analysis, 2: data preprocessing, 
                          3: model training, 4: correlation analysis, 5: model interpretation)
"""

import os
import sys
import argparse
import time
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import TruncatedSVD
from sklearn.preprocessing import OneHotEncoder
from sklearn.ensemble import RandomForestRegressor
import logging
import pickle
import traceback
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_squared_error
from sklearn.multioutput import MultiOutputRegressor

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("fixed_analysis.log"),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger("fixed_analysis")


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


def step3_train_models_fixed(original_df=None, encoded_df=None):
    """Step 3: Train ensemble models with properly configured multi-output support"""
    logger.info("STEP 3: Starting ensemble model training (fixed for multi-output)...")
    
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
        
        # Properly prepare features - drop any non-numeric columns or encode them
        # First, identify all the columns we need to check
        X_columns = [col for col in encoded_df.columns if col not in original_gene_cols]
        
        # Create a clean features dataframe
        X_clean = encoded_df[X_columns].copy()
        
        # Check for non-numeric columns
        non_numeric_cols = []
        for col in X_clean.columns:
            if X_clean[col].dtype == 'object' or pd.api.types.is_string_dtype(X_clean[col]):
                non_numeric_cols.append(col)
        
        logger.info(f"Found {len(non_numeric_cols)} non-numeric columns that need encoding: {non_numeric_cols}")
        
        # One-hot encode any remaining categorical columns
        if non_numeric_cols:
            # Create a one-hot encoder for these columns
            encoder = OneHotEncoder(sparse_output=False)
            encoded_cats = encoder.fit_transform(X_clean[non_numeric_cols])
            
            # Create column names for the encoded categories
            encoded_col_names = []
            for i, col in enumerate(non_numeric_cols):
                for cat in encoder.categories_[i]:
                    encoded_col_names.append(f"{col}_{cat}")
            
            # Create a dataframe with the encoded categories
            encoded_df_cats = pd.DataFrame(encoded_cats, columns=encoded_col_names, index=X_clean.index)
            
            # Drop the original categorical columns and join the encoded ones
            X_clean = X_clean.drop(columns=non_numeric_cols)
            X_clean = pd.concat([X_clean, encoded_df_cats], axis=1)
            
            logger.info(f"One-hot encoded {len(non_numeric_cols)} categorical columns into {len(encoded_col_names)} features")
        
        # Final check to make sure all columns are numeric
        X_clean = X_clean.select_dtypes(include=['number'])
        logger.info(f"Final feature matrix shape: {X_clean.shape}")
        
        # Create base models properly wrapped for multi-output regression
        # Fix: Use MultiOutputRegressor to handle multiple target columns
        rf_model = MultiOutputRegressor(
            RandomForestRegressor(n_estimators=100, max_depth=20, random_state=42, n_jobs=-1)
        )
        
        xgb_model = MultiOutputRegressor(
            xgb.XGBRegressor(n_estimators=100, max_depth=5, learning_rate=0.1, random_state=42)
        )
        
        # Split data for training
        X = X_clean
        y = original_df[top_genes]
        
        logger.info(f"Feature matrix shape: {X.shape}, Target matrix shape: {y.shape}")
        
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        
        # Train RF model first and evaluate
        logger.info("Training Random Forest model...")
        start_time = time.time()
        rf_model.fit(X_train, y_train)
        rf_training_time = time.time() - start_time
        logger.info(f"RF model training completed in {rf_training_time:.2f} seconds")
        
        # Train XGBoost model and evaluate
        logger.info("Training XGBoost model...")
        start_time = time.time()
        xgb_model.fit(X_train, y_train)
        xgb_training_time = time.time() - start_time
        logger.info(f"XGBoost model training completed in {xgb_training_time:.2f} seconds")
        
        # Save models
        with open('models/rf_model.pkl', 'wb') as f:
            pickle.dump(rf_model, f)
        
        with open('models/xgb_model.pkl', 'wb') as f:
            pickle.dump(xgb_model, f)
        
        # Save list of top genes used
        with open('models/top_genes_used.pkl', 'wb') as f:
            pickle.dump(top_genes, f)
        
        # Evaluate models
        rf_pred = rf_model.predict(X_test)
        xgb_pred = xgb_model.predict(X_test)
        
        # Evaluate predictions and create ensemble by averaging
        ensemble_pred = (rf_pred + xgb_pred) / 2
        
        # Calculate metrics for each model
        rf_r2_scores = []
        xgb_r2_scores = []
        ensemble_r2_scores = []
        
        rf_mse_scores = []
        xgb_mse_scores = []
        ensemble_mse_scores = []
        
        for i, gene in enumerate(top_genes):
            # Random Forest metrics
            rf_r2 = r2_score(y_test.iloc[:, i], rf_pred[:, i])
            rf_mse = mean_squared_error(y_test.iloc[:, i], rf_pred[:, i])
            rf_r2_scores.append(rf_r2)
            rf_mse_scores.append(rf_mse)
            
            # XGBoost metrics
            xgb_r2 = r2_score(y_test.iloc[:, i], xgb_pred[:, i])
            xgb_mse = mean_squared_error(y_test.iloc[:, i], xgb_pred[:, i])
            xgb_r2_scores.append(xgb_r2)
            xgb_mse_scores.append(xgb_mse)
            
            # Ensemble metrics
            ens_r2 = r2_score(y_test.iloc[:, i], ensemble_pred[:, i])
            ens_mse = mean_squared_error(y_test.iloc[:, i], ensemble_pred[:, i])
            ensemble_r2_scores.append(ens_r2)
            ensemble_mse_scores.append(ens_mse)
        
        # Calculate averages
        avg_rf_r2 = np.mean(rf_r2_scores)
        avg_rf_mse = np.mean(rf_mse_scores)
        
        avg_xgb_r2 = np.mean(xgb_r2_scores)
        avg_xgb_mse = np.mean(xgb_mse_scores)
        
        avg_ensemble_r2 = np.mean(ensemble_r2_scores)
        avg_ensemble_mse = np.mean(ensemble_mse_scores)
        
        # Create detailed metrics dataframe
        gene_metrics = pd.DataFrame({
            'Gene': top_genes,
            'RF_R2': rf_r2_scores,
            'RF_MSE': rf_mse_scores,
            'XGB_R2': xgb_r2_scores,
            'XGB_MSE': xgb_mse_scores,
            'Ensemble_R2': ensemble_r2_scores,
            'Ensemble_MSE': ensemble_mse_scores
        })
        
        gene_metrics.to_csv('results/gene_performance_metrics.csv', index=False)
        
        # Save summary metrics
        summary_metrics = pd.DataFrame({
            'Model': ['Random Forest', 'XGBoost', 'Ensemble'],
            'Avg_R2': [avg_rf_r2, avg_xgb_r2, avg_ensemble_r2],
            'Avg_MSE': [avg_rf_mse, avg_xgb_mse, avg_ensemble_mse],
            'Training_Time(s)': [rf_training_time, xgb_training_time, 'N/A']
        })
        
        summary_metrics.to_csv('results/ensemble_model_evaluation.csv', index=False)
        
        logger.info(f"Model evaluation:")
        logger.info(f"RF: R²={avg_rf_r2:.4f}, MSE={avg_rf_mse:.4f}")
        logger.info(f"XGB: R²={avg_xgb_r2:.4f}, MSE={avg_xgb_mse:.4f}")
        logger.info(f"Ensemble: R²={avg_ensemble_r2:.4f}, MSE={avg_ensemble_mse:.4f}")
        
        # Optional: Plot predictions vs actual for a few genes
        try:
            plt.figure(figsize=(12, 8))
            for i, gene in enumerate(top_genes[:4]):  # Plot first 4 genes
                plt.subplot(2, 2, i+1)
                
                # Plot Random Forest predictions
                plt.scatter(y_test.iloc[:, i], rf_pred[:, i], alpha=0.3, label='RF', color='blue')
                
                # Plot XGBoost predictions
                plt.scatter(y_test.iloc[:, i], xgb_pred[:, i], alpha=0.3, label='XGB', color='green')
                
                # Plot Ensemble predictions
                plt.scatter(y_test.iloc[:, i], ensemble_pred[:, i], alpha=0.5, label='Ensemble', color='red')
                
                # Plot the ideal y=x line
                plt.plot([y_test.iloc[:, i].min(), y_test.iloc[:, i].max()], 
                         [y_test.iloc[:, i].min(), y_test.iloc[:, i].max()], 'k--')
                
                plt.title(f'Gene: {gene}\nEns R² = {ensemble_r2_scores[i]:.4f}')
                plt.xlabel('Actual Expression')
                plt.ylabel('Predicted Expression')
                plt.legend()
            
            plt.tight_layout()
            plt.savefig('plots/gene_predictions.png', dpi=300, bbox_inches='tight')
            plt.close()
            
            # Also create a bar chart comparing model performance
            plt.figure(figsize=(10, 6))
            bar_width = 0.25
            index = np.arange(len(top_genes[:10]))  # Show only first 10 genes for clarity
            
            plt.bar(index, [ensemble_r2_scores[i] for i in range(10)], bar_width, 
                   label='Ensemble', color='red', alpha=0.7)
            plt.bar(index + bar_width, [rf_r2_scores[i] for i in range(10)], bar_width,
                   label='Random Forest', color='blue', alpha=0.7)
            plt.bar(index + 2*bar_width, [xgb_r2_scores[i] for i in range(10)], bar_width,
                   label='XGBoost', color='green', alpha=0.7)
            
            plt.xlabel('Gene')
            plt.ylabel('R² Score')
            plt.title('Model Performance Comparison (R²)')
            plt.xticks(index + bar_width, [top_genes[i][:8] + '...' for i in range(10)], rotation=45)
            plt.legend()
            plt.tight_layout()
            
            plt.savefig('plots/model_comparison.png', dpi=300, bbox_inches='tight')
            plt.close()
            
        except Exception as viz_error:
            logger.error(f"Error in prediction visualization: {viz_error}")
            logger.error(traceback.format_exc())
        
        logger.info("STEP 3: Model training complete")
        return True
    
    except Exception as e:
        logger.error(f"STEP 3 FAILED: Error in model training: {e}")
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


def main():
    """Main function to run the fixed model training"""
    parser = argparse.ArgumentParser(description="Run the fixed model training pipeline")
    args = parser.parse_args()
    
    # Set up environment
    setup_environment()
    
    # Execute fixed model training
    success = step3_train_models_fixed()
    
    if success:
        logger.info("Fixed model training completed successfully!")
    else:
        logger.error("Fixed model training failed.")


if __name__ == "__main__":
    main() 