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
    descriptors.to_csv('molecular_descriptors.csv', index=False)
    logger.info(f"Saved molecular descriptors for {len(descriptors)} compounds to molecular_descriptors.csv")
    
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


def train_models():
    """Train the drug response models"""
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
    if not os.path.exists('models/drug_response_model.pkl'):
        logger.error("Model file not found. Please run train_model.py first.")
        return False
    
    # Initialize interpreter
    interpreter = ModelInterpreter(
        model_path='models/drug_response_model.pkl',
        top_genes_path='models/top_variable_genes.pkl'
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


if __name__ == "__main__":
    main() 