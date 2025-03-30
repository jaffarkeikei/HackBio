#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Advanced Gene Expression Prediction Model

This script implements neural network approaches with dimensionality reduction
following the winning approaches from the NeurIPS 2023 competition:
- SVD for dimensionality reduction
- 1D CNN, LSTM, and GRU models
- Ensemble of models

Usage:
    python advanced_model.py
"""

import os
import sys
import time
import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import logging
import pickle
import traceback
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.model_selection import train_test_split, KFold
from sklearn.decomposition import TruncatedSVD
from sklearn.metrics import r2_score, mean_squared_error
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset, TensorDataset
import re

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("advanced_model.log"),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger("advanced_model")

# Set random seeds for reproducibility
np.random.seed(42)
torch.manual_seed(42)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(42)

# Check if CUDA is available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
logger.info(f"Using device: {device}")


# Neural Network Models
class CNNModel(nn.Module):
    """1D CNN model for gene expression prediction"""
    def __init__(self, input_dim, hidden_dim, output_dim, dropout=0.2):
        super(CNNModel, self).__init__()
        
        self.cnn = nn.Sequential(
            nn.Conv1d(1, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.BatchNorm1d(64),
            nn.MaxPool1d(2),
            nn.Dropout(dropout),
            
            nn.Conv1d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.BatchNorm1d(128),
            nn.MaxPool1d(2),
            nn.Dropout(dropout),
            
            nn.Conv1d(128, 256, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.BatchNorm1d(256),
            nn.AdaptiveAvgPool1d(1)
        )
        
        self.fc = nn.Sequential(
            nn.Linear(256, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, output_dim)
        )
    
    def forward(self, x):
        # Reshape for 1D CNN [batch, channels, sequence_length]
        x = x.unsqueeze(1)
        x = self.cnn(x)
        x = x.squeeze(-1)
        x = self.fc(x)
        return x


class LSTMModel(nn.Module):
    """LSTM model for gene expression prediction"""
    def __init__(self, input_dim, hidden_dim, output_dim, num_layers=2, dropout=0.2):
        super(LSTMModel, self).__init__()
        
        self.lstm = nn.LSTM(
            input_size=1,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0,
            bidirectional=True
        )
        
        self.fc = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),  # *2 because bidirectional
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, output_dim)
        )
    
    def forward(self, x):
        # Reshape for LSTM [batch, sequence_length, features]
        x = x.unsqueeze(2)
        lstm_out, _ = self.lstm(x)
        # Use the last time step output
        lstm_out = lstm_out[:, -1, :]
        x = self.fc(lstm_out)
        return x


class GRUModel(nn.Module):
    """GRU model for gene expression prediction"""
    def __init__(self, input_dim, hidden_dim, output_dim, num_layers=2, dropout=0.2):
        super(GRUModel, self).__init__()
        
        self.gru = nn.GRU(
            input_size=1,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0,
            bidirectional=True
        )
        
        self.fc = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),  # *2 because bidirectional
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, output_dim)
        )
    
    def forward(self, x):
        # Reshape for GRU [batch, sequence_length, features]
        x = x.unsqueeze(2)
        gru_out, _ = self.gru(x)
        # Use the last time step output
        gru_out = gru_out[:, -1, :]
        x = self.fc(gru_out)
        return x


class GeneExpressionDataset(Dataset):
    """Custom PyTorch Dataset for gene expression data"""
    def __init__(self, X, y):
        self.X = torch.FloatTensor(X)
        self.y = torch.FloatTensor(y)
    
    def __len__(self):
        return len(self.X)
    
    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]


def setup_environment():
    """Create necessary directories and check data availability"""
    # Create output directories if they don't exist
    os.makedirs('advanced_models', exist_ok=True)
    os.makedirs('advanced_plots', exist_ok=True)
    os.makedirs('advanced_results', exist_ok=True)
    
    # Check if dataset exists
    if not os.path.exists('dataset/de_train_split.parquet'):
        logger.error("Dataset not found at dataset/de_train_split.parquet")
        logger.info("Please make sure the dataset is available in the dataset directory")
        sys.exit(1)
    
    logger.info("Environment setup complete")


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
    df_copy['gene_skew'] = df[gene_cols].skew(axis=1)
    df_copy['gene_kurtosis'] = df[gene_cols].kurtosis(axis=1)
    
    return df_copy


def prepare_data(svd_components=100):
    """
    Prepare data for model training:
    1. Load and preprocess data
    2. Add statistical features
    3. Encode categorical features
    4. Apply SVD dimensionality reduction
    5. Split data into train and validation sets
    """
    logger.info("Loading and preprocessing data...")
    
    # Load data
    if os.path.exists('results/processed_data.parquet'):
        logger.info("Loading preprocessed data...")
        data = pd.read_parquet('results/processed_data.parquet')
        original_df = pd.read_parquet('dataset/de_train_split.parquet')
    else:
        logger.info("Loading raw data and preprocessing...")
        data = pd.read_parquet('dataset/de_train_split.parquet')
        original_df = data.copy()
    
    # Add statistical features
    logger.info("Adding statistical features...")
    data = add_statistical_features(data)
    
    # Identify metadata and gene columns
    metadata_cols = ['cell_type', 'sm_name', 'sm_lincs_id', 'SMILES', 'control']
    gene_cols = [col for col in original_df.columns if col not in metadata_cols]
    
    # Prepare target data (gene expression)
    y_data = original_df[gene_cols].values
    
    # Apply SVD for dimensionality reduction on target
    logger.info(f"Applying SVD dimensionality reduction to {len(gene_cols)} genes -> {svd_components} components...")
    svd = TruncatedSVD(n_components=svd_components, random_state=42)
    y_reduced = svd.fit_transform(y_data)
    logger.info(f"Explained variance ratio with {svd_components} components: {svd.explained_variance_ratio_.sum():.4f}")
    
    # Save SVD model for later reconstruction
    with open(f'advanced_models/svd_model_{svd_components}.pkl', 'wb') as f:
        pickle.dump(svd, f)
    
    # Extract categorical features that need encoding
    cat_features = ['cell_type', 'sm_name']
    X_cat = data[cat_features]
    
    # One-hot encode categorical features
    encoder = OneHotEncoder(handle_unknown='ignore', sparse_output=False)
    X_encoded = encoder.fit_transform(X_cat)
    
    # Get encoded feature names with XGBoost compatibility
    encoded_feature_names = []
    for i, feature in enumerate(cat_features):
        for category in encoder.categories_[i]:
            # Replace special characters
            safe_category = re.sub(r'[\[\]<>]', '_', str(category))
            encoded_feature_names.append(f"{feature}_{safe_category}")
    
    # Save encoder for future use
    with open('advanced_models/categorical_encoder.pkl', 'wb') as f:
        pickle.dump((encoder, encoded_feature_names), f)
    
    # Extract numerical features for the feature matrix
    num_features = ['gene_mean', 'gene_std', 'gene_median', 'gene_q1', 'gene_q3', 
                    'gene_iqr', 'gene_skew', 'gene_kurtosis']
    X_num = data[num_features].values
    
    # Standardize numerical features
    scaler = StandardScaler()
    X_num_scaled = scaler.fit_transform(X_num)
    
    # Save scaler for future use
    with open('advanced_models/numerical_scaler.pkl', 'wb') as f:
        pickle.dump(scaler, f)
    
    # Combine one-hot encoded categorical features with numerical features
    X_combined = np.hstack((X_encoded, X_num_scaled))
    
    logger.info(f"Final feature matrix shape: {X_combined.shape}")
    logger.info(f"Reduced target matrix shape: {y_reduced.shape}")
    
    # Split data into train and validation sets
    X_train, X_val, y_train, y_val = train_test_split(
        X_combined, y_reduced, test_size=0.2, random_state=42
    )
    
    # Save original gene column names for future reference
    with open('advanced_models/gene_columns.pkl', 'wb') as f:
        pickle.dump(gene_cols, f)
    
    # Save the prepared data
    np.save('advanced_results/X_train.npy', X_train)
    np.save('advanced_results/X_val.npy', X_val)
    np.save('advanced_results/y_train.npy', y_train)
    np.save('advanced_results/y_val.npy', y_val)
    
    return X_train, X_val, y_train, y_val, svd, gene_cols


def train_model(model, X_train, y_train, X_val, y_val, model_name, batch_size=32, 
                epochs=50, learning_rate=0.001, patience=5):
    """Train a PyTorch model with early stopping"""
    logger.info(f"Training {model_name} model...")
    
    # Create PyTorch datasets and dataloaders
    train_dataset = GeneExpressionDataset(X_train, y_train)
    val_dataset = GeneExpressionDataset(X_val, y_val)
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    
    # Move model to device
    model = model.to(device)
    
    # Set up loss function and optimizer
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    
    # Training loop with early stopping
    best_val_loss = float('inf')
    early_stop_counter = 0
    
    train_losses = []
    val_losses = []
    
    # Start timing
    start_time = time.time()
    
    for epoch in range(epochs):
        # Training phase
        model.train()
        train_loss = 0.0
        
        for X_batch, y_batch in train_loader:
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)
            
            # Zero the gradients
            optimizer.zero_grad()
            
            # Forward pass
            outputs = model(X_batch)
            loss = criterion(outputs, y_batch)
            
            # Backward pass and optimize
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
        
        avg_train_loss = train_loss / len(train_loader)
        train_losses.append(avg_train_loss)
        
        # Validation phase
        model.eval()
        val_loss = 0.0
        
        with torch.no_grad():
            for X_batch, y_batch in val_loader:
                X_batch, y_batch = X_batch.to(device), y_batch.to(device)
                
                outputs = model(X_batch)
                loss = criterion(outputs, y_batch)
                
                val_loss += loss.item()
        
        avg_val_loss = val_loss / len(val_loader)
        val_losses.append(avg_val_loss)
        
        # Print progress
        logger.info(f"Epoch {epoch+1}/{epochs} - Train Loss: {avg_train_loss:.4f}, Val Loss: {avg_val_loss:.4f}")
        
        # Check for early stopping
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            # Save best model
            torch.save(model.state_dict(), f'advanced_models/{model_name}_best.pt')
            early_stop_counter = 0
        else:
            early_stop_counter += 1
            if early_stop_counter >= patience:
                logger.info(f"Early stopping at epoch {epoch+1}")
                break
    
    # Calculate training time
    training_time = time.time() - start_time
    logger.info(f"Training completed in {training_time:.2f} seconds")
    
    # Load best model for evaluation
    model.load_state_dict(torch.load(f'advanced_models/{model_name}_best.pt'))
    
    # Save training history
    history = {
        'train_loss': train_losses,
        'val_loss': val_losses,
        'training_time': training_time
    }
    
    with open(f'advanced_results/{model_name}_history.pkl', 'wb') as f:
        pickle.dump(history, f)
    
    # Plot training and validation loss
    plt.figure(figsize=(10, 6))
    plt.plot(train_losses, label='Training Loss')
    plt.plot(val_losses, label='Validation Loss')
    plt.title(f'{model_name} - Training and Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss (MSE)')
    plt.legend()
    plt.savefig(f'advanced_plots/{model_name}_loss.png', dpi=300)
    plt.close()
    
    return model, history


def evaluate_model(model, X_val, y_val, svd, model_name):
    """Evaluate a trained model and calculate performance metrics"""
    logger.info(f"Evaluating {model_name} model...")
    
    # Create dataset and dataloader for evaluation
    val_dataset = GeneExpressionDataset(X_val, y_val)
    val_loader = DataLoader(val_dataset, batch_size=64, shuffle=False)
    
    # Move model to device and set to evaluation mode
    model = model.to(device)
    model.eval()
    
    # Make predictions
    all_predictions = []
    
    with torch.no_grad():
        for X_batch, _ in val_loader:
            X_batch = X_batch.to(device)
            outputs = model(X_batch)
            all_predictions.append(outputs.cpu().numpy())
    
    # Concatenate predictions
    y_pred_reduced = np.vstack(all_predictions)
    
    # Inverse transform predictions using SVD
    y_pred_original = svd.inverse_transform(y_pred_reduced)
    y_true_original = svd.inverse_transform(y_val)
    
    # Calculate R² and MSE per gene
    r2_scores = []
    mse_scores = []
    
    for i in range(y_pred_original.shape[1]):
        r2 = r2_score(y_true_original[:, i], y_pred_original[:, i])
        mse = mean_squared_error(y_true_original[:, i], y_pred_original[:, i])
        r2_scores.append(r2)
        mse_scores.append(mse)
    
    # Calculate average metrics
    avg_r2 = np.mean(r2_scores)
    avg_mse = np.mean(mse_scores)
    
    logger.info(f"{model_name} - Average R²: {avg_r2:.4f}, Average MSE: {avg_mse:.4f}")
    
    # Save evaluation results
    results = {
        'r2_scores': r2_scores,
        'mse_scores': mse_scores,
        'avg_r2': avg_r2,
        'avg_mse': avg_mse
    }
    
    with open(f'advanced_results/{model_name}_evaluation.pkl', 'wb') as f:
        pickle.dump(results, f)
    
    # Plot R² distribution
    plt.figure(figsize=(12, 6))
    sns.histplot(r2_scores, kde=True)
    plt.axvline(avg_r2, color='red', linestyle='--', label=f'Average R² = {avg_r2:.4f}')
    plt.title(f'{model_name} - R² Score Distribution')
    plt.xlabel('R² Score')
    plt.ylabel('Count')
    plt.legend()
    plt.savefig(f'advanced_plots/{model_name}_r2_distribution.png', dpi=300)
    plt.close()
    
    return avg_r2, avg_mse, results


def create_ensemble(models, X_val, y_val, svd, model_names):
    """Create an ensemble prediction by averaging model outputs"""
    logger.info("Creating ensemble prediction...")
    
    # Create dataset and dataloader for evaluation
    val_dataset = GeneExpressionDataset(X_val, y_val)
    val_loader = DataLoader(val_dataset, batch_size=64, shuffle=False)
    
    # Get predictions from each model
    model_predictions = []
    
    for i, model in enumerate(models):
        model = model.to(device)
        model.eval()
        
        predictions = []
        
        with torch.no_grad():
            for X_batch, _ in val_loader:
                X_batch = X_batch.to(device)
                outputs = model(X_batch)
                predictions.append(outputs.cpu().numpy())
        
        model_predictions.append(np.vstack(predictions))
        logger.info(f"Collected predictions from {model_names[i]}")
    
    # Create ensemble by averaging predictions
    ensemble_pred_reduced = np.mean(model_predictions, axis=0)
    
    # Inverse transform predictions using SVD
    ensemble_pred_original = svd.inverse_transform(ensemble_pred_reduced)
    y_true_original = svd.inverse_transform(y_val)
    
    # Calculate R² and MSE per gene
    r2_scores = []
    mse_scores = []
    
    for i in range(ensemble_pred_original.shape[1]):
        r2 = r2_score(y_true_original[:, i], ensemble_pred_original[:, i])
        mse = mean_squared_error(y_true_original[:, i], ensemble_pred_original[:, i])
        r2_scores.append(r2)
        mse_scores.append(mse)
    
    # Calculate average metrics
    avg_r2 = np.mean(r2_scores)
    avg_mse = np.mean(mse_scores)
    
    logger.info(f"Ensemble - Average R²: {avg_r2:.4f}, Average MSE: {avg_mse:.4f}")
    
    # Save evaluation results
    results = {
        'r2_scores': r2_scores,
        'mse_scores': mse_scores,
        'avg_r2': avg_r2,
        'avg_mse': avg_mse
    }
    
    with open('advanced_results/ensemble_evaluation.pkl', 'wb') as f:
        pickle.dump(results, f)
    
    # Plot R² distribution
    plt.figure(figsize=(12, 6))
    sns.histplot(r2_scores, kde=True)
    plt.axvline(avg_r2, color='red', linestyle='--', label=f'Average R² = {avg_r2:.4f}')
    plt.title('Ensemble - R² Score Distribution')
    plt.xlabel('R² Score')
    plt.ylabel('Count')
    plt.legend()
    plt.savefig('advanced_plots/ensemble_r2_distribution.png', dpi=300)
    plt.close()
    
    return avg_r2, avg_mse, results


def visualize_comparisons(model_results, gene_cols):
    """Visualize model comparison results"""
    logger.info("Generating comparison visualizations...")
    
    # Create a comparison bar chart for R² and MSE
    plt.figure(figsize=(12, 6))
    
    model_names = list(model_results.keys())
    r2_values = [model_results[name]['avg_r2'] for name in model_names]
    mse_values = [model_results[name]['avg_mse'] for name in model_names]
    
    # Plot R² values
    plt.subplot(1, 2, 1)
    bars = plt.bar(model_names, r2_values, color=['blue', 'green', 'orange', 'red'])
    
    # Add value labels
    for bar, value in zip(bars, r2_values):
        plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                 f'{value:.4f}', ha='center', va='bottom', rotation=0)
    
    plt.title('Model Comparison - R² Score')
    plt.ylabel('Average R² Score')
    plt.ylim(0, max(r2_values) + 0.1)
    
    # Plot MSE values
    plt.subplot(1, 2, 2)
    bars = plt.bar(model_names, mse_values, color=['blue', 'green', 'orange', 'red'])
    
    # Add value labels
    for bar, value in zip(bars, mse_values):
        plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 5,
                 f'{value:.1f}', ha='center', va='bottom', rotation=0)
    
    plt.title('Model Comparison - MSE')
    plt.ylabel('Average MSE')
    
    plt.tight_layout()
    plt.savefig('advanced_plots/model_comparison.png', dpi=300)
    plt.close()
    
    # Create a heatmap of the top 20 genes with highest R² scores for each model
    plt.figure(figsize=(15, 10))
    
    # Create dataframe for heatmap
    heatmap_data = []
    
    for model_name in model_names:
        r2_scores = model_results[model_name]['r2_scores']
        # Get indices of top 20 genes by R² score
        top_indices = np.argsort(r2_scores)[-20:]
        
        for idx in top_indices:
            heatmap_data.append({
                'Model': model_name,
                'Gene': gene_cols[idx],
                'R²': r2_scores[idx]
            })
    
    heatmap_df = pd.DataFrame(heatmap_data)
    heatmap_pivot = heatmap_df.pivot(index='Gene', columns='Model', values='R²')
    
    # Sort by average R² across models
    heatmap_pivot['avg'] = heatmap_pivot.mean(axis=1)
    heatmap_pivot = heatmap_pivot.sort_values(by='avg', ascending=False).drop(columns=['avg'])
    
    # Plot heatmap
    sns.heatmap(heatmap_pivot, cmap='viridis', annot=True, fmt='.2f', linewidths=0.5)
    plt.title('Top Genes by R² Score Across Models')
    plt.tight_layout()
    plt.savefig('advanced_plots/top_genes_heatmap.png', dpi=300)
    plt.close()
    
    # Create summary table
    summary = pd.DataFrame({
        'Model': model_names,
        'Average R²': r2_values,
        'Average MSE': mse_values
    })
    
    summary.to_csv('advanced_results/model_comparison_summary.csv', index=False)
    
    logger.info("Comparison visualizations complete")


def main():
    """Main function to run the advanced model pipeline"""
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Train advanced models for gene expression prediction')
    parser.add_argument('--svd_components', type=int, default=100,
                        help='Number of SVD components for dimensionality reduction')
    parser.add_argument('--batch_size', type=int, default=32,
                        help='Batch size for training')
    parser.add_argument('--epochs', type=int, default=30,
                        help='Maximum number of epochs for training')
    parser.add_argument('--patience', type=int, default=5,
                        help='Patience for early stopping')
    args = parser.parse_args()
    
    try:
        # Set up environment
        setup_environment()
        
        # Prepare data with SVD
        X_train, X_val, y_train, y_val, svd, gene_cols = prepare_data(
            svd_components=args.svd_components
        )
        
        # Define input dimensions for models
        input_dim = X_train.shape[1]
        hidden_dim = 128
        output_dim = y_train.shape[1]
        
        logger.info(f"Model dimensions - Input: {input_dim}, Hidden: {hidden_dim}, Output: {output_dim}")
        
        # Initialize models
        cnn_model = CNNModel(input_dim, hidden_dim, output_dim)
        lstm_model = LSTMModel(input_dim, hidden_dim, output_dim)
        gru_model = GRUModel(input_dim, hidden_dim, output_dim)
        
        # Train models
        cnn_model, cnn_history = train_model(
            cnn_model, X_train, y_train, X_val, y_val, 'cnn',
            batch_size=args.batch_size, epochs=args.epochs, patience=args.patience
        )
        
        lstm_model, lstm_history = train_model(
            lstm_model, X_train, y_train, X_val, y_val, 'lstm',
            batch_size=args.batch_size, epochs=args.epochs, patience=args.patience
        )
        
        gru_model, gru_history = train_model(
            gru_model, X_train, y_train, X_val, y_val, 'gru',
            batch_size=args.batch_size, epochs=args.epochs, patience=args.patience
        )
        
        # Evaluate individual models
        cnn_r2, cnn_mse, cnn_results = evaluate_model(cnn_model, X_val, y_val, svd, 'cnn')
        lstm_r2, lstm_mse, lstm_results = evaluate_model(lstm_model, X_val, y_val, svd, 'lstm')
        gru_r2, gru_mse, gru_results = evaluate_model(gru_model, X_val, y_val, svd, 'gru')
        
        # Create ensemble
        models = [cnn_model, lstm_model, gru_model]
        model_names = ['cnn', 'lstm', 'gru']
        ensemble_r2, ensemble_mse, ensemble_results = create_ensemble(
            models, X_val, y_val, svd, model_names
        )
        
        # Collect results
        model_results = {
            'cnn': {'avg_r2': cnn_r2, 'avg_mse': cnn_mse, 'r2_scores': cnn_results['r2_scores']},
            'lstm': {'avg_r2': lstm_r2, 'avg_mse': lstm_mse, 'r2_scores': lstm_results['r2_scores']},
            'gru': {'avg_r2': gru_r2, 'avg_mse': gru_mse, 'r2_scores': gru_results['r2_scores']},
            'ensemble': {'avg_r2': ensemble_r2, 'avg_mse': ensemble_mse, 'r2_scores': ensemble_results['r2_scores']}
        }
        
        # Visualize comparisons
        visualize_comparisons(model_results, gene_cols)
        
        # Create performance summary
        summary = pd.DataFrame({
            'Model': ['CNN', 'LSTM', 'GRU', 'Ensemble'],
            'Average R²': [cnn_r2, lstm_r2, gru_r2, ensemble_r2],
            'Average MSE': [cnn_mse, lstm_mse, gru_mse, ensemble_mse],
            'Training Time (s)': [
                cnn_history['training_time'],
                lstm_history['training_time'],
                gru_history['training_time'],
                'N/A'
            ]
        })
        
        summary.to_csv('advanced_results/model_performance_summary.csv', index=False)
        
        # Print final results
        logger.info("\nFinal Model Performance:")
        logger.info(f"CNN - R²: {cnn_r2:.4f}, MSE: {cnn_mse:.2f}")
        logger.info(f"LSTM - R²: {lstm_r2:.4f}, MSE: {lstm_mse:.2f}")
        logger.info(f"GRU - R²: {gru_r2:.4f}, MSE: {gru_mse:.2f}")
        logger.info(f"Ensemble - R²: {ensemble_r2:.4f}, MSE: {ensemble_mse:.2f}")
        
        # Generate text summary
        with open('advanced_results/analysis_summary.txt', 'w') as f:
            f.write("HackBio Advanced Models - Performance Summary\n")
            f.write("===========================================\n\n")
            f.write(f"SVD Components: {args.svd_components}\n")
            f.write(f"Explained Variance with SVD: {svd.explained_variance_ratio_.sum():.4f}\n\n")
            
            f.write("Model Performance:\n")
            f.write(f"CNN - R²: {cnn_r2:.4f}, MSE: {cnn_mse:.2f}\n")
            f.write(f"LSTM - R²: {lstm_r2:.4f}, MSE: {lstm_mse:.2f}\n")
            f.write(f"GRU - R²: {gru_r2:.4f}, MSE: {gru_mse:.2f}\n")
            f.write(f"Ensemble - R²: {ensemble_r2:.4f}, MSE: {ensemble_mse:.2f}\n\n")
            
            f.write("Next Steps:\n")
            f.write("1. Try different SVD component sizes\n")
            f.write("2. Experiment with model architectures\n")
            f.write("3. Implement k-fold cross-validation\n")
            f.write("4. Explore transfer learning approaches\n")
        
        logger.info("Advanced model analysis complete")
        
    except Exception as e:
        logger.error(f"Error in advanced model pipeline: {e}")
        logger.error(traceback.format_exc())
        return False
    
    return True


if __name__ == "__main__":
    main() 