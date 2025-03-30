import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import TruncatedSVD
from sklearn.metrics import r2_score, mean_squared_error
import matplotlib.pyplot as plt
import seaborn as sns
import os
import logging
import pickle
from datetime import datetime
from sklearn.model_selection import train_test_split

# Set up logging
logging.basicConfig(
    filename='logs/optimized_cnn.log',
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

class SimpleCNN(nn.Module):
    def __init__(self, input_size):
        super(SimpleCNN, self).__init__()
        
        # First convolutional block
        self.conv1 = nn.Conv1d(1, 32, kernel_size=7, padding=3)
        self.bn1 = nn.BatchNorm1d(32)
        self.pool1 = nn.MaxPool1d(2, stride=2)
        self.dropout1 = nn.Dropout(0.2)
        
        # Second convolutional block
        self.conv2 = nn.Conv1d(32, 64, kernel_size=5, padding=2)
        self.bn2 = nn.BatchNorm1d(64)
        self.pool2 = nn.MaxPool1d(2, stride=2)
        self.dropout2 = nn.Dropout(0.2)
        
        # Third convolutional block
        self.conv3 = nn.Conv1d(64, 128, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm1d(128)
        self.pool3 = nn.MaxPool1d(2, stride=2)
        self.dropout3 = nn.Dropout(0.2)
        
        # Calculate size after convolutions and pooling
        conv_output_size = input_size // 8  # After three max pooling layers
        self.fc_input_size = 128 * conv_output_size
        
        # Fully connected layers
        self.fc1 = nn.Linear(self.fc_input_size, 256)
        self.fc_bn1 = nn.BatchNorm1d(256)
        self.fc_dropout1 = nn.Dropout(0.3)
        
        self.fc2 = nn.Linear(256, 128)
        self.fc_bn2 = nn.BatchNorm1d(128)
        self.fc_dropout2 = nn.Dropout(0.3)
        
        self.fc3 = nn.Linear(128, 1)
        
        self.relu = nn.ReLU()
    
    def forward(self, x):
        # Add channel dimension
        x = x.unsqueeze(1)
        
        # First block
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.pool1(x)
        x = self.dropout1(x)
        
        # Second block
        x = self.conv2(x)
        x = self.bn2(x)
        x = self.relu(x)
        x = self.pool2(x)
        x = self.dropout2(x)
        
        # Third block
        x = self.conv3(x)
        x = self.bn3(x)
        x = self.relu(x)
        x = self.pool3(x)
        x = self.dropout3(x)
        
        # Flatten
        x = x.view(x.size(0), -1)
        
        # Fully connected layers
        x = self.fc1(x)
        x = self.fc_bn1(x)
        x = self.relu(x)
        x = self.fc_dropout1(x)
        
        x = self.fc2(x)
        x = self.fc_bn2(x)
        x = self.relu(x)
        x = self.fc_dropout2(x)
        
        x = self.fc3(x)
        
        return x.squeeze()

class GeneExpressionDataset(Dataset):
    def __init__(self, X, y):
        self.X = torch.FloatTensor(X)
        self.y = torch.FloatTensor(y)
    
    def __len__(self):
        return len(self.X)
    
    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]

def preprocess_data():
    try:
        # Load data from parquet file
        train_data = pd.read_parquet('data/raw/dataset/de_train_split.parquet')
        logging.info(f"Initial data shape: {train_data.shape}")
        
        # Separate features and target
        target_gene = 'ZZEF1'
        metadata_cols = ['cell_type', 'sm_name', 'sm_lincs_id', 'SMILES', 'control']
        features = train_data.drop(metadata_cols + [target_gene], axis=1)
        logging.info(f"Features shape after dropping metadata: {features.shape}")
        
        # Convert target to numeric
        y = train_data[target_gene].values
        
        # Convert features to numpy array
        X = features.values
        
        # Normalize features
        feature_means = np.mean(X, axis=0)
        feature_stds = np.std(X, axis=0)
        X = (X - feature_means) / (feature_stds + 1e-8)  # Add small epsilon to avoid division by zero
        
        # Split the data
        X_train, X_val, y_train, y_val = train_test_split(
            X, y, test_size=0.2, random_state=42
        )
        
        logging.info(f"Training data shape: {X_train.shape}")
        logging.info(f"Validation data shape: {X_val.shape}")
        logging.info(f"Number of features: {X_train.shape[1]}")
        logging.info(f"Number of samples: {len(X)}")
        
        return X_train, X_val, y_train, y_val
        
    except Exception as e:
        logging.error(f"Error in preprocessing: {str(e)}")
        raise

def train_model(model, train_loader, val_loader, criterion, optimizer, 
                num_epochs=100, patience=15, device='cuda' if torch.cuda.is_available() else 'cpu'):
    model = model.to(device)
    best_val_loss = float('inf')
    patience_counter = 0
    train_losses = []
    val_losses = []
    
    # Learning rate scheduler
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=5, verbose=True
    )
    
    for epoch in range(num_epochs):
        # Training phase
        model.train()
        total_train_loss = 0
        for batch_X, batch_y in train_loader:
            batch_X, batch_y = batch_X.to(device), batch_y.to(device)
            
            optimizer.zero_grad()
            outputs = model(batch_X)
            loss = criterion(outputs, batch_y)
            
            # L2 regularization
            l2_lambda = 0.001
            l2_reg = torch.tensor(0.).to(device)
            for param in model.parameters():
                l2_reg += torch.norm(param)
            loss += l2_lambda * l2_reg
            
            loss.backward()
            
            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            
            optimizer.step()
            total_train_loss += loss.item()
        
        avg_train_loss = total_train_loss / len(train_loader)
        train_losses.append(avg_train_loss)
        
        # Validation phase
        model.eval()
        total_val_loss = 0
        with torch.no_grad():
            for batch_X, batch_y in val_loader:
                batch_X, batch_y = batch_X.to(device), batch_y.to(device)
                outputs = model(batch_X)
                loss = criterion(outputs, batch_y)
                total_val_loss += loss.item()
        
        avg_val_loss = total_val_loss / len(val_loader)
        val_losses.append(avg_val_loss)
        
        # Learning rate scheduling
        scheduler.step(avg_val_loss)
        
        logging.info(f'Epoch [{epoch+1}/{num_epochs}] - '
                    f'Train Loss: {avg_train_loss:.4f}, '
                    f'Val Loss: {avg_val_loss:.4f}, '
                    f'LR: {optimizer.param_groups[0]["lr"]:.6f}')
        
        # Early stopping
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            patience_counter = 0
            # Save best model
            torch.save(model.state_dict(), 'results/optimized/optimized_models/best_model.pt')
        else:
            patience_counter += 1
            if patience_counter >= patience:
                logging.info(f'Early stopping triggered after {epoch+1} epochs')
                break
    
    return train_losses, val_losses

def evaluate_model(model, val_loader, device='cuda' if torch.cuda.is_available() else 'cpu'):
    model.eval()
    all_preds = []
    all_targets = []
    
    with torch.no_grad():
        for batch_X, batch_y in val_loader:
            batch_X, batch_y = batch_X.to(device), batch_y.to(device)
            outputs = model(batch_X)
            all_preds.extend(outputs.cpu().numpy())
            all_targets.extend(batch_y.cpu().numpy())
    
    all_preds = np.array(all_preds)
    all_targets = np.array(all_targets)
    
    r2 = r2_score(all_targets, all_preds)
    mse = mean_squared_error(all_targets, all_preds)
    
    return r2, mse, all_preds, all_targets

def plot_results(train_losses, val_losses, r2, mse, predictions, targets):
    # Create directories if they don't exist
    os.makedirs('results/optimized/optimized_plots', exist_ok=True)
    
    # Plot training curves
    plt.figure(figsize=(10, 6))
    plt.plot(train_losses, label='Training Loss')
    plt.plot(val_losses, label='Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training and Validation Loss')
    plt.legend()
    plt.savefig('results/optimized/optimized_plots/training_curves.png')
    plt.close()
    
    # Plot predictions vs actual
    plt.figure(figsize=(10, 6))
    plt.scatter(targets, predictions, alpha=0.5)
    plt.plot([min(targets), max(targets)], [min(targets), max(targets)], 'r--')
    plt.xlabel('Actual Values')
    plt.ylabel('Predicted Values')
    plt.title(f'Predictions vs Actual (R² = {r2:.4f}, MSE = {mse:.4f})')
    plt.savefig('results/optimized/optimized_plots/predictions_vs_actual.png')
    plt.close()

def main():
    try:
        # Create necessary directories
        os.makedirs('results/optimized/optimized_models', exist_ok=True)
        os.makedirs('results/optimized/optimized_results', exist_ok=True)
        
        # Set random seeds for reproducibility
        torch.manual_seed(42)
        np.random.seed(42)
        
        # Preprocess data
        X_train, X_val, y_train, y_val = preprocess_data()
        
        # Create data loaders
        train_dataset = GeneExpressionDataset(X_train, y_train)
        val_dataset = GeneExpressionDataset(X_val, y_val)
        
        train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)
        
        # Initialize model and training components
        input_size = X_train.shape[1]  # Number of features
        model = SimpleCNN(input_size)
        criterion = nn.MSELoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=0.001, weight_decay=0.0001)
        
        # Train model
        train_losses, val_losses = train_model(
            model, train_loader, val_loader, criterion, optimizer,
            num_epochs=150, patience=15
        )
        
        # Load best model for evaluation
        model.load_state_dict(torch.load('results/optimized/optimized_models/best_model.pt'))
        
        # Evaluate model
        r2, mse, predictions, targets = evaluate_model(model, val_loader)
        
        logging.info(f'Final R² Score: {r2:.4f}')
        logging.info(f'Final MSE: {mse:.4f}')
        
        # Plot and save results
        plot_results(train_losses, val_losses, r2, mse, predictions, targets)
        
        # Save performance metrics
        with open('results/optimized/optimized_results/performance_metrics.txt', 'w') as f:
            f.write(f'R² Score: {r2:.4f}\n')
            f.write(f'MSE: {mse:.4f}\n')
        
    except Exception as e:
        logging.error(f"Error in main execution: {str(e)}")
        raise

if __name__ == "__main__":
    main() 