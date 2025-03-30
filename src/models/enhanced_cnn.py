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

# Set up logging
logging.basicConfig(
    filename='enhanced_cnn.log',
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

class ResidualBlock(nn.Module):
    def __init__(self, channels):
        super(ResidualBlock, self).__init__()
        self.conv1 = nn.Conv1d(channels, channels, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm1d(channels)
        self.conv2 = nn.Conv1d(channels, channels, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm1d(channels)
        self.relu = nn.ReLU()
        
    def forward(self, x):
        residual = x
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += residual
        out = self.relu(out)
        return out

class EnhancedCNN(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(EnhancedCNN, self).__init__()
        
        # Initial convolution with more filters
        self.conv1 = nn.Conv1d(1, 64, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm1d(64)
        
        # Residual blocks
        self.res1 = ResidualBlock(64)
        self.res2 = ResidualBlock(64)
        
        # Additional convolutions with increasing filters
        self.conv2 = nn.Conv1d(64, 128, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm1d(128)
        self.conv3 = nn.Conv1d(128, 256, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm1d(256)
        
        # Global average pooling
        self.gap = nn.AdaptiveAvgPool1d(1)
        
        # Fully connected layers with dropout
        self.fc1 = nn.Linear(256, 512)
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, output_dim)
        
        self.dropout = nn.Dropout(0.3)
        self.relu = nn.ReLU()

    def forward(self, x):
        # Reshape input for 1D convolution
        x = x.unsqueeze(1)  # Add channel dimension
        
        # Initial convolution block
        x = self.relu(self.bn1(self.conv1(x)))
        
        # Residual blocks
        x = self.res1(x)
        x = self.res2(x)
        
        # Additional convolution blocks
        x = self.relu(self.bn2(self.conv2(x)))
        x = self.relu(self.bn3(self.conv3(x)))
        
        # Global average pooling
        x = self.gap(x)
        x = x.view(x.size(0), -1)
        
        # Fully connected layers
        x = self.dropout(self.relu(self.fc1(x)))
        x = self.dropout(self.relu(self.fc2(x)))
        x = self.fc3(x)
        
        return x

class GeneExpressionDataset(Dataset):
    def __init__(self, X, y):
        self.X = torch.FloatTensor(X)
        self.y = torch.FloatTensor(y)
    
    def __len__(self):
        return len(self.X)
    
    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]

def train_model(model, train_loader, val_loader, criterion, optimizer, device, num_epochs=50, patience=10):
    train_losses = []
    val_losses = []
    best_val_loss = float('inf')
    patience_counter = 0
    best_model = None
    
    for epoch in range(num_epochs):
        # Training phase
        model.train()
        train_loss = 0
        for X_batch, y_batch in train_loader:
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)
            
            optimizer.zero_grad()
            outputs = model(X_batch)
            loss = criterion(outputs, y_batch)
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
        
        train_loss /= len(train_loader)
        train_losses.append(train_loss)
        
        # Validation phase
        model.eval()
        val_loss = 0
        with torch.no_grad():
            for X_batch, y_batch in val_loader:
                X_batch, y_batch = X_batch.to(device), y_batch.to(device)
                outputs = model(X_batch)
                loss = criterion(outputs, y_batch)
                val_loss += loss.item()
        
        val_loss /= len(val_loader)
        val_losses.append(val_loss)
        
        # Early stopping check
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_model = model.state_dict().copy()
            patience_counter = 0
        else:
            patience_counter += 1
        
        logging.info(f'Epoch {epoch+1}/{num_epochs}, Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}')
        
        if patience_counter >= patience:
            logging.info(f'Early stopping triggered after {epoch+1} epochs')
            break
    
    # Load best model
    model.load_state_dict(best_model)
    return model, train_losses, val_losses

def main():
    # Create directories for results
    os.makedirs('enhanced_results', exist_ok=True)
    os.makedirs('enhanced_models', exist_ok=True)
    os.makedirs('enhanced_plots', exist_ok=True)
    
    # Load preprocessed data
    try:
        X_train = np.load('advanced_results/X_train.npy')
        X_val = np.load('advanced_results/X_val.npy')
        y_train = np.load('advanced_results/y_train.npy')
        y_val = np.load('advanced_results/y_val.npy')
        
        logging.info(f'Loaded preprocessed data: X_train shape {X_train.shape}, y_train shape {y_train.shape}')
    except Exception as e:
        logging.error(f'Error loading data: {str(e)}')
        return
    
    # Model parameters
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    input_dim = X_train.shape[1]
    output_dim = y_train.shape[1]
    batch_size = 64
    learning_rate = 0.001
    num_epochs = 50
    patience = 10
    
    # Create data loaders
    train_dataset = GeneExpressionDataset(X_train, y_train)
    val_dataset = GeneExpressionDataset(X_val, y_val)
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size)
    
    # Initialize model
    model = EnhancedCNN(input_dim, output_dim).to(device)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    
    # Train model
    logging.info('Starting model training...')
    model, train_losses, val_losses = train_model(
        model, train_loader, val_loader, criterion, optimizer, device,
        num_epochs=num_epochs, patience=patience
    )
    
    # Evaluate model
    model.eval()
    with torch.no_grad():
        val_predictions = []
        val_targets = []
        
        for X_batch, y_batch in val_loader:
            X_batch = X_batch.to(device)
            outputs = model(X_batch)
            val_predictions.extend(outputs.cpu().numpy())
            val_targets.extend(y_batch.numpy())
    
    val_predictions = np.array(val_predictions)
    val_targets = np.array(val_targets)
    
    # Calculate metrics
    r2_scores = [r2_score(val_targets[:, i], val_predictions[:, i]) for i in range(output_dim)]
    mse_scores = [mean_squared_error(val_targets[:, i], val_predictions[:, i]) for i in range(output_dim)]
    
    avg_r2 = np.mean(r2_scores)
    avg_mse = np.mean(mse_scores)
    
    logging.info(f'Average R² Score: {avg_r2:.4f}')
    logging.info(f'Average MSE: {avg_mse:.4f}')
    
    # Save results
    results = {
        'r2_scores': r2_scores,
        'mse_scores': mse_scores,
        'train_losses': train_losses,
        'val_losses': val_losses,
        'predictions': val_predictions,
        'targets': val_targets
    }
    
    with open('enhanced_results/model_performance.pkl', 'wb') as f:
        pickle.dump(results, f)
    
    # Save model
    torch.save(model.state_dict(), 'enhanced_models/enhanced_cnn.pt')
    
    # Plot training curves
    plt.figure(figsize=(10, 6))
    plt.plot(train_losses, label='Training Loss')
    plt.plot(val_losses, label='Validation Loss')
    plt.title('Enhanced CNN Training Progress')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.savefig('enhanced_plots/training_curves.png')
    plt.close()
    
    # Plot R² distribution
    plt.figure(figsize=(10, 6))
    sns.histplot(r2_scores, bins=20)
    plt.title('Distribution of R² Scores Across Genes')
    plt.xlabel('R² Score')
    plt.ylabel('Count')
    plt.savefig('enhanced_plots/r2_distribution.png')
    plt.close()
    
    logging.info('Enhanced CNN training and evaluation completed')

if __name__ == '__main__':
    main() 