# Gene Expression Prediction with Optimized CNN

## Optimization Process

### Initial Approach
- We initially attempted to use SVD-transformed data to train a simple CNN model
- The model had poor performance with negative R² scores, indicating it was performing worse than a mean-based predictor
- Early stopping was triggered after only 11 epochs with an R² score of -0.0745 and MSE of 0.5698

### Improved Architecture
- We modified the CNN architecture with better regularization and learning rate scheduling
- Added batch normalization, dropout, and L2 regularization
- Implemented learning rate scheduling with ReduceLROnPlateau
- Performance improved slightly but still had negative R² score (-0.0063) and MSE of 0.5336

### Direct Gene Expression Approach
- Changed approach to use raw gene expression data instead of SVD-transformed data
- Implemented proper data preprocessing and normalization
- Modified CNN architecture with appropriate kernel sizes for biological data
- Used a deeper architecture with three convolutional blocks and pooling
- Implemented early stopping and learning rate scheduling

## Final Model Architecture

```python
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
```

## Training Process
- Used Adam optimizer with learning rate of 0.001 and weight decay of 0.0001
- Implemented learning rate scheduling with factor of 0.5 and patience of 5
- Trained for a maximum of 150 epochs with early stopping (patience=15)
- Training stopped after 36 epochs due to early stopping

## Results
- **Final R² Score**: 0.1864
- **Final MSE**: 0.4314

## Observations
1. **Raw data outperformed SVD**: Using the raw gene expression data directly produced better results than using SVD-transformed data.
2. **Regularization was crucial**: Dropout, batch normalization, and weight decay helped prevent overfitting.
3. **Learning rate scheduling**: Automatic adjustment of learning rate improved convergence.
4. **Architecture matters**: A deeper CNN with larger kernels at early layers and smaller kernels at later layers captured patterns at different scales.

## Comparison with Traditional Methods
- The optimized CNN achieved an R² score of 0.1864, which is lower than the reported R² score of ~0.33 for Random Forest.
- This suggests that for this particular gene expression prediction task, traditional machine learning approaches may still be more effective than neural networks.

## Future Improvements
1. **Feature selection**: Identifying the most relevant genes before modeling
2. **Ensemble approach**: Combining CNN with traditional methods
3. **Attention mechanisms**: Implementing attention to focus on the most important gene interactions
4. **Transfer learning**: Using pre-trained models on larger gene expression datasets
5. **Graph neural networks**: Modeling gene-gene interactions as a graph 