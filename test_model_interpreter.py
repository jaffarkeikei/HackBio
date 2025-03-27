import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
import pickle
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
import warnings

# Suppress warnings
warnings.filterwarnings('ignore')

# Create a simple class mimicking our DrugResponsePredictor for testing
class TestModel:
    def __init__(self):
        self.model = None
        self.preprocessor = None
        self.X_test = None
        self.y_test = None
    
    def _preprocess_data(self, X, fit=False):
        if fit:
            # Define preprocessor
            categorical_cols = ['cell_type', 'sm_name']
            numerical_cols = []
            
            # Create preprocessing steps
            categorical_processor = OneHotEncoder(handle_unknown='ignore', sparse_output=False)
            
            # Create column transformer
            self.preprocessor = ColumnTransformer(
                transformers=[
                    ('cat', categorical_processor, categorical_cols)
                ]
            )
            
            # Fit and transform
            X_processed = self.preprocessor.fit_transform(X)
        else:
            X_processed = self.preprocessor.transform(X)
            
        return X_processed
    
    def fit(self, X, y):
        # Preprocess data
        X_processed = self._preprocess_data(X, fit=True)
        
        # Create and fit model
        self.model = RandomForestRegressor(n_estimators=10, random_state=42)
        self.model.fit(X_processed, y)
        
        # Store test data
        self.X_test = X
        self.y_test = y
        
        return self
    
    def predict(self, X):
        X_processed = self._preprocess_data(X, fit=False)
        return self.model.predict(X_processed)

# Create synthetic test data
print("Creating synthetic test data...")
np.random.seed(42)

# Generate drug names and cell types
n_samples = 100
drugs = [f'Drug_{i}' for i in range(10)]
cell_types = ['T cells', 'B cells', 'NK cells', 'Monocytes']

# Create features
X = pd.DataFrame({
    'sm_name': np.random.choice(drugs, n_samples),
    'cell_type': np.random.choice(cell_types, n_samples)
})

# Create target variables (3 genes)
n_genes = 3
gene_cols = [f'Gene_{i}' for i in range(n_genes)]
y = pd.DataFrame(
    np.random.normal(0, 1, (n_samples, n_genes)),
    columns=gene_cols
)

# Train model
print("Training test model...")
model = TestModel()
model.fit(X, y)

# Save model to file
os.makedirs('models', exist_ok=True)
with open('models/test_model.pkl', 'wb') as f:
    pickle.dump(model, f)
print("Saved test model to models/test_model.pkl")

# Test model interpreter
try:
    from model_interpreter import ModelInterpreter
    
    print("Testing model interpreter...")
    interpreter = ModelInterpreter('models/test_model.pkl')
    
    # Evaluate model
    results = interpreter.evaluate_model()
    print(f"Model evaluation complete. Overall R²: {results['overall_r2']:.4f}")
    
    # Generate visualizations
    print("Generating visualizations...")
    
    # Feature importance
    fig_imp = interpreter.feature_importance()
    fig_imp.savefig('plots/test_feature_importance.png', dpi=200, bbox_inches='tight')
    print("Saved feature importance plot to plots/test_feature_importance.png")
    
    # Prediction errors
    fig_err = interpreter.plot_prediction_errors(results)
    fig_err.savefig('plots/test_prediction_errors.png', dpi=200, bbox_inches='tight')
    print("Saved prediction errors plot to plots/test_prediction_errors.png")
    
    # Cell type performance
    fig_cell = interpreter.plot_cell_type_performance(results)
    fig_cell.savefig('plots/test_cell_performance.png', dpi=200, bbox_inches='tight')
    print("Saved cell type performance plot to plots/test_cell_performance.png")
    
    # Drug performance
    fig_drug = interpreter.plot_drug_performance(results)
    fig_drug.savefig('plots/test_drug_performance.png', dpi=200, bbox_inches='tight')
    print("Saved drug performance plot to plots/test_drug_performance.png")
    
except Exception as e:
    print(f"Error in model interpreter test: {e}")
    import traceback
    traceback.print_exc() 