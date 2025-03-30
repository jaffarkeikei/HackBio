import pandas as pd
import numpy as np
import os
import pickle
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, classification_report, mean_squared_error, r2_score
import xgboost as xgb
import argparse
from datetime import datetime
import warnings

# Suppress warnings
warnings.filterwarnings("ignore", category=UserWarning)

class CellTypeClassifier:
    """
    A classifier for predicting cell types from gene expression data.
    
    This class trains a model to identify cell types based on gene expression signatures.
    """
    
    def __init__(self, n_top_genes=1000, model_type='rf'):
        """
        Initialize the cell type classifier.
        
        Parameters:
        -----------
        n_top_genes : int
            Number of top variable genes to use for classification
        model_type : str
            Model type ('rf' for Random Forest, 'xgb' for XGBoost)
        """
        self.n_top_genes = n_top_genes
        self.model_type = model_type
        self.encoder = LabelEncoder()
        self.model = None
        self.top_genes = None
        self.feature_importance = None
    
    def preprocess_data(self, df):
        """
        Preprocess the dataset for cell type classification.
        
        Parameters:
        -----------
        df : pandas DataFrame
            Input dataset with gene expression data
            
        Returns:
        --------
        tuple
            X (features) and y (cell type labels)
        """
        # Verify cell_type column exists
        if 'cell_type' not in df.columns:
            raise ValueError("Dataset must contain 'cell_type' column")
        
        # Identify metadata columns
        metadata_cols = ['cell_type', 'sm_name', 'sm_lincs_id', 'SMILES', 'control']
        gene_cols = [col for col in df.columns if col not in metadata_cols]
        
        print(f"Dataset contains {len(gene_cols)} gene expression features")
        
        # Select top variable genes
        if len(gene_cols) > self.n_top_genes:
            gene_variance = df[gene_cols].var().sort_values(ascending=False)
            self.top_genes = gene_variance.index[:self.n_top_genes].tolist()
            print(f"Selected top {len(self.top_genes)} variable genes")
        else:
            self.top_genes = gene_cols
            print(f"Using all {len(self.top_genes)} genes (less than requested {self.n_top_genes})")
        
        # Encode cell types
        y = self.encoder.fit_transform(df['cell_type'])
        
        # Extract features
        X = df[self.top_genes].values
        
        return X, y
    
    def train(self, X, y, cv=5):
        """
        Train the cell type classifier.
        
        Parameters:
        -----------
        X : numpy array
            Feature matrix
        y : numpy array
            Target labels
        cv : int
            Number of cross-validation folds
            
        Returns:
        --------
        object
            Trained model
        """
        print(f"Training cell type classifier with {X.shape[1]} features...")
        
        # Define model and parameter grid
        if self.model_type == 'rf':
            model = RandomForestClassifier(random_state=42)
            param_grid = {
                'n_estimators': [100, 200],
                'max_depth': [None, 20, 30]
            }
        elif self.model_type == 'xgb':
            model = xgb.XGBClassifier(random_state=42)
            param_grid = {
                'n_estimators': [100, 200],
                'max_depth': [3, 5, 7],
                'learning_rate': [0.1, 0.05]
            }
        else:
            raise ValueError(f"Unknown model type: {self.model_type}")
        
        # Perform grid search
        grid_search = GridSearchCV(model, param_grid, cv=cv, scoring='accuracy', n_jobs=-1)
        grid_search.fit(X, y)
        
        self.model = grid_search
        print(f"Best parameters: {grid_search.best_params_}")
        print(f"CV accuracy: {grid_search.best_score_:.4f}")
        
        # Calculate feature importance
        if self.model_type == 'rf':
            self.feature_importance = grid_search.best_estimator_.feature_importances_
        elif self.model_type == 'xgb':
            self.feature_importance = grid_search.best_estimator_.feature_importances_
        
        return self.model
    
    def evaluate(self, X_test, y_test):
        """
        Evaluate the classifier on test data.
        
        Parameters:
        -----------
        X_test : numpy array
            Test features
        y_test : numpy array
            Test labels
            
        Returns:
        --------
        dict
            Evaluation metrics
        """
        if self.model is None:
            raise ValueError("Model not trained. Call train() first.")
        
        # Make predictions
        y_pred = self.model.predict(X_test)
        
        # Calculate metrics
        accuracy = accuracy_score(y_test, y_pred)
        report = classification_report(y_test, y_pred, target_names=self.encoder.classes_, output_dict=True)
        
        print(f"Test accuracy: {accuracy:.4f}")
        
        # Calculate confusion matrix
        cm = pd.crosstab(
            pd.Series(y_test, name='Actual'), 
            pd.Series(y_pred, name='Predicted'),
            rownames=['Actual'], 
            colnames=['Predicted']
        )
        cm.index = [self.encoder.classes_[i] for i in range(len(self.encoder.classes_))]
        cm.columns = [self.encoder.classes_[i] for i in range(len(self.encoder.classes_))]
        
        results = {
            'accuracy': accuracy,
            'report': report,
            'confusion_matrix': cm,
            'y_pred': y_pred,
            'y_test': y_test
        }
        
        return results
    
    def plot_confusion_matrix(self, results):
        """
        Plot confusion matrix from evaluation results.
        
        Parameters:
        -----------
        results : dict
            Results dictionary from evaluate()
            
        Returns:
        --------
        matplotlib.figure.Figure
            The confusion matrix plot
        """
        plt.figure(figsize=(8, 6))
        cm = results['confusion_matrix']
        
        # Normalize confusion matrix
        cm_norm = cm.div(cm.sum(axis=1), axis=0)
        
        # Plot heatmap
        sns.heatmap(cm_norm, annot=cm, fmt='g', cmap='Blues', cbar=False)
        plt.title('Cell Type Classification Confusion Matrix')
        plt.tight_layout()
        
        # Save figure
        os.makedirs('plots', exist_ok=True)
        plt.savefig('plots/cell_type_confusion_matrix.png', dpi=300)
        
        return plt.gcf()
    
    def plot_feature_importance(self, top_n=20):
        """
        Plot feature importance.
        
        Parameters:
        -----------
        top_n : int
            Number of top features to plot
            
        Returns:
        --------
        matplotlib.figure.Figure
            The feature importance plot
        """
        if self.feature_importance is None:
            raise ValueError("Feature importance not calculated. Train model first.")
        
        # Create importance DataFrame
        importance_df = pd.DataFrame({
            'Feature': self.top_genes,
            'Importance': self.feature_importance
        })
        
        # Sort by importance
        importance_df = importance_df.sort_values('Importance', ascending=False).head(top_n)
        
        # Plot
        plt.figure(figsize=(10, 8))
        sns.barplot(x='Importance', y='Feature', data=importance_df)
        plt.title(f'Top {top_n} Features for Cell Type Classification')
        plt.tight_layout()
        
        # Save figure
        os.makedirs('plots', exist_ok=True)
        plt.savefig('plots/cell_type_feature_importance.png', dpi=300)
        
        return plt.gcf()
    
    def save_model(self, filename):
        """
        Save the trained model to a file.
        
        Parameters:
        -----------
        filename : str
            Path to save the model
            
        Returns:
        --------
        bool
            True if saved successfully
        """
        if self.model is None:
            raise ValueError("Model not trained. Call train() first.")
        
        # Create directory if it doesn't exist
        os.makedirs(os.path.dirname(filename) if os.path.dirname(filename) else '.', exist_ok=True)
        
        # Create dictionary with model and metadata
        model_data = {
            'model': self.model,
            'encoder': self.encoder,
            'top_genes': self.top_genes,
            'feature_importance': self.feature_importance,
            'model_type': self.model_type,
            'date_trained': datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        }
        
        # Save to file
        with open(filename, 'wb') as f:
            pickle.dump(model_data, f)
        
        print(f"Model saved to {filename}")
        return True
    
    def load_model(self, filename):
        """
        Load a trained model from file.
        
        Parameters:
        -----------
        filename : str
            Path to the model file
            
        Returns:
        --------
        object
            The loaded model
        """
        if not os.path.exists(filename):
            raise FileNotFoundError(f"Model file not found: {filename}")
        
        # Load from file
        with open(filename, 'rb') as f:
            model_data = pickle.load(f)
        
        # Extract model and metadata
        self.model = model_data['model']
        self.encoder = model_data['encoder']
        self.top_genes = model_data['top_genes']
        self.feature_importance = model_data['feature_importance']
        self.model_type = model_data['model_type']
        
        print(f"Model loaded from {filename}")
        print(f"Model type: {self.model_type}")
        print(f"Date trained: {model_data['date_trained']}")
        
        return self.model


class DrugResponsePredictor:
    """
    A predictor for drug response based on cell type and drug identity.
    
    This class trains a model to predict gene expression changes in response to drugs.
    """
    
    def __init__(self, n_top_genes=1000, model_type='rf'):
        """
        Initialize the drug response predictor.
        
        Parameters:
        -----------
        n_top_genes : int
            Number of top variable genes to predict
        model_type : str
            Model type ('rf' for Random Forest, 'xgb' for XGBoost)
        """
        self.n_top_genes = n_top_genes
        self.model_type = model_type
        self.model = None
        self.top_genes = None
        self.cell_encoder = LabelEncoder()
        self.drug_encoder = LabelEncoder()
        self.feature_importance = None
        self.X_train = None
        self.y_train = None
        self.X_test = None
        self.y_test = None
    
    def preprocess_data(self, df, test_size=0.2):
        """
        Preprocess the dataset for drug response prediction.
        
        Parameters:
        -----------
        df : pandas DataFrame
            Input dataset with gene expression data
        test_size : float
            Proportion of dataset to use for testing
            
        Returns:
        --------
        tuple
            X_train, X_test, y_train, y_test
        """
        # Verify required columns exist
        required_cols = ['cell_type', 'sm_name']
        for col in required_cols:
            if col not in df.columns:
                raise ValueError(f"Dataset must contain '{col}' column")
        
        # Identify metadata columns
metadata_cols = ['cell_type', 'sm_name', 'sm_lincs_id', 'SMILES', 'control']
gene_cols = [col for col in df.columns if col not in metadata_cols]

        print(f"Dataset contains {len(gene_cols)} gene expression features")
        
        # Filter out control samples
        if 'control' in df.columns:
            df = df[df['control'] == 0].copy()
            print(f"Filtered out control samples. Remaining samples: {len(df)}")
        
        # Select top variable genes
        if len(gene_cols) > self.n_top_genes:
            gene_variance = df[gene_cols].var().sort_values(ascending=False)
            self.top_genes = gene_variance.index[:self.n_top_genes].tolist()
            print(f"Selected top {len(self.top_genes)} variable genes")
        else:
            self.top_genes = gene_cols
            print(f"Using all {len(self.top_genes)} genes (less than requested {self.n_top_genes})")
        
        # Encode categorical features
        df['cell_type_encoded'] = self.cell_encoder.fit_transform(df['cell_type'])
        df['drug_encoded'] = self.drug_encoder.fit_transform(df['sm_name'])
        
        # Create feature matrix
        X = df[['cell_type_encoded', 'drug_encoded']].values
        y = df[self.top_genes].values
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=42
        )
        
        # Store data for later use
        self.X_train = X_train
        self.y_train = y_train
        self.X_test = X_test
        self.y_test = y_test
        
        print(f"Training data shape: {X_train.shape}, {y_train.shape}")
        print(f"Testing data shape: {X_test.shape}, {y_test.shape}")
        
        return X_train, X_test, y_train, y_test
    
    def train(self, X_train=None, y_train=None):
        """
        Train the drug response predictor.
        
        Parameters:
        -----------
        X_train : numpy array
            Training features
        y_train : numpy array
            Training targets
            
        Returns:
        --------
        object
            Trained model
        """
        if X_train is None:
            X_train = self.X_train
        if y_train is None:
            y_train = self.y_train
        
        if X_train is None or y_train is None:
            raise ValueError("Training data not provided. Call preprocess_data() first.")
        
        print(f"Training drug response predictor...")
        
        # Define model
        if self.model_type == 'rf':
            model = RandomForestRegressor(
                n_estimators=200,
                max_depth=30,
                n_jobs=-1,
                random_state=42
            )
        elif self.model_type == 'xgb':
            model = xgb.XGBRegressor(
                n_estimators=200,
                max_depth=7,
                learning_rate=0.05,
                n_jobs=-1,
                random_state=42
            )
        else:
            raise ValueError(f"Unknown model type: {self.model_type}")

# Train model
        model.fit(X_train, y_train)
        self.model = model
        
        # Calculate feature importance
        self.feature_importance = model.feature_importances_
        
        print(f"Model training complete")
        return self.model
    
    def evaluate(self, X_test=None, y_test=None):
        """
        Evaluate the predictor on test data.
        
        Parameters:
        -----------
        X_test : numpy array
            Test features
        y_test : numpy array
            Test targets
            
        Returns:
        --------
        dict
            Evaluation metrics
        """
        if X_test is None:
            X_test = self.X_test
        if y_test is None:
            y_test = self.y_test
            
        if X_test is None or y_test is None:
            raise ValueError("Test data not provided. Call preprocess_data() first.")
            
        if self.model is None:
            raise ValueError("Model not trained. Call train() first.")
        
        # Make predictions
        y_pred = self.model.predict(X_test)
        
        # Calculate metrics
        mse = mean_squared_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)
        
        print(f"Test MSE: {mse:.4f}")
        print(f"Test R²: {r2:.4f}")
        
        # Calculate per-gene metrics
        gene_metrics = []
        for i, gene in enumerate(self.top_genes):
            gene_mse = mean_squared_error(y_test[:, i], y_pred[:, i])
            gene_r2 = r2_score(y_test[:, i], y_pred[:, i])
            gene_metrics.append({
                'gene': gene,
                'mse': gene_mse,
                'r2': gene_r2
            })
        
        gene_metrics_df = pd.DataFrame(gene_metrics)
        
        results = {
            'mse': mse,
            'r2': r2,
            'gene_metrics': gene_metrics_df,
            'y_pred': y_pred,
            'y_test': y_test
        }
        
        return results
    
    def plot_prediction_performance(self, results, n_genes=5):
        """
        Plot actual vs predicted values for top genes.
        
        Parameters:
        -----------
        results : dict
            Results dictionary from evaluate()
        n_genes : int
            Number of top genes to plot
            
        Returns:
        --------
        matplotlib.figure.Figure
            The performance plot
        """
        # Sort genes by R²
        top_genes_by_r2 = results['gene_metrics'].sort_values('r2', ascending=False).head(n_genes)
        
        # Create figure
        fig, axes = plt.subplots(n_genes, 1, figsize=(10, 4*n_genes))
        
        for i, (_, row) in enumerate(top_genes_by_r2.iterrows()):
            gene = row['gene']
            gene_idx = self.top_genes.index(gene)
            
            # Get actual and predicted values
            y_actual = results['y_test'][:, gene_idx]
            y_pred = results['y_pred'][:, gene_idx]
            
            # Plot
            ax = axes[i] if n_genes > 1 else axes
            ax.scatter(y_actual, y_pred, alpha=0.5)
            
            # Calculate regression line
            min_val = min(y_actual.min(), y_pred.min())
            max_val = max(y_actual.max(), y_pred.max())
            ax.plot([min_val, max_val], [min_val, max_val], 'r--')
            
            ax.set_xlabel('Actual Expression')
            ax.set_ylabel('Predicted Expression')
            ax.set_title(f"Gene: {gene}, R²: {row['r2']:.4f}, MSE: {row['mse']:.4f}")
        
        plt.tight_layout()
        
        # Save figure
        os.makedirs('plots', exist_ok=True)
        plt.savefig('plots/drug_response_predictions.png', dpi=300)
        
        return fig
    
    def save_model(self, filename):
        """
        Save the trained model to a file.
        
        Parameters:
        -----------
        filename : str
            Path to save the model
            
        Returns:
        --------
        bool
            True if saved successfully
        """
        if self.model is None:
            raise ValueError("Model not trained. Call train() first.")
        
        # Create directory if it doesn't exist
        os.makedirs(os.path.dirname(filename) if os.path.dirname(filename) else '.', exist_ok=True)
        
        # Create dictionary with model and metadata
        model_data = {
            'model': self.model,
            'top_genes': self.top_genes,
            'cell_encoder': self.cell_encoder,
            'drug_encoder': self.drug_encoder,
            'feature_importance': self.feature_importance,
            'model_type': self.model_type,
            'date_trained': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            'X_train': self.X_train,
            'y_train': self.y_train,
            'X_test': self.X_test,
            'y_test': self.y_test
        }
        
        # Save to file
        with open(filename, 'wb') as f:
            pickle.dump(model_data, f)
        
        print(f"Model saved to {filename}")
        return True
    
    def load_model(self, filename):
        """
        Load a trained model from file.
        
        Parameters:
        -----------
        filename : str
            Path to the model file
            
        Returns:
        --------
        object
            The loaded model
        """
        if not os.path.exists(filename):
            raise FileNotFoundError(f"Model file not found: {filename}")
        
        # Load from file
        with open(filename, 'rb') as f:
            model_data = pickle.load(f)
        
        # Extract model and metadata
        self.model = model_data['model']
        self.top_genes = model_data['top_genes']
        self.cell_encoder = model_data['cell_encoder']
        self.drug_encoder = model_data['drug_encoder']
        self.feature_importance = model_data['feature_importance']
        self.model_type = model_data['model_type']
        self.X_train = model_data['X_train']
        self.y_train = model_data['y_train']
        self.X_test = model_data['X_test']
        self.y_test = model_data['y_test']
        
        print(f"Model loaded from {filename}")
        print(f"Model type: {self.model_type}")
        print(f"Date trained: {model_data['date_trained']}")
        
        return self.model


def main():
    """
    Main function to run the training pipeline.
    """
    # Parse command-line arguments
    parser = argparse.ArgumentParser(description='Train drug response and cell type models')
    parser.add_argument('--data', type=str, default='dataset/de_train_split.parquet',
                        help='Path to input data file')
    parser.add_argument('--cell-type-only', action='store_true',
                        help='Train only the cell type classifier')
    parser.add_argument('--drug-response-only', action='store_true',
                        help='Train only the drug response predictor')
    parser.add_argument('--model-type', type=str, default='rf', choices=['rf', 'xgb'],
                        help='Model type (rf: Random Forest, xgb: XGBoost)')
    parser.add_argument('--n-top-genes', type=int, default=1000,
                        help='Number of top variable genes to use')
    parser.add_argument('--save-dir', type=str, default='models',
                        help='Directory to save models')
    args = parser.parse_args()
    
    # Check if data file exists
    if not os.path.exists(args.data):
        print(f"Error: Data file '{args.data}' not found.")
        return 1
    
    # Create save directory if it doesn't exist
    os.makedirs(args.save_dir, exist_ok=True)
    
    # Load data
    print(f"Loading data from {args.data}...")
    if args.data.endswith('.parquet'):
        df = pd.read_parquet(args.data)
    elif args.data.endswith('.csv'):
        df = pd.read_csv(args.data)
    else:
        print("Error: Unsupported file format. Only .parquet and .csv are supported.")
        return 1
    
    print(f"Data loaded with shape: {df.shape}")
    
    # Train cell type classifier
    if not args.drug_response_only:
        print("\n=== Training Cell Type Classifier ===")
        classifier = CellTypeClassifier(n_top_genes=args.n_top_genes, model_type=args.model_type)
        
        # Preprocess data
        X, y = classifier.preprocess_data(df)
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        
        # Train model
        classifier.train(X_train, y_train)
        
        # Evaluate model
        results = classifier.evaluate(X_test, y_test)
        
        # Plot results
        classifier.plot_confusion_matrix(results)
        classifier.plot_feature_importance()

# Save model
        classifier.save_model(os.path.join(args.save_dir, 'cell_type_classifier.pkl'))
        
        # Save top genes for reference
        with open(os.path.join(args.save_dir, 'top_genes_classifier.pkl'), 'wb') as f:
            pickle.dump(classifier.top_genes, f)
        
        print("\nCell type classifier training complete.")
    
    # Train drug response predictor
    if not args.cell_type_only:
        print("\n=== Training Drug Response Predictor ===")
        predictor = DrugResponsePredictor(n_top_genes=args.n_top_genes, model_type=args.model_type)
        
        # Preprocess data
        X_train, X_test, y_train, y_test = predictor.preprocess_data(df)
        
        # Train model
        predictor.train(X_train, y_train)
        
        # Evaluate model
        results = predictor.evaluate(X_test, y_test)
        
        # Plot results
        predictor.plot_prediction_performance(results)

# Save model
        predictor.save_model(os.path.join(args.save_dir, 'drug_response_predictor.pkl'))
        
        # Save top genes for reference
        with open(os.path.join(args.save_dir, 'top_genes_predictor.pkl'), 'wb') as f:
            pickle.dump(predictor.top_genes, f)
        
        print("\nDrug response predictor training complete.")
    
    print("\nAll models trained successfully!")
    return 0


if __name__ == "__main__":
    main() 