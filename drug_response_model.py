import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt
import seaborn as sns

class DrugResponsePredictor:
    """
    A class to predict drug responses in single-cell data.
    
    This model combines gene expression data with drug and cell type information
    to predict gene expression changes in response to drug treatments.
    """
    
    def __init__(self, n_estimators=100, max_features='sqrt', max_depth=None):
        """
        Initialize the drug response predictor.
        
        Parameters:
        -----------
        n_estimators : int
            Number of trees in the random forest
        max_features : str or int
            Max features parameter for the random forest
        max_depth : int or None
            Maximum depth of the trees
        """
        self.n_estimators = n_estimators
        self.max_features = max_features
        self.max_depth = max_depth
        self.model = None
        self.feature_selector = None
        self.preprocessor = None
        
    def _preprocess_data(self, X, fit=False):
        """
        Preprocess the data including one-hot encoding categorical variables.
        
        Parameters:
        -----------
        X : pandas DataFrame
            The input data containing both numerical and categorical features
        fit : bool
            Whether to fit the preprocessor or just transform
            
        Returns:
        --------
        X_processed : numpy array
            The preprocessed data
        """
        if fit:
            # Define preprocessor
            categorical_cols = ['cell_type', 'sm_name']
            numerical_cols = [col for col in X.columns if col not in categorical_cols]
            
            # Create preprocessing steps
            categorical_processor = OneHotEncoder(handle_unknown='ignore')
            numerical_processor = StandardScaler()
            
            # Create column transformer
            self.preprocessor = ColumnTransformer(
                transformers=[
                    ('cat', categorical_processor, categorical_cols),
                    ('num', numerical_processor, numerical_cols)
                ]
            )
            
            # Fit and transform
            X_processed = self.preprocessor.fit_transform(X)
        else:
            X_processed = self.preprocessor.transform(X)
            
        return X_processed
    
    def fit(self, X, y, top_n_genes=None, test_size=0.2, random_state=42, tune_hyperparams=False):
        """
        Fit the model to the data.
        
        Parameters:
        -----------
        X : pandas DataFrame
            The input features including gene expression, cell types, and drug identifiers
        y : pandas DataFrame or Series or ndarray
            The target variables (gene expressions to predict)
        top_n_genes : int or None
            If provided, select only the top N most variable genes
        test_size : float
            The proportion of data to use for testing
        random_state : int
            Random seed for reproducibility
        tune_hyperparams : bool
            Whether to tune hyperparameters using GridSearchCV
            
        Returns:
        --------
        self : object
            Returns self
        """
        # Select top variable genes if requested
        if top_n_genes is not None and isinstance(y, pd.DataFrame):
            gene_variance = y.var().sort_values(ascending=False)
            selected_genes = gene_variance.head(top_n_genes).index
            y = y[selected_genes]
            print(f"Selected top {len(selected_genes)} most variable genes")
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=random_state
        )
        
        # Preprocess the data
        X_train_processed = self._preprocess_data(X_train, fit=True)
        X_test_processed = self._preprocess_data(X_test, fit=False)
        
        # Define model
        if tune_hyperparams:
            param_grid = {
                'n_estimators': [50, 100, 200],
                'max_features': ['sqrt', 'log2', None],
                'max_depth': [None, 10, 20, 30]
            }
            
            self.model = GridSearchCV(
                RandomForestRegressor(random_state=random_state),
                param_grid=param_grid,
                cv=3,
                n_jobs=-1,
                verbose=1
            )
        else:
            self.model = RandomForestRegressor(
                n_estimators=self.n_estimators,
                max_features=self.max_features,
                max_depth=self.max_depth,
                random_state=random_state,
                n_jobs=-1
            )
        
        # Fit model
        print("Fitting model...")
        self.model.fit(X_train_processed, y_train)
        
        # If grid search was used, update parameters
        if tune_hyperparams:
            print(f"Best hyperparameters: {self.model.best_params_}")
            self.n_estimators = self.model.best_params_['n_estimators']
            self.max_features = self.model.best_params_['max_features']
            self.max_depth = self.model.best_params_['max_depth']
        
        # Evaluate model
        train_preds = self.model.predict(X_train_processed)
        test_preds = self.model.predict(X_test_processed)
        
        train_mse = mean_squared_error(y_train, train_preds)
        test_mse = mean_squared_error(y_test, test_preds)
        train_r2 = r2_score(y_train, train_preds)
        test_r2 = r2_score(y_test, test_preds)
        
        print(f"Train MSE: {train_mse:.4f}, R²: {train_r2:.4f}")
        print(f"Test MSE: {test_mse:.4f}, R²: {test_r2:.4f}")
        
        # Store test data for later visualization
        self.X_test = X_test
        self.y_test = y_test
        self.y_pred = test_preds
        
        return self
    
    def predict(self, X):
        """
        Make predictions with the trained model.
        
        Parameters:
        -----------
        X : pandas DataFrame
            The input features
            
        Returns:
        --------
        y_pred : numpy array
            The predicted values
        """
        if self.model is None:
            raise ValueError("Model has not been fitted yet. Call fit() first.")
        
        X_processed = self._preprocess_data(X, fit=False)
        return self.model.predict(X_processed)
    
    def visualize_predictions(self, n_genes=5):
        """
        Visualize the predictions vs actual values for a subset of genes.
        
        Parameters:
        -----------
        n_genes : int
            Number of genes to visualize
            
        Returns:
        --------
        None
        """
        if not hasattr(self, 'y_test') or not hasattr(self, 'y_pred'):
            raise ValueError("Model has not been evaluated. Call fit() first.")
        
        # Select a few genes randomly
        if isinstance(self.y_test, pd.DataFrame):
            genes = np.random.choice(self.y_test.columns, size=min(n_genes, len(self.y_test.columns)), replace=False)
            
            plt.figure(figsize=(15, 10))
            for i, gene in enumerate(genes):
                plt.subplot(n_genes, 1, i+1)
                
                true_vals = self.y_test[gene].values
                pred_vals = self.y_pred[:, self.y_test.columns.get_loc(gene)]
                
                plt.scatter(true_vals, pred_vals, alpha=0.5)
                plt.plot([min(true_vals), max(true_vals)], [min(true_vals), max(true_vals)], 'r--')
                plt.title(f'Gene: {gene}')
                plt.xlabel('True Value')
                plt.ylabel('Predicted Value')
                
            plt.tight_layout()
            plt.show()
        else:
            # For single target regression
            plt.figure(figsize=(8, 6))
            plt.scatter(self.y_test, self.y_pred, alpha=0.5)
            plt.plot([min(self.y_test), max(self.y_test)], [min(self.y_test), max(self.y_test)], 'r--')
            plt.title('Predicted vs True Values')
            plt.xlabel('True Value')
            plt.ylabel('Predicted Value')
            plt.show()
    
    def feature_importance(self, top_n=20):
        """
        Visualize feature importance from the trained model.
        
        Parameters:
        -----------
        top_n : int
            Number of top features to show
            
        Returns:
        --------
        feature_importance_df : pandas DataFrame
            DataFrame containing feature names and their importance scores
        """
        if self.model is None:
            raise ValueError("Model has not been fitted yet. Call fit() first.")
        
        # Get feature names from preprocessor
        categorical_features = self.preprocessor.transformers_[0][1].get_feature_names_out(
            input_features=self.preprocessor.transformers_[0][2]
        )
        numerical_features = self.preprocessor.transformers_[1][2]
        all_features = np.concatenate([categorical_features, numerical_features])
        
        # Get importance scores
        if hasattr(self.model, 'best_estimator_'):
            importance = self.model.best_estimator_.feature_importances_
        else:
            importance = self.model.feature_importances_
        
        # Create dataframe
        feature_importance = pd.DataFrame({
            'Feature': all_features,
            'Importance': importance
        }).sort_values('Importance', ascending=False)
        
        # Visualize
        plt.figure(figsize=(10, 8))
        sns.barplot(x='Importance', y='Feature', data=feature_importance.head(top_n))
        plt.title(f'Top {top_n} Most Important Features')
        plt.tight_layout()
        plt.show()
        
        return feature_importance

# Example usage (not executed)
if __name__ == "__main__":
    # Load data
    df = pd.read_parquet('dataset/de_train_split.parquet')
    
    # Separate features and target
    metadata_cols = ['cell_type', 'sm_name', 'sm_lincs_id', 'SMILES', 'control']
    gene_cols = [col for col in df.columns if col not in metadata_cols]
    
    # Select subset of features for demonstration
    X = df[['cell_type', 'sm_name']]  # Categorical features
    y = df[gene_cols]  # Gene expression to predict
    
    # Initialize and fit model
    model = DrugResponsePredictor(n_estimators=100)
    model.fit(X, y, top_n_genes=100)  # Select top 100 most variable genes
    
    # Visualize results
    model.visualize_predictions(n_genes=3)
    importance_df = model.feature_importance(top_n=20) 