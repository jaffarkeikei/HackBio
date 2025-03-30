import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pickle
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import os
import sys
from matplotlib.colors import LinearSegmentedColormap
import shap

class ModelInterpreter:
    """
    A class for interpreting drug response prediction models and analyzing their results.
    
    This class provides methods for:
    - Loading and evaluating trained models
    - Calculating feature importance
    - Generating SHAP explanations
    - Visualizing model predictions and errors
    """
    
    def __init__(self, model_path=None, top_genes_path=None):
        """
        Initialize the model interpreter.
        
        Parameters:
        -----------
        model_path : str
            Path to the trained model pickle file
        top_genes_path : str
            Path to the pickle file containing top variable genes
        """
        self.model = None
        self.top_genes = None
        self.dataset = None
        self.explainer = None
        
        # Try to load model if paths are provided
        if model_path and os.path.exists(model_path):
            self.load_model(model_path)
        
        if top_genes_path and os.path.exists(top_genes_path):
            self.load_top_genes(top_genes_path)
    
    def load_model(self, model_path):
        """
        Load a trained model from a pickle file.
        
        Parameters:
        -----------
        model_path : str
            Path to the model pickle file
            
        Returns:
        --------
        object
            The loaded model
        """
        try:
            with open(model_path, 'rb') as f:
                self.model = pickle.load(f)
            print(f"Model loaded successfully from {model_path}")
            return self.model
        except Exception as e:
            print(f"Error loading model: {e}")
            return None
    
    def load_top_genes(self, genes_path):
        """
        Load the list of top variable genes.
        
        Parameters:
        -----------
        genes_path : str
            Path to the genes pickle file
            
        Returns:
        --------
        list
            List of top gene names
        """
        try:
            with open(genes_path, 'rb') as f:
                self.top_genes = pickle.load(f)
            print(f"Loaded {len(self.top_genes)} top variable genes")
            return self.top_genes
        except Exception as e:
            print(f"Error loading top genes: {e}")
            return None
    
    def load_dataset(self, dataset_path):
        """
        Load the dataset for evaluation.
        
        Parameters:
        -----------
        dataset_path : str
            Path to the dataset file (parquet or CSV)
            
        Returns:
        --------
        pandas DataFrame
            The loaded dataset
        """
        try:
            if dataset_path.endswith('.parquet'):
                self.dataset = pd.read_parquet(dataset_path)
            elif dataset_path.endswith('.csv'):
                self.dataset = pd.read_csv(dataset_path)
            else:
                raise ValueError("Unsupported file format. Only .parquet and .csv are supported.")
                
            print(f"Dataset loaded with shape: {self.dataset.shape}")
            return self.dataset
        except Exception as e:
            print(f"Error loading dataset: {e}")
            return None
    
    def evaluate_model(self, test_data=None, test_features=None, test_targets=None):
        """
        Evaluate the model on test data.
        
        Parameters:
        -----------
        test_data : pandas DataFrame or None
            Test dataset (if None, will use model's stored test data)
        test_features : pandas DataFrame or None
            Test features (if provided, overrides test_data)
        test_targets : pandas DataFrame or None
            Test targets (if provided, overrides test_data)
            
        Returns:
        --------
        dict
            Dictionary of evaluation metrics
        """
        if self.model is None:
            raise ValueError("No model loaded. Please load a model first.")
        
        # If using model's stored test data
        if test_data is None and test_features is None and hasattr(self.model, 'X_test'):
            test_features = self.model.X_test
            test_targets = self.model.y_test
        # If test_data is provided but not features/targets
        elif test_data is not None and (test_features is None or test_targets is None):
            # Extract metadata and gene columns
            metadata_cols = ['cell_type', 'sm_name', 'sm_lincs_id', 'SMILES', 'control']
            gene_cols = [col for col in test_data.columns if col not in metadata_cols]
            
            if self.top_genes is not None:
                gene_cols = [gene for gene in self.top_genes if gene in gene_cols]
                
            test_features = test_data[['cell_type', 'sm_name']]
            test_targets = test_data[gene_cols]
        
        # Make predictions
        predictions = self.model.predict(test_features)
        
        # Calculate metrics
        mse = mean_squared_error(test_targets, predictions)
        r2 = r2_score(test_targets, predictions)
        
        # Calculate per-gene metrics
        gene_metrics = []
        for i, gene in enumerate(test_targets.columns):
            gene_mse = mean_squared_error(test_targets[gene], predictions[:, i])
            gene_r2 = r2_score(test_targets[gene], predictions[:, i])
            gene_metrics.append({
                'gene': gene,
                'mse': gene_mse,
                'r2': gene_r2
            })
        
        gene_metrics_df = pd.DataFrame(gene_metrics)
        
        # Calculate per-cell-type metrics
        cell_metrics = []
        for cell_type in test_features['cell_type'].unique():
            cell_mask = test_features['cell_type'] == cell_type
            cell_mse = mean_squared_error(test_targets[cell_mask], predictions[cell_mask])
            cell_r2 = r2_score(test_targets[cell_mask], predictions[cell_mask])
            cell_metrics.append({
                'cell_type': cell_type,
                'mse': cell_mse,
                'r2': cell_r2,
                'sample_count': sum(cell_mask)
            })
        
        cell_metrics_df = pd.DataFrame(cell_metrics)
        
        # Calculate per-drug metrics
        drug_metrics = []
        for drug in test_features['sm_name'].unique():
            drug_mask = test_features['sm_name'] == drug
            if sum(drug_mask) > 1:  # Need at least 2 samples for r2
                drug_mse = mean_squared_error(test_targets[drug_mask], predictions[drug_mask])
                drug_r2 = r2_score(test_targets[drug_mask], predictions[drug_mask])
                drug_metrics.append({
                    'drug': drug,
                    'mse': drug_mse,
                    'r2': drug_r2,
                    'sample_count': sum(drug_mask)
                })
        
        drug_metrics_df = pd.DataFrame(drug_metrics)
        
        # Store results
        results = {
            'overall_mse': mse,
            'overall_r2': r2,
            'gene_metrics': gene_metrics_df,
            'cell_metrics': cell_metrics_df,
            'drug_metrics': drug_metrics_df,
            'predictions': predictions,
            'actual': test_targets,
            'features': test_features
        }
        
        print(f"Model evaluation complete. Overall MSE: {mse:.4f}, R²: {r2:.4f}")
        return results
    
    def generate_shap_explanations(self, test_samples=20):
        """
        Generate SHAP explanations for model predictions.
        
        Parameters:
        -----------
        test_samples : int
            Number of test samples to explain (random selection)
            
        Returns:
        --------
        dict
            Dictionary with SHAP values and explanation objects
        """
        if self.model is None:
            raise ValueError("No model loaded. Please load a model first.")
        
        if not hasattr(self.model, 'X_test'):
            raise ValueError("Model does not have test data. Please evaluate the model first.")
        
        # Handle case of wrapped model (e.g., GridSearchCV)
        if hasattr(self.model, 'best_estimator_'):
            model_to_explain = self.model.best_estimator_
        else:
            model_to_explain = self.model.model  # For DrugResponsePredictor
        
        # Select subset of test samples
        if test_samples < len(self.model.X_test):
            indices = np.random.choice(len(self.model.X_test), test_samples, replace=False)
            X_subset = self.model.X_test.iloc[indices]
        else:
            X_subset = self.model.X_test
        
        # Preprocess data for SHAP
        X_processed = self.model._preprocess_data(X_subset, fit=False)
        
        # Create explainer
        print("Generating SHAP explanations (this may take a while)...")
        try:
            explainer = shap.TreeExplainer(model_to_explain)
            shap_values = explainer.shap_values(X_processed)
            
            # Store results
            self.explainer = explainer
            
            results = {
                'explainer': explainer,
                'shap_values': shap_values,
                'X_subset': X_subset,
                'X_processed': X_processed
            }
            
            print("SHAP explanations generated successfully")
            return results
        except Exception as e:
            print(f"Error generating SHAP explanations: {e}")
            print("Falling back to simpler explanation method")
            
            # Fallback to feature importance
            return self.feature_importance()
    
    def plot_shap_summary(self, gene_index=0):
        """
        Plot SHAP summary for a specific gene.
        
        Parameters:
        -----------
        gene_index : int
            Index of the gene to explain
            
        Returns:
        --------
        matplotlib.figure.Figure
            The generated figure
        """
        if self.explainer is None:
            print("No SHAP explainer found. Generating explanations...")
            self.generate_shap_explanations()
            
        if self.explainer is None:
            raise ValueError("Could not generate SHAP explanations.")
        
        # Get feature names
        if hasattr(self.model, 'preprocessor') and hasattr(self.model.preprocessor, 'get_feature_names_out'):
            try:
                feature_names = self.model.preprocessor.get_feature_names_out()
            except:
                feature_names = None
        else:
            feature_names = None
        
        # Create figure
        plt.figure(figsize=(12, 8))
        
        # Get gene name
        if hasattr(self.model, 'y_test'):
            gene_name = self.model.y_test.columns[gene_index]
        else:
            gene_name = f"Gene_{gene_index}"
        
        # Plot summary
        shap.summary_plot(
            self.explainer.shap_values(gene_index), 
            feature_names=feature_names,
            show=False
        )
        
        plt.title(f"SHAP Feature Importance for {gene_name}")
        plt.tight_layout()
        
        return plt.gcf()
    
    def feature_importance(self, top_n=20):
        """
        Calculate and plot feature importance from the model.
        
        Parameters:
        -----------
        top_n : int
            Number of top features to show
            
        Returns:
        --------
        matplotlib.figure.Figure
            The generated figure
        """
        if self.model is None:
            raise ValueError("No model loaded. Please load a model first.")
        
        # Get underlying model
        if hasattr(self.model, 'best_estimator_'):
            model_to_use = self.model.best_estimator_
        else:
            model_to_use = self.model.model  # For DrugResponsePredictor
        
        # Extract feature importances
        if hasattr(model_to_use, 'feature_importances_'):
            importances = model_to_use.feature_importances_
        else:
            print("Model does not have feature_importances_ attribute.")
            return None
        
        # Get feature names
        if hasattr(self.model, 'preprocessor') and hasattr(self.model.preprocessor, 'get_feature_names_out'):
            try:
                feature_names = self.model.preprocessor.get_feature_names_out()
            except:
                feature_names = [f"Feature_{i}" for i in range(len(importances))]
        else:
            feature_names = [f"Feature_{i}" for i in range(len(importances))]
        
        # Create DataFrame for plotting
        importance_df = pd.DataFrame({
            'Feature': feature_names,
            'Importance': importances
        }).sort_values('Importance', ascending=False)
        
        # Keep top N features
        if top_n < len(importance_df):
            importance_df = importance_df.head(top_n)
        
        # Plot
        plt.figure(figsize=(12, 8))
        sns.barplot(x='Importance', y='Feature', data=importance_df)
        plt.title(f'Top {top_n} Feature Importances')
        plt.tight_layout()
        
        return plt.gcf()
    
    def plot_prediction_errors(self, results=None, n_genes=5):
        """
        Plot prediction errors for top genes.
        
        Parameters:
        -----------
        results : dict or None
            Results from evaluate_model (if None, will evaluate model)
        n_genes : int
            Number of genes to plot
            
        Returns:
        --------
        matplotlib.figure.Figure
            The generated figure
        """
        if results is None:
            if self.model is None:
                raise ValueError("No model loaded. Please load a model first.")
            
            if not hasattr(self.model, 'X_test'):
                raise ValueError("Model does not have test data. Please evaluate the model first.")
            
            results = self.evaluate_model()
        
        # Select top genes by R2
        top_genes = results['gene_metrics'].sort_values('r2', ascending=False).head(n_genes)
        
        # Create figure
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        axes = axes.flatten()
        
        # Create colormap for cell types
        cell_types = results['features']['cell_type'].unique()
        colors = sns.color_palette("hsv", len(cell_types))
        cell_color_map = {cell: color for cell, color in zip(cell_types, colors)}
        
        for i, (_, gene_row) in enumerate(top_genes.iterrows()):
            if i >= len(axes):
                break
            
            gene = gene_row['gene']
            gene_idx = list(results['actual'].columns).index(gene)
            
            # Extract predictions and actual values for this gene
            actual = results['actual'][gene].values
            predicted = results['predictions'][:, gene_idx]
            
            # Create scatter plot
            for cell_type in cell_types:
                mask = results['features']['cell_type'] == cell_type
                axes[i].scatter(
                    actual[mask], 
                    predicted[mask], 
                    c=[cell_color_map[cell_type]],
                    label=cell_type,
                    alpha=0.7
                )
            
            # Add diagonal line
            min_val = min(actual.min(), predicted.min())
            max_val = max(actual.max(), predicted.max())
            axes[i].plot([min_val, max_val], [min_val, max_val], 'k--', alpha=0.5)
            
            # Add metrics
            axes[i].set_title(f"{gene}\nR² = {gene_row['r2']:.3f}, MSE = {gene_row['mse']:.3f}")
            axes[i].set_xlabel('Actual')
            axes[i].set_ylabel('Predicted')
        
        # Add legend to the last subplot
        if len(axes) > i+1:
            handles, labels = axes[i].get_legend_handles_labels()
            axes[i+1].legend(handles, labels, title='Cell Type')
            axes[i+1].axis('off')  # Hide axes for legend-only subplot
        else:
            # Add legend outside the plots
            plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', title='Cell Type')
        
        # Remove empty subplots
        for j in range(i+2, len(axes)):
            fig.delaxes(axes[j])
        
        plt.tight_layout()
        return fig
    
    def plot_cell_type_performance(self, results=None):
        """
        Plot model performance by cell type.
        
        Parameters:
        -----------
        results : dict or None
            Results from evaluate_model (if None, will evaluate model)
            
        Returns:
        --------
        matplotlib.figure.Figure
            The generated figure
        """
        if results is None:
            results = self.evaluate_model()
        
        cell_metrics = results['cell_metrics'].sort_values('r2', ascending=False)
        
        # Create figure with two subplots
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
        
        # Plot R²
        sns.barplot(x='cell_type', y='r2', data=cell_metrics, ax=ax1)
        ax1.set_title('R² Score by Cell Type')
        ax1.set_xlabel('Cell Type')
        ax1.set_ylabel('R²')
        ax1.set_xticklabels(ax1.get_xticklabels(), rotation=45, ha='right')
        
        # Add sample counts
        for i, row in enumerate(cell_metrics.itertuples()):
            ax1.text(i, row.r2 / 2, f"n={row.sample_count}", 
                    ha='center', va='center', color='white', fontweight='bold')
        
        # Plot MSE
        sns.barplot(x='cell_type', y='mse', data=cell_metrics, ax=ax2)
        ax2.set_title('Mean Squared Error by Cell Type')
        ax2.set_xlabel('Cell Type')
        ax2.set_ylabel('MSE')
        ax2.set_xticklabels(ax2.get_xticklabels(), rotation=45, ha='right')
        
        # Add sample counts
        for i, row in enumerate(cell_metrics.itertuples()):
            ax2.text(i, row.mse / 2, f"n={row.sample_count}", 
                    ha='center', va='center', color='white', fontweight='bold')
        
        plt.tight_layout()
        return fig
    
    def plot_drug_performance(self, results=None, top_n=10):
        """
        Plot model performance by drug.
        
        Parameters:
        -----------
        results : dict or None
            Results from evaluate_model (if None, will evaluate model)
        top_n : int
            Number of top drugs to show
            
        Returns:
        --------
        matplotlib.figure.Figure
            The generated figure
        """
        if results is None:
            results = self.evaluate_model()
        
        # Sort by R2
        drug_metrics = results['drug_metrics'].sort_values('r2', ascending=False)
        
        # Limit to top N
        drug_metrics_top = drug_metrics.head(top_n)
        
        # Create figure with two subplots
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(18, 8))
        
        # Plot R²
        sns.barplot(x='r2', y='drug', data=drug_metrics_top, ax=ax1)
        ax1.set_title(f'R² Score for Top {top_n} Drugs')
        ax1.set_xlabel('R²')
        ax1.set_ylabel('Drug')
        
        # Add sample counts
        for i, row in enumerate(drug_metrics_top.itertuples()):
            ax1.text(row.r2 / 2, i, f"n={row.sample_count}", 
                    ha='center', va='center', color='white', fontweight='bold')
        
        # Plot MSE for the same drugs, in the same order
        drug_metrics_mse = drug_metrics_top.sort_values('drug').copy()
        drug_metrics_mse['drug'] = pd.Categorical(
            drug_metrics_mse['drug'], 
            categories=drug_metrics_top['drug'].values,
            ordered=True
        )
        drug_metrics_mse = drug_metrics_mse.sort_values('drug', ascending=False)
        
        sns.barplot(x='mse', y='drug', data=drug_metrics_mse, ax=ax2)
        ax2.set_title(f'Mean Squared Error for Top {top_n} Drugs')
        ax2.set_xlabel('MSE')
        ax2.set_ylabel('')  # No need to repeat the label
        
        # Add sample counts
        for i, row in enumerate(drug_metrics_mse.itertuples()):
            ax2.text(row.mse / 2, i, f"n={row.sample_count}", 
                    ha='center', va='center', color='white', fontweight='bold')
        
        plt.tight_layout()
        return fig
    
    def summarize_model(self, results=None):
        """
        Generate a comprehensive model summary.
        
        Parameters:
        -----------
        results : dict or None
            Results from evaluate_model (if None, will evaluate model)
            
        Returns:
        --------
        dict
            Dictionary of summary visualizations
        """
        if results is None:
            results = self.evaluate_model()
            
        # Generate summary plots
        summary = {}
        
        # 1. Feature importance
        print("Generating feature importance plot...")
        summary['feature_importance'] = self.feature_importance()
        
        # 2. Prediction errors
        print("Generating prediction error plots...")
        summary['prediction_errors'] = self.plot_prediction_errors(results)
        
        # 3. Cell type performance
        print("Generating cell type performance plot...")
        summary['cell_performance'] = self.plot_cell_type_performance(results)
        
        # 4. Drug performance
        print("Generating drug performance plot...")
        summary['drug_performance'] = self.plot_drug_performance(results)
        
        # 5. Try to generate SHAP summary
        try:
            print("Generating SHAP summary plot...")
            self.generate_shap_explanations()
            summary['shap_summary'] = self.plot_shap_summary()
        except Exception as e:
            print(f"Could not generate SHAP summary: {e}")
        
        # Save plots
        os.makedirs('plots', exist_ok=True)
        
        for name, fig in summary.items():
            if fig is not None:
                fig.savefig(f'plots/{name}.png', dpi=300, bbox_inches='tight')
        
        print("Model summary complete. Plots saved to plots/ directory")
        return summary


if __name__ == "__main__":
    # Check if models directory exists
    if not os.path.exists('models'):
        print("Error: 'models' directory not found. Please run train_model.py first.")
        sys.exit(1)
        
    # Initialize interpreter
    interpreter = ModelInterpreter(
        model_path='models/drug_response_model.pkl',
        top_genes_path='models/top_variable_genes.pkl'
    )
    
    # Load dataset if it exists
    if os.path.exists('dataset/de_train_split.parquet'):
        interpreter.load_dataset('dataset/de_train_split.parquet')
    
    # Generate comprehensive model summary
    interpreter.summarize_model()
    
    print("Model interpretation complete. Check the plots/ directory for visualizations.") 