import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import pearsonr, spearmanr
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
import os
import warnings

# Suppress specific warnings that might arise from correlation calculations
warnings.filterwarnings("ignore", category=RuntimeWarning, message="invalid value encountered in true_divide")
warnings.filterwarnings("ignore", category=RuntimeWarning, message="invalid value encountered in double_scalars")

class CorrelationAnalyzer:
    """
    A class for analyzing correlations between drug molecular properties and gene expression responses.
    
    This class provides methods for:
    - Loading and preprocessing drug response and molecular descriptor data
    - Calculating correlations between properties and responses
    - Identifying genes most affected by specific molecular properties
    - Visualizing correlation patterns
    """
    
    def __init__(self, descriptor_file='molecular_descriptors.csv'):
        """
        Initialize the correlation analyzer.
        
        Parameters:
        -----------
        descriptor_file : str
            Path to the file containing molecular descriptors
        """
        self.descriptors = None
        self.response_data = None
        self.correlation_matrix = None
        
        # Try to load descriptor file if it exists
        if os.path.exists(descriptor_file):
            self.descriptors = pd.read_csv(descriptor_file)
            print(f"Loaded molecular descriptors for {len(self.descriptors)} compounds")
        else:
            print(f"Warning: Descriptor file '{descriptor_file}' not found.")
    
    def load_response_data(self, dataset_file, smiles_to_drug_mapping=None):
        """
        Load gene expression response data.
        
        Parameters:
        -----------
        dataset_file : str
            Path to the dataset file containing gene expression data
        smiles_to_drug_mapping : dict or None
            Optional mapping from SMILES to drug names
            
        Returns:
        --------
        pandas DataFrame
            The loaded response data
        """
        # Load dataset
        print(f"Loading dataset from {dataset_file}...")
        
        if dataset_file.endswith('.parquet'):
            df = pd.read_parquet(dataset_file)
        elif dataset_file.endswith('.csv'):
            df = pd.read_csv(dataset_file)
        else:
            raise ValueError("Unsupported file format. Only .parquet and .csv are supported.")
        
        print(f"Dataset loaded with shape: {df.shape}")
        
        # Extract metadata
        metadata_cols = ['cell_type', 'sm_name', 'sm_lincs_id', 'SMILES', 'control']
        metadata = [col for col in metadata_cols if col in df.columns]
        
        # Store response data
        self.response_data = df
        
        # Map SMILES to drug names if needed
        if smiles_to_drug_mapping is not None:
            if 'SMILES' in df.columns:
                # Create a new column with mapped drug names
                df['mapped_drug'] = df['SMILES'].map(smiles_to_drug_mapping)
                print("Applied SMILES to drug name mapping")
        
        return df
    
    def prepare_mean_responses(self, by='sm_name', control_norm=True):
        """
        Calculate mean gene expression for each drug.
        
        Parameters:
        -----------
        by : str
            Column to group by (usually drug identifier)
        control_norm : bool
            Whether to normalize by control values
            
        Returns:
        --------
        pandas DataFrame
            Mean expression values for each drug
        """
        if self.response_data is None:
            raise ValueError("No response data loaded. Call load_response_data() first.")
        
        # Identify gene columns
        metadata_cols = ['cell_type', 'sm_name', 'sm_lincs_id', 'SMILES', 'control', 'mapped_drug']
        gene_cols = [col for col in self.response_data.columns if col not in metadata_cols]
        
        # Extract control samples if normalizing
        if control_norm and 'control' in self.response_data.columns:
            control_samples = self.response_data[self.response_data['control'] == 1]
            control_means = control_samples[gene_cols].mean()
            
            # Normalize treatment samples
            treatment_samples = self.response_data[self.response_data['control'] == 0]
            
            # Calculate fold change compared to control (log2-scale)
            normalized = pd.DataFrame()
            normalized[gene_cols] = treatment_samples[gene_cols].apply(lambda x: x - control_means)
            normalized[by] = treatment_samples[by]
            
            # Group by drug and calculate mean
            mean_responses = normalized.groupby(by)[gene_cols].mean()
        else:
            # Group by drug and calculate mean without normalization
            mean_responses = self.response_data.groupby(by)[gene_cols].mean()
        
        print(f"Calculated mean responses for {len(mean_responses)} drugs across {len(gene_cols)} genes")
        return mean_responses
    
    def calculate_correlations(self, mean_responses, method='pearson', min_abs_corr=0.4):
        """
        Calculate correlations between molecular descriptors and gene expression.
        
        Parameters:
        -----------
        mean_responses : pandas DataFrame
            Mean gene expression responses
        method : str
            Correlation method ('pearson' or 'spearman')
        min_abs_corr : float
            Minimum absolute correlation value to consider significant
            
        Returns:
        --------
        pandas DataFrame
            Correlation matrix
        """
        if self.descriptors is None:
            raise ValueError("No descriptor data available.")
        
        # Ensure drug IDs are consistent
        common_drugs = set(mean_responses.index) & set(self.descriptors['Drug'] 
                                                     if 'Drug' in self.descriptors.columns 
                                                     else self.descriptors.index)
        
        if len(common_drugs) == 0:
            raise ValueError("No common drugs found between descriptors and response data.")
        
        print(f"Found {len(common_drugs)} common drugs for correlation analysis")
        
        # Filter data to common drugs
        if 'Drug' in self.descriptors.columns:
            filtered_descriptors = self.descriptors[self.descriptors['Drug'].isin(common_drugs)]
            descriptor_cols = [col for col in self.descriptors.columns if col != 'Drug']
            desc_values = filtered_descriptors[descriptor_cols].values
            desc_index = filtered_descriptors['Drug']
        else:
            filtered_descriptors = self.descriptors.loc[self.descriptors.index.isin(common_drugs)]
            descriptor_cols = self.descriptors.columns
            desc_values = filtered_descriptors.values
            desc_index = filtered_descriptors.index
        
        filtered_responses = mean_responses.loc[mean_responses.index.isin(common_drugs)]
        gene_cols = filtered_responses.columns
        
        # Calculate correlations
        correlation_matrix = pd.DataFrame(index=descriptor_cols, columns=gene_cols)
        p_values = pd.DataFrame(index=descriptor_cols, columns=gene_cols)
        
        for i, desc in enumerate(descriptor_cols):
            desc_data = desc_values[:, i]
            
            for gene in gene_cols:
                gene_data = filtered_responses[gene].values
                
                # Remove any NaN values
                valid_indices = ~np.isnan(desc_data) & ~np.isnan(gene_data)
                
                if valid_indices.sum() < 3:  # Need at least 3 points for correlation
                    correlation_matrix.loc[desc, gene] = np.nan
                    p_values.loc[desc, gene] = np.nan
                    continue
                
                if method == 'pearson':
                    corr, p_val = pearsonr(desc_data[valid_indices], gene_data[valid_indices])
                elif method == 'spearman':
                    corr, p_val = spearmanr(desc_data[valid_indices], gene_data[valid_indices])
                else:
                    raise ValueError(f"Unknown correlation method: {method}")
                
                correlation_matrix.loc[desc, gene] = corr
                p_values.loc[desc, gene] = p_val
        
        # Store results
        self.correlation_matrix = correlation_matrix
        self.p_values = p_values
        
        # Filter significant correlations
        significant_corrs = correlation_matrix[correlation_matrix.abs() >= min_abs_corr]
        
        print(f"Found {significant_corrs.count().sum()} significant correlations (|corr| >= {min_abs_corr})")
        return correlation_matrix, p_values
    
    def plot_correlation_heatmap(self, n_top_genes=20):
        """
        Plot a heatmap of top correlations.
        
        Parameters:
        -----------
        n_top_genes : int
            Number of top genes to include in the heatmap
            
        Returns:
        --------
        matplotlib.figure.Figure
            The generated figure
        """
        if self.correlation_matrix is None:
            raise ValueError("No correlation matrix available. Call calculate_correlations() first.")
        
        # Calculate the total absolute correlation for each gene
        gene_total_abs_corr = self.correlation_matrix.abs().sum()
        
        # Select top genes
        top_genes = gene_total_abs_corr.sort_values(ascending=False).head(n_top_genes).index
        
        # Create heatmap
        plt.figure(figsize=(14, 10))
        selected_corr = self.correlation_matrix[top_genes]
        
        # Mask non-significant correlations
        mask = np.abs(selected_corr) < 0.4
        
        # Plot heatmap
        sns.heatmap(selected_corr, cmap='coolwarm', center=0, vmin=-1, vmax=1, 
                    mask=mask, annot=False, linewidths=0.5)
        plt.title(f'Correlation Between Molecular Properties and Top {n_top_genes} Genes')
        plt.xticks(rotation=45, ha='right')
        plt.tight_layout()
        
        return plt.gcf()
    
    def find_property_gene_relationships(self, top_n=10):
        """
        Find strongest relationships between properties and genes.
        
        Parameters:
        -----------
        top_n : int
            Number of top relationships to find
            
        Returns:
        --------
        pandas DataFrame
            Top property-gene relationships
        """
        if self.correlation_matrix is None:
            raise ValueError("No correlation matrix available. Call calculate_correlations() first.")
        
        # Flatten the correlation matrix
        corr_flat = self.correlation_matrix.stack().reset_index()
        corr_flat.columns = ['Property', 'Gene', 'Correlation']
        
        # Get absolute correlation
        corr_flat['AbsCorrelation'] = corr_flat['Correlation'].abs()
        
        # Filter NaN values
        corr_flat = corr_flat.dropna()
        
        # Get top correlations
        top_pos = corr_flat[corr_flat['Correlation'] > 0].sort_values('Correlation', ascending=False).head(top_n)
        top_neg = corr_flat[corr_flat['Correlation'] < 0].sort_values('Correlation', ascending=True).head(top_n)
        
        # Combine results
        top_relationships = pd.concat([top_pos, top_neg])
        
        # Add p-values
        p_flat = self.p_values.stack().reset_index()
        p_flat.columns = ['Property', 'Gene', 'p_value']
        
        top_relationships = pd.merge(top_relationships, p_flat, on=['Property', 'Gene'])
        
        # Sort by absolute correlation
        top_relationships = top_relationships.sort_values('AbsCorrelation', ascending=False)
        
        return top_relationships
    
    def cluster_properties_by_effect(self, n_clusters=3):
        """
        Cluster molecular properties by their effect on gene expression.
        
        Parameters:
        -----------
        n_clusters : int
            Number of clusters to create
            
        Returns:
        --------
        tuple
            (cluster labels, clustered correlation matrix)
        """
        if self.correlation_matrix is None:
            raise ValueError("No correlation matrix available. Call calculate_correlations() first.")
        
        # Prepare data for clustering
        # Replace NaN with 0 for clustering
        corr_data = self.correlation_matrix.fillna(0).values
        
        # Standardize the data
        scaler = StandardScaler()
        corr_scaled = scaler.fit_transform(corr_data)
        
        # Apply K-means clustering
        kmeans = KMeans(n_clusters=n_clusters, random_state=42)
        labels = kmeans.fit_predict(corr_scaled)
        
        # Add cluster information to properties
        property_clusters = pd.DataFrame({
            'Property': self.correlation_matrix.index,
            'Cluster': labels
        })
        
        # Sort correlation matrix by cluster
        sorted_indices = property_clusters.sort_values('Cluster').index
        sorted_corr = self.correlation_matrix.iloc[sorted_indices]
        
        return property_clusters, sorted_corr
    
    def plot_clustered_heatmap(self, n_clusters=3, n_top_genes=30):
        """
        Plot a clustered heatmap of property-gene correlations.
        
        Parameters:
        -----------
        n_clusters : int
            Number of clusters for properties
        n_top_genes : int
            Number of top genes to include
            
        Returns:
        --------
        matplotlib.figure.Figure
            The generated figure
        """
        # Cluster properties
        property_clusters, _ = self.cluster_properties_by_effect(n_clusters)
        
        # Select top genes
        gene_total_abs_corr = self.correlation_matrix.abs().sum()
        top_genes = gene_total_abs_corr.sort_values(ascending=False).head(n_top_genes).index
        
        # Prepare data
        selected_corr = self.correlation_matrix[top_genes].copy()
        
        # Sort by cluster
        clustered_props = property_clusters.sort_values('Cluster')
        selected_corr = selected_corr.loc[clustered_props['Property']]
        
        # Create figure
        plt.figure(figsize=(16, 12))
        
        # Create cluster boundary lines
        cluster_boundaries = []
        current_cluster = -1
        
        for i, cluster in enumerate(clustered_props['Cluster']):
            if cluster != current_cluster:
                cluster_boundaries.append(i)
                current_cluster = cluster
        
        # Plot heatmap
        sns.heatmap(selected_corr, cmap='coolwarm', center=0, vmin=-1, vmax=1, 
                    annot=False, linewidths=0.5)
        
        # Add cluster separators
        for boundary in cluster_boundaries[1:]:
            plt.axhline(y=boundary, color='black', linestyle='-', linewidth=2)
        
        plt.title(f'Clustered Molecular Properties by Gene Expression Effect')
        plt.xticks(rotation=45, ha='right')
        plt.ylabel('Molecular Properties (Clustered)')
        plt.xlabel('Genes')
        plt.tight_layout()
        
        return plt.gcf()


if __name__ == "__main__":
    # Example usage
    print("Initializing correlation analysis...")
    analyzer = CorrelationAnalyzer()
    
    # Check if dataset exists
    if os.path.exists('dataset/de_train_split.parquet'):
        # Load response data
        df = analyzer.load_response_data('dataset/de_train_split.parquet')
        
        # Calculate mean responses
        mean_responses = analyzer.prepare_mean_responses(by='sm_name')
        
        # If we have descriptor data
        if analyzer.descriptors is not None:
            # Calculate correlations
            corr_matrix, p_vals = analyzer.calculate_correlations(mean_responses, method='pearson')
            
            # Plot correlation heatmap
            fig = analyzer.plot_correlation_heatmap(n_top_genes=20)
            fig.savefig('plots/property_gene_correlations.png', dpi=300, bbox_inches='tight')
            
            # Find top relationships
            top_relations = analyzer.find_property_gene_relationships(top_n=15)
            print("\nTop Property-Gene Relationships:")
            print(top_relations[['Property', 'Gene', 'Correlation', 'p_value']])
            
            # Plot clustered heatmap
            fig_clustered = analyzer.plot_clustered_heatmap(n_clusters=3, n_top_genes=30)
            fig_clustered.savefig('plots/clustered_property_gene_correlations.png', dpi=300, bbox_inches='tight')
            
            print("\nAnalysis complete. Results saved to plots/ directory")
        else:
            print("No descriptor data available. Please run molecular_analysis.py first to generate descriptors.")
    else:
        print("Dataset not found. Please ensure the dataset file exists in the dataset directory.") 