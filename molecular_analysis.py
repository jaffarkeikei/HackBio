import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from rdkit import Chem
from rdkit.Chem import Draw, AllChem, Descriptors, Lipinski, rdMolDescriptors, DataStructs
from rdkit.Chem.Draw import IPythonConsole
from rdkit.Chem.Fingerprints import FingerprintMols
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.cluster import KMeans, DBSCAN
import seaborn as sns

class MolecularAnalyzer:
    """
    A class for analyzing drug molecular structures using RDKit.
    
    This class provides methods for:
    - Computing molecular descriptors from SMILES strings
    - Generating molecular fingerprints
    - Clustering drugs based on structural similarity
    - Visualizing molecular structures and properties
    """
    
    def __init__(self, smiles_data):
        """
        Initialize the molecular analyzer.
        
        Parameters: 
        -----------
        smiles_data : pandas Series or list
            SMILES strings representing the molecular structures
        """
        self.smiles_data = smiles_data
        self.mols = self._convert_smiles_to_mol()
        self.descriptors = None
        self.fingerprints = None
        
    def _convert_smiles_to_mol(self):
        """
        Convert SMILES strings to RDKit molecule objects.
        
        Returns:
        --------
        list
            List of RDKit molecule objects
        """
        mols = []
        valid_indices = []
        
        for i, smiles in enumerate(self.smiles_data):
            if isinstance(smiles, str):
                mol = Chem.MolFromSmiles(smiles)
                if mol is not None:
                    mols.append(mol)
                    valid_indices.append(i)
        
        # Store valid indices for later reference
        self.valid_indices = valid_indices
        
        print(f"Successfully converted {len(mols)} out of {len(self.smiles_data)} SMILES strings")
        return mols
    
    def calculate_descriptors(self):
        """
        Calculate molecular descriptors for all molecules.
        
        Returns:
        --------
        pandas DataFrame
            DataFrame containing calculated descriptors
        """
        if not self.mols:
            raise ValueError("No valid molecules found")
        
        descriptors_list = []
        
        for mol in self.mols:
            # Calculate basic descriptors
            mw = Descriptors.ExactMolWt(mol)
            logp = Descriptors.MolLogP(mol)
            tpsa = Descriptors.TPSA(mol)
            h_donors = Lipinski.NumHDonors(mol)
            h_acceptors = Lipinski.NumHAcceptors(mol)
            rotatable_bonds = Descriptors.NumRotatableBonds(mol)
            num_atoms = mol.GetNumAtoms()
            num_rings = rdMolDescriptors.CalcNumRings(mol)
            
            # Store descriptors
            descriptors_list.append({
                'MolecularWeight': mw,
                'LogP': logp,
                'TPSA': tpsa,
                'HBondDonors': h_donors,
                'HBondAcceptors': h_acceptors,
                'RotatableBonds': rotatable_bonds,
                'NumAtoms': num_atoms,
                'NumRings': num_rings
            })
        
        # Create DataFrame
        self.descriptors = pd.DataFrame(descriptors_list)
        return self.descriptors
    
    def generate_fingerprints(self, fp_type='morgan', radius=2, n_bits=2048):
        """
        Generate molecular fingerprints for all molecules.
        
        Parameters:
        -----------
        fp_type : str
            Type of fingerprint to generate ('morgan', 'maccs', 'rdkit')
        radius : int
            Radius for Morgan fingerprints
        n_bits : int
            Number of bits for Morgan fingerprints
            
        Returns:
        --------
        numpy array
            Array of molecular fingerprints
        """
        if not self.mols:
            raise ValueError("No valid molecules found")
        
        fingerprints = []
        
        for mol in self.mols:
            if fp_type == 'morgan':
                fp = AllChem.GetMorganFingerprintAsBitVect(mol, radius, nBits=n_bits)
            elif fp_type == 'maccs':
                fp = AllChem.GetMACCSKeysFingerprint(mol)
            elif fp_type == 'rdkit':
                fp = FingerprintMols.FingerprintMol(mol)
            else:
                raise ValueError(f"Unknown fingerprint type: {fp_type}")
            
            # Convert to numpy array
            fp_array = np.zeros((0,), dtype=np.int8)
            DataStructs.ConvertToNumpyArray(fp, fp_array)
            fingerprints.append(fp_array)
        
        self.fingerprints = np.array(fingerprints)
        return self.fingerprints
    
    def cluster_molecules(self, n_clusters=5, method='kmeans'):
        """
        Cluster molecules based on their fingerprints or descriptors.
        
        Parameters:
        -----------
        n_clusters : int
            Number of clusters for KMeans
        method : str
            Clustering method ('kmeans' or 'dbscan')
            
        Returns:
        --------
        numpy array
            Array of cluster labels
        """
        # Ensure we have fingerprints
        if self.fingerprints is None:
            try:
                self.generate_fingerprints()
            except NameError:
                # If there's an issue with fingerprints, use descriptors
                if self.descriptors is None:
                    self.calculate_descriptors()
                data = self.descriptors
            else:
                data = self.fingerprints
        else:
            data = self.fingerprints
        
        # Apply dimensionality reduction if using fingerprints
        if data.shape[1] > 50:  # If using fingerprints
            pca = PCA(n_components=20)
            data_reduced = pca.fit_transform(data)
        else:  # If using descriptors
            data_reduced = data
        
        # Cluster the molecules
        if method == 'kmeans':
            clusterer = KMeans(n_clusters=n_clusters, random_state=42)
            cluster_labels = clusterer.fit_predict(data_reduced)
        elif method == 'dbscan':
            clusterer = DBSCAN(eps=0.5, min_samples=5)
            cluster_labels = clusterer.fit_predict(data_reduced)
        else:
            raise ValueError(f"Unknown clustering method: {method}")
        
        self.cluster_labels = cluster_labels
        return cluster_labels
    
    def visualize_clusters(self):
        """
        Visualize molecule clusters using t-SNE.
        
        Returns:
        --------
        None
        """
        if not hasattr(self, 'cluster_labels'):
            self.cluster_molecules()
        
        # Ensure we have data to work with
        if self.fingerprints is not None:
            data = self.fingerprints
        elif self.descriptors is not None:
            data = self.descriptors
        else:
            self.calculate_descriptors()
            data = self.descriptors
        
        # Apply t-SNE
        tsne = TSNE(n_components=2, random_state=42)
        tsne_result = tsne.fit_transform(data)
        
        # Plot results
        plt.figure(figsize=(10, 8))
        scatter = plt.scatter(tsne_result[:, 0], tsne_result[:, 1], c=self.cluster_labels, 
                             cmap='viridis', alpha=0.7)
        plt.colorbar(scatter, label='Cluster')
        plt.title('t-SNE Visualization of Molecule Clusters')
        plt.tight_layout()
        plt.show()
    
    def visualize_molecules(self, indices=None, n_mols=10):
        """
        Visualize molecular structures.
        
        Parameters:
        -----------
        indices : list or None
            Indices of molecules to visualize, if None, random selection is used
        n_mols : int
            Number of molecules to visualize if indices is None
            
        Returns:
        --------
        None
        """
        if not self.mols:
            raise ValueError("No valid molecules found")
        
        if indices is None:
            # Randomly select molecules
            indices = np.random.choice(len(self.mols), size=min(n_mols, len(self.mols)), replace=False)
        
        # Select molecules
        selected_mols = [self.mols[i] for i in indices]
        
        # Add molecule names if available
        legends = [f"Molecule {i}" for i in indices]
        
        # Display molecules
        img = Draw.MolsToGridImage(selected_mols, molsPerRow=5, subImgSize=(200, 200),
                                   legends=legends)
        display(img)
    
    def draw_molecule(self, index):
        """
        Draw a single molecule.
        
        Parameters:
        -----------
        index : int
            Index of the molecule to visualize
            
        Returns:
        --------
        None
        """
        if index >= len(self.mols):
            raise ValueError(f"Index {index} out of range, only {len(self.mols)} molecules available")
        
        mol = self.mols[index]
        display(Draw.MolToImage(mol, size=(400, 400)))
        
        # Print some basic information
        print(f"Molecule {index}")
        print(f"SMILES: {Chem.MolToSmiles(mol)}")
        print(f"Molecular Weight: {Descriptors.ExactMolWt(mol):.2f}")
        print(f"LogP: {Descriptors.MolLogP(mol):.2f}")
        print(f"H-Bond Donors: {Lipinski.NumHDonors(mol)}")
        print(f"H-Bond Acceptors: {Lipinski.NumHAcceptors(mol)}")
        
    def plot_descriptor_distributions(self):
        """
        Plot distributions of molecular descriptors.
        
        Returns:
        --------
        None
        """
        if self.descriptors is None:
            self.calculate_descriptors()
        
        # Plot distributions
        fig, axes = plt.subplots(4, 2, figsize=(15, 20))
        axes = axes.flatten()
        
        for i, col in enumerate(self.descriptors.columns):
            sns.histplot(self.descriptors[col], ax=axes[i], kde=True)
            axes[i].set_title(f'Distribution of {col}')
            axes[i].set_xlabel(col)
            
        plt.tight_layout()
        plt.show()
        
        # Correlation matrix
        plt.figure(figsize=(10, 8))
        sns.heatmap(self.descriptors.corr(), annot=True, cmap='coolwarm', center=0)
        plt.title('Correlation Matrix of Molecular Descriptors')
        plt.tight_layout()
        plt.show()
    
    def analyze_drug_likeness(self):
        """
        Analyze drug-likeness based on Lipinski's Rule of Five.
        
        Returns:
        --------
        pandas DataFrame
            DataFrame with drug-likeness properties
        """
        if not self.mols:
            raise ValueError("No valid molecules found")
        
        results = []
        
        for i, mol in enumerate(self.mols):
            mw = Descriptors.ExactMolWt(mol)
            logp = Descriptors.MolLogP(mol)
            h_donors = Lipinski.NumHDonors(mol)
            h_acceptors = Lipinski.NumHAcceptors(mol)
            
            # Lipinski's Rule of Five
            lipinski_violations = 0
            if mw > 500: lipinski_violations += 1
            if logp > 5: lipinski_violations += 1
            if h_donors > 5: lipinski_violations += 1
            if h_acceptors > 10: lipinski_violations += 1
            
            drug_like = lipinski_violations <= 1
            
            results.append({
                'MoleculeIndex': i,
                'MolecularWeight': mw,
                'LogP': logp,
                'HBondDonors': h_donors,
                'HBondAcceptors': h_acceptors,
                'LipinskiViolations': lipinski_violations,
                'DrugLike': drug_like
            })
        
        df_results = pd.DataFrame(results)
        
        # Plot summary
        plt.figure(figsize=(8, 6))
        sns.countplot(x='LipinskiViolations', data=df_results)
        plt.title("Lipinski's Violations Distribution")
        plt.xlabel('Number of Violations')
        plt.ylabel('Count')
        plt.show()
        
        print(f"Drug-like molecules: {df_results['DrugLike'].sum()} out of {len(df_results)} ({df_results['DrugLike'].mean()*100:.1f}%)")
        
        return df_results

# Example usage (not executed)
if __name__ == "__main__":
    # Load data
    df = pd.read_parquet('dataset/de_train_split.parquet')
    
    # Create analyzer
    analyzer = MolecularAnalyzer(df['SMILES'])
    
    # Calculate descriptors
    descriptors = analyzer.calculate_descriptors()
    print(descriptors.head())
    
    # Visualize random molecules
    analyzer.visualize_molecules()
    
    # Analyze drug-likeness
    drug_likeness = analyzer.analyze_drug_likeness()
    
    # Cluster molecules
    cluster_labels = analyzer.cluster_molecules(n_clusters=6)
    analyzer.visualize_clusters() 