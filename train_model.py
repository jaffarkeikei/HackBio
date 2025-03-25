import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
import os
import pickle
from drug_response_model import DrugResponsePredictor
from molecular_analysis import MolecularAnalyzer

# Set random seed for reproducibility
np.random.seed(42)

print("Loading dataset...")
df = pd.read_parquet('dataset/de_train_split.parquet')
print(f"Dataset loaded with shape: {df.shape}")

# Separate metadata and gene columns
metadata_cols = ['cell_type', 'sm_name', 'sm_lincs_id', 'SMILES', 'control']
gene_cols = [col for col in df.columns if col not in metadata_cols]

print(f"Number of gene features: {len(gene_cols)}")

# Create output directory for models
os.makedirs('models', exist_ok=True)

# Step 1: Analyze drug structures
print("\nAnalyzing drug structures...")
analyzer = MolecularAnalyzer(df['SMILES'])
descriptors = analyzer.calculate_descriptors()
print("Generated molecular descriptors for drugs")

# Save descriptors to CSV
descriptors.to_csv('molecular_descriptors.csv', index=False)
print("Saved molecular descriptors to molecular_descriptors.csv")

# Drug likeness analysis
drug_likeness = analyzer.analyze_drug_likeness()
print(f"Drug-like molecules: {drug_likeness['DrugLike'].sum()} out of {len(drug_likeness)}")

# Step 2: Train a model to predict cell type from gene expression
print("\nTraining cell type classification model...")

# Prepare data
X = df[gene_cols]
y = df['cell_type']

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Select top 500 variable genes for efficiency
gene_variance = X.var().sort_values(ascending=False)
top_variable_genes = gene_variance.head(500).index.tolist()
X_train_selected = X_train[top_variable_genes]
X_test_selected = X_test[top_variable_genes]

# Import Random Forest Classifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score

# Train model
clf = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
clf.fit(X_train_selected, y_train)

# Evaluate
y_pred = clf.predict(X_test_selected)
accuracy = accuracy_score(y_test, y_pred)
print(f"Cell type classification accuracy: {accuracy:.4f}")
print("\nClassification report:")
print(classification_report(y_test, y_pred))

# Save model
with open('models/cell_type_classifier.pkl', 'wb') as f:
    pickle.dump(clf, f)
print("Saved cell type classification model to models/cell_type_classifier.pkl")

# Step 3: Train drug response model
print("\nTraining drug response model...")

# We'll use the drug and cell type to predict gene expression
X_features = df[['cell_type', 'sm_name']]  # Using categorical features only
y_targets = df[top_variable_genes]  # Using top variable genes

# Create and train model
drug_model = DrugResponsePredictor(n_estimators=100)
drug_model.fit(X_features, y_targets, test_size=0.2, random_state=42)

# Save model
with open('models/drug_response_model.pkl', 'wb') as f:
    pickle.dump(drug_model, f)
print("Saved drug response model to models/drug_response_model.pkl")

# Save top variable genes list for future reference
with open('models/top_variable_genes.pkl', 'wb') as f:
    pickle.dump(top_variable_genes, f)
print("Saved list of top variable genes to models/top_variable_genes.pkl")

# Print completion message
print("\nModel training complete!")
print("Files generated:")
print("  - molecular_descriptors.csv")
print("  - models/cell_type_classifier.pkl")
print("  - models/drug_response_model.pkl")
print("  - models/top_variable_genes.pkl")

print("\nTo use these models for prediction, import them using:")
print("with open('models/drug_response_model.pkl', 'rb') as f:")
print("    model = pickle.load(f)")

print("\nNext steps:")
print("1. Run exploration_notebook.ipynb for interactive data exploration")
print("2. Use the models to predict drug responses for new compounds")
print("3. Analyze the top variable genes for biological insights") 