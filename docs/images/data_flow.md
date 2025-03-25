# Data Flow Diagram

This diagram illustrates how data flows through the single-cell perturbation analysis system:

```mermaid
flowchart LR
    subgraph Input
        A1[Single-Cell Expression Data]
        A2[Molecular Structure Data]
        A3[Cell Type Labels]
        A4[Drug Response Data]
    end

    subgraph Processing
        B1[Data Loading & Preprocessing]
        B2[Feature Engineering]
        B3[Quality Control]
    end

    subgraph Analysis
        C1[Molecular Analysis]
        C2[Correlation Analysis]
        C3[Clustering]
        C4[Model Training]
    end

    subgraph Output
        D1[Molecular Property Visualizations]
        D2[Correlation Heatmaps]
        D3[Predictive Models]
        D4[Performance Metrics]
        D5[Feature Importance]
    end

    A1 --> B1
    A2 --> B1
    A3 --> B1
    A4 --> B1
    
    B1 --> B2
    B2 --> B3
    
    B3 --> C1
    B3 --> C2
    B3 --> C3
    B3 --> C4
    
    C1 --> D1
    C2 --> D2
    C3 --> D2
    C4 --> D3
    
    D3 --> D4
    D3 --> D5

    %% Data transformation details
    B1 -.-> |"Parse, Clean, Normalize"| B2
    B2 -.-> |"Feature Selection, Scaling"| B3
    C1 -.-> |"Calculate Descriptors"| D1
    C2 -.-> |"Pearson/Spearman Correlation"| D2
    C4 -.-> |"RandomForest, XGBoost"| D3
```

The data flow diagram above shows how information is processed from raw input data through various processing and analysis stages, ultimately resulting in visualizations, models, and insights. 