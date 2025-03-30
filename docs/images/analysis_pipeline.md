# Analysis Pipeline Diagram

This diagram illustrates the complete analysis workflow implemented in our system:

```mermaid
graph TD
    classDef inputClass fill:#d1f0ff,stroke:#0077b6
    classDef processClass fill:#d8f3dc,stroke:#2d6a4f
    classDef analysisClass fill:#ffddd2,stroke:#e29578  
    classDef outputClass fill:#ddd0ff,stroke:#7b2cbf
    
    A1[Single-Cell Expression Data] --> B1[Load Expression Data]
    A2[Molecular Structure Data] --> B2[Parse SMILES]
    
    B1 --> C1[Normalize Expression]
    B2 --> C2[Calculate Molecular Descriptors]
    
    C1 --> D1[Feature Selection]
    C1 --> D2[Cell Type Classification]
    C2 --> D3[Molecule Clustering]
    
    D1 --> E1[Correlation Analysis]
    D2 --> E2[Expression-Drug Response Model]
    D3 --> E3[Structure-Activity Relationship]
    
    E1 --> F1[Correlation Heatmaps]
    E2 --> F2[Performance Metrics]
    E3 --> F3[Molecular Property Visualization]
    
    E1 --> G1[Gene-Drug Correlations]
    E2 --> G2[Feature Importance]
    E3 --> G3[Structure Analysis]
    
    G1 --> H[Biological Insights]
    G2 --> H
    G3 --> H
    
    %% Classification for styling
    class A1,A2 inputClass
    class B1,B2,C1,C2,D1,D2,D3 processClass
    class E1,E2,E3,G1,G2,G3 analysisClass
    class F1,F2,F3,H outputClass
    
    %% Module labels
    subgraph "molecular_analysis.py"
        B2
        C2
        D3
        E3
        F3
        G3
    end
    
    subgraph "correlation_analysis.py"
        E1
        F1
        G1
    end
    
    subgraph "model_interpreter.py"
        F2
        G2
    end
```

The analysis pipeline diagram shows the key computational steps in our analysis workflow, from data input through processing stages to final insights. Each color-coded section corresponds to a specific process type, and module boundaries indicate which Python file implements each component. 