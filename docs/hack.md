# HackOS 4: Hack Bio

## Challenge Overview

Welcome to HackOS 4! This hackathon focuses on the open problem of single-cell perturbations.

### TL;DR of the Challenge
- You're given an input pair consisting of cell type and small molecule name(s)
- Task: predict a sequence of length 18211 corresponding to gene expressions
- Dataset: 614 samples (501 training, 113 testing)
- Repository: [https://github.com/aniketsrinivasan/hackos-4](https://github.com/aniketsrinivasan/hackos-4)

## Evaluation Criteria
1. **Model accuracy**: How well does your model perform at the task?
2. **Model efficiency**: How (time and memory) efficient is your model?
3. **Model creativity**: How creative is your solution?

## Winning Approaches from NeurIPS 2023

### 1st Place Winner Approach
- Used one-hot encoding of cell type and small molecule name(s)
- Added "statistics" to target values (mean, standard deviation, quartiles)
- Trained 1D CNNs, LSTM, and GRU with 5-fold cross-validation
- Found that BCELoss (binary cross-entropy loss) performed best

### 2nd Place Winner Approach
- Used a Transformer-based approach for sequence modeling
- Used truncated singular-value decomposition (SVD) to reduce sequence dimensionality from 18211 to 25/50/100
- Trained model with k-fold cross-validation

## Ideas to Explore

### Model Architectures
- Decision trees and ensemble methods (XGBoost)
- Neural networks (MLP, CNN, RNN, LSTM, GRU)
- Transformers
- Treat as time-series modeling problem

### Techniques
- Dimensionality reduction (SVD, autoencoders)
- Ensemble models combining different approaches
- Cell-type specific models (mixture of experts)
- Masked sequence modeling
- Transfer learning
- Curriculum learning
- Test-time adaptation
- Contrastive learning

### Feature Engineering
- One-hot encoding categorical variables
- Statistical features (mean, std, quartiles)
- External knowledge integration

## Biology Background

### Cellular Heterogeneity
- Human biology has ~37 trillion cells, divided into 200+ cell types
- Each cell type has a distinct molecular profile
- Gene expression is context-dependent

### Gene Regulation
- Cells respond to internal programs and external signals
- Transcriptional regulation involves multiple factors
- Different cell types respond differently to the same stimulus

### Single-Cell Analysis
- Single-cell RNA sequencing (scRNA-seq) reveals cell-specific responses
- Allows study of rare cell populations and cellular identity
- Multiomic technologies provide deeper insights

### Challenges
- scRNA-seq experiments are costly and time-consuming
- Biological responses have inherent variability
- Computational modeling helps fill gaps in experimental data

## References

Multiple scientific papers (see full instructions for complete references) 