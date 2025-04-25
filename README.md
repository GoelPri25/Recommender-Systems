# To run the file 
python train.py

## About This Project

This repository implements and evaluates Neural Collaborative Filtering (NCF) models for movie recommendation, based on the work by He et al. (2017). It specifically focuses on applying Generalized Matrix Factorization (GMF), Multi-Layer Perceptron (MLP), and their combined Neural Matrix Factorization (NeuMF) model to the MovieLens 1M dataset. The project investigates the impact of different rating thresholds, data splitting strategies, hyperparameter choices (MLP depth, predictive factor size, negative sampling rate), and pre-training on recommendation performance.

## Overview

This project provides an implementation and experimental analysis of Neural Collaborative Filtering (NCF) for enhancing movie recommendations on the MovieLens 1M dataset.

*   **Models Implemented:**
    *   **Generalized Matrix Factorization (GMF):** Captures linear latent factor interactions.
    *   **Multi-Layer Perceptron (MLP):** Models non-linear user-item interactions using concatenated embeddings.
    *   **Neural Matrix Factorization (NeuMF):** Combines GMF and MLP with separate embeddings, fusing their outputs for a final prediction, capturing both linear and non-linear patterns.
*   **Key Experiments & Analyses:**
    *   **Rating Threshold:** Compared performance using rating thresholds of 3 vs. 4 to define positive interactions.
    *   **Data Handling:** Evaluated different dataset splitting methods and negative sampling strategies.
    *   **Hyperparameter Tuning:** Conducted ablation studies on the number of MLP layers, predictive factor dimensions, and the number of negative samples per positive instance.
    *   **Model Comparison:** Assessed the performance of GMF, MLP, and NeuMF using Recall@10 and NDCG@10.
    *   **Pre-training:** Investigated the effect of pre-training GMF and MLP components before combining them in NeuMF.
    *   **Impact of K:** Analyzed how recommendation list length (K) affects Recall@K and NDCG@K.
*   **Core Findings:** NeuMF consistently outperformed standalone GMF and MLP, demonstrating the benefit of combining linear and non-linear modeling. Performance was sensitive to hyperparameter choices, with moderate MLP depth, larger predictive factors, and appropriate negative sampling yielding the best results. Overfitting was observed, highlighting the need for careful tuning and potential regularization.

