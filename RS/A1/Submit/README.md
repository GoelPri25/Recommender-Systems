# Neural Collaborative Filtering on MovieLens 1M

This repository implements Neural Collaborative Filtering (NCF) models—GMF, MLP, and NeuMF—on the MovieLens 1M dataset. It includes data preprocessing, model training, evaluation, and ablation studies to analyze the effects of key architectural and hyperparameter choices.

## 📁 File Descriptions

| File               | Description |
|--------------------|-------------|
| `data_load.py`     | Loads and preprocesses the MovieLens 1M dataset. Handles data splitting and negative sampling. |
| `evaluation_f1.py` | Computes F1 score for evaluating the effect of different rating thresholds. |
| `evaluation.py`    | Calculates NDCG@10 and Recall@10 for recommendation performance evaluation. |
| `factors.py`       | Ablation experiment on the impact of predictive factor size (latent dimension). |
| `layers.py`        | Ablation experiment on the number of hidden layers in the MLP architecture. |
| `model.py`         | Compares GMF, MLP, and NeuMF models under unified training settings. |
| `GMF.py`           | Defines the standalone GMF model architecture. |
| `MLP.py`           | Defines the standalone MLP model architecture. |
| `NeuMF.py`         | Defines the full NeuMF model combining GMF and MLP components. |
| `negsample.py`     | Tests the effect of different negative sampling rates on performance. |
| `pre_train.py`     | Compares NeuMF with and without pre-training using saved GMF and MLP weights. |
| `threshold.py`     | Evaluates model performance under different rating threshold settings (e.g., 3 vs. 4). |

## 📦 Pre-trained Model Files

- `gmf_best.pth`: Trained GMF model weights for pre-training NeuMF  
- `mlp_best.pth`: Trained MLP model weights for pre-training NeuMF  

These files are automatically loaded in `pre_train.py` when comparing pre-trained vs. non-pretrained NeuMF models.

## 🚀 Running the Code

1. **Download** the [MovieLens 1M dataset](https://grouplens.org/datasets/movielens/1m/).
2. **Place** the `ratings.dat` file into the root directory of this project.
3. **Run** any of the executable scripts below to train models and output evaluation results.

## 🧪 Runnable Experiment Scripts

Each of the following scripts runs a complete experiment and prints performance metrics upon completion:

- `factors.py` — Predictive factor ablation  
- `layers.py` — Hidden layer depth ablation  
- `model.py` — GMF vs. MLP vs. NeuMF  
- `negsample.py` — Negative sampling analysis  
- `pre_train.py` — Pre-training vs. no pre-training (loads `gmf_best.pth` and `mlp_best.pth`)  
- `threshold.py` — Threshold setting comparison

## 📊 Evaluation Metrics

- **Recall@10**: Measures how many relevant items are successfully recommended.
- **NDCG@10**: Evaluates the ranking quality of recommended items.
- **F1 Score**: Used for comparing label quality under different threshold settings.

## 📌 Notes

- All models use Kaiming initialization and a learning rate scheduler for better convergence.
- Training curves are smoothed for visualization; original data points are also plotted.
- Evaluation results are printed after training; you may redirect outputs as needed.

## 📎 Reference

> He, X., Liao, L., Zhang, H., Nie, L., Hu, X., & Chua, T.S. (2017). Neural Collaborative Filtering. [arXiv:1708.05031](https://arxiv.org/abs/1708.05031)
