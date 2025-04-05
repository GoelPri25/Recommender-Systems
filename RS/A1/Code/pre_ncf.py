import torch
from NeuMF import NeuMF
import os
from data_load_perpos import simple_load_data_rate, get_model_data
from evaluation import model_evaluation_metric
from collections import defaultdict
import pandas as pd
import matplotlib.pyplot as plt

# Define the parameters used when the model was trained
layer = [16, 10]
predictive_factor = 10
base_dir = os.getcwd()
name_rating_dir = "ratings.dat"
rating_data_file = os.path.join(base_dir, name_rating_dir)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load the rating data and prepare it
train_dict, valid_dict, test_dict, movie_num, user_num, removed_users_info, _ = simple_load_data_rate(
    rating_data_file, negative_sample_no_train=5, negative_sample_no_valid=100, threshold=3
)

# Reload the model architecture
ncf_model = NeuMF(
    num_users=user_num + 1,  # +1 for 0-based indexing
    num_items=movie_num + 1,
    mf_dim=predictive_factor,
    layers=layer,
).to(device)
model_state_dict = ncf_model.state_dict()

# Load the pre-trained best model weights
pretrained_dict = torch.load('./ncf_best.pth')

# Create a new state_dict without the '_orig_mod.' prefix
pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_state_dict and v.size() == model_state_dict[k].size()}
model_state_dict.update(pretrained_dict)
ncf_model.load_state_dict(model_state_dict)

# Evaluate the model before training (optional)
metrics = defaultdict(list)

Ks = [1, 3, 5, 7, 10]
recalls = []
ndcgs = []

with torch.no_grad():
    for k in Ks:
        test_recall, test_ndcg = model_evaluation_metric(ncf_model, test_dict, device, K=k)
        recalls.append(test_recall)
        ndcgs.append(test_ndcg)
        print(f"K={k}: Recall@{k}: {test_recall:.4f}, NDCG@{k}: {test_ndcg:.4f}")


plt.figure(figsize=(8, 5))
plt.plot(Ks, recalls, marker='o', label='Recall@K')
plt.plot(Ks, ndcgs, marker='s', label='NDCG@K')
plt.title('NeuMF Performance with Pre-training', fontsize=14)
plt.xlabel('K', fontsize=12)
plt.ylabel('Score', fontsize=12)
plt.legend()
plt.grid(True, linestyle='--', alpha=0.7)
plt.tight_layout()
plt.show()