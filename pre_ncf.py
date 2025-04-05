import torch
from NeuMF import NeuMF
import os
from data_load_perpos import simple_load_data_rate, get_model_data
from evaluation import model_evaluation_metric
from collections import defaultdict
import pandas as pd

# Define the parameters used when the model was trained
layer = [128, 64]
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

# Load the pre-trained best model weights
state_dict = torch.load('./best_model_ncf_15.pth')

# Create a new state_dict without the '_orig_mod.' prefix
new_state_dict = {}
for key, value in state_dict.items():
    new_key = key.replace('_orig_mod.', '')  # Remove the prefix
    new_state_dict[new_key] = value

# Now load the new state_dict into your model
ncf_model.load_state_dict(new_state_dict)

# Evaluate the model before training (optional)
metrics = defaultdict(list)

with torch.no_grad():
    test_recall, test_ndcg = model_evaluation_metric(ncf_model, test_dict, device, K=10)
    metrics['test_recall@10'] = [test_recall] * len(metrics['epoch'])
    metrics['test_ndcg@10'] = [test_ndcg] * len(metrics['epoch'])
    df_metrics = pd.DataFrame(metrics)
    # df_metrics.to_csv(f'./pre_after_loading_test.csv', index=False)
    print(f"Test Results for pre-loading: Recall@10: {test_recall:.4f}, NDCG@10: {test_ndcg:.4f}")