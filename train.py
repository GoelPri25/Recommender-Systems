import torch
import os
from neuMF import NeuMF
from dataset_updated import simple_load_data_rate, get_model_data
import random
import torch
import torch.nn as nn
import torch.multiprocessing as mp
mp.set_start_method('spawn', force=True)
# import data as data
# import model_evaluation as evaluation
import torch.optim as optim
import torch._dynamo
import numpy as np
import heapq
from torch.optim.lr_scheduler import ReduceLROnPlateau
from evaluation import model_evaluation
import pandas as pd
from collections import defaultdict
torch._dynamo.config.suppress_errors = True

random.seed(1000)


#file_name = "/Users/priyanjaligoel/Documents/Recommender_systems/ratings.dat"


base_dir = os.getcwd()
name_rating_dir = "ml-1m/ratings.dat"
rating_data_file = os.path.join(base_dir, name_rating_dir)

#train_dict, val_dict, test_dict, non_interacted_movies, movie_num, user_num = load_data_rate(file_name)

train_dict, valid_dict, test_dict, movie_num, user_num, removed_users_info= simple_load_data_rate(rating_data_file, threshold=3)

train_user_input, train_movie_input, train_labels = get_model_data(train_dict)
valid_user_input, valid_movie_input, valid_labels = get_model_data(valid_dict)
test_user_input, test_movie_input, test_labels = get_model_data(test_dict)

print(len(train_user_input), len(train_movie_input), len(train_labels ))
print(len(valid_user_input), len(valid_movie_input), len(valid_labels ))
print(len(test_user_input), len(test_movie_input), len(test_labels ))

print(removed_users_info)

train_losses_ncf = []
val_losses_ncf = []
recalls_ncf = []
ndcgs_ncf = []
# f1s_rate3 = []
# train_recalls_rate3 = []
# train_ndcgs_rate3 = []
patience = 10
counter = 0
best_val_loss = float('inf')
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")



batch_size = 64
num_epochs = 30
model_ncf = NeuMF(
    num_users=user_num + 1,  # +1 for 0-based indexing
    num_items=movie_num + 1,
    mf_dim=8,       # Larger embeddings (vs. 10) for richer latent features
    layers=[32, 16, 8], # Deeper MLP for nonlinear interactions
            
).to(device)

optimizer = optim.Adam(
    model_ncf.parameters(),
    lr=0.001,          # Higher initial LR for faster convergence (MovieLens is dense)
    weight_decay=1e-5,  # Mild L2 regularization to prevent overfitting
    betas=(0.9, 0.999) # Default momentum terms
)
scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.7, patience=5, verbose=True)


criterion = nn.BCEWithLogitsLoss()
scaler = torch.cuda.amp.GradScaler()

user_input = torch.tensor(train_user_input, dtype=torch.long).to(device)
movie_input = torch.tensor(train_movie_input, dtype=torch.long).to(device)
train_labels = torch.tensor(train_labels, dtype=torch.float32).to(device)

dataset = torch.utils.data.TensorDataset(user_input, movie_input, train_labels)
dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size)


val_user_input = torch.tensor(valid_user_input, dtype=torch.long).to(device)
val_movie_input = torch.tensor(valid_movie_input, dtype=torch.long).to(device)
val_labels = torch.tensor(valid_labels, dtype=torch.float32).to(device)

val_dataset = torch.utils.data.TensorDataset(val_user_input, val_movie_input, val_labels)
val_dataloader = torch.utils.data.DataLoader(val_dataset, batch_size=batch_size)
model_ncf = torch.compile(model_ncf)  # Optimize model (PyTorch 2.0+)
model_ncf = torch.nn.DataParallel(model_ncf)  # Enable Multi-GPU
metrics = defaultdict(list)  # Tracks all metrics across epochs

for epoch in range(num_epochs):
    model_ncf.train()
    total_loss = 0

    for batch_users, batch_items, batch_labels in dataloader:
        batch_users = batch_users.to(device)
        batch_items = batch_items.to(device)
        batch_labels = batch_labels.to(device)
        optimizer.zero_grad(set_to_none=True)
        with torch.amp.autocast('cuda'):
            predictions = model_ncf(batch_users, batch_items)
            loss = criterion(predictions, batch_labels.view(-1, 1))
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        total_loss += loss.item()

    print(f"Epoch {epoch + 1}, Training Loss: {total_loss / len(dataloader)}")

    model_ncf.eval()
    val_loss = 0
   
    
    with torch.inference_mode():  # Best for inference
        with torch.cuda.amp.autocast():  # Mixed precision
            for batch_users, batch_items, batch_labels in val_dataloader:
                batch_users = batch_users.to(device, non_blocking=True)
                batch_items = batch_items.to(device, non_blocking=True)
                batch_labels = batch_labels.to(device, non_blocking=True)
    
                predictions = model_ncf(batch_users, batch_items)  # Optimized inference

   
                loss = criterion(predictions, batch_labels.view(-1, 1))
                val_loss += loss.item()
            val_loss_avg = val_loss / len(val_dataloader)
            scheduler.step(val_loss_avg)
            print(f"Epoch {epoch + 1}, Validation Loss: {val_loss_avg}")
    
    
    with torch.no_grad():
        recall, ndcg = model_evaluation(model_ncf, valid_dict, device, K=10)
        # train_recall, train_ndcg = model_evaluation(model_3, train_dict, device, K=10)
        recalls_ncf.append(recall)
        ndcgs_ncf.append(ndcg)
        # f1s_rate3.append(f1)
        # early stop
        
        metrics['epoch'].append(epoch + 1)
        metrics['train_loss'].append(total_loss / len(dataloader))
        metrics['val_loss'].append(val_loss_avg)
        metrics['recall@10'].append(recall)
        metrics['ndcg@10'].append(ndcg)
        metrics['lr'].append(optimizer.param_groups[0]['lr'])  # Log learning rate
        df_metrics = pd.DataFrame(metrics)
        df_metrics.to_csv('./training_metrics.csv', index=False)
        
        
        tolerance = 0.001  # Allow minor fluctuations
        if val_loss_avg < (best_val_loss - tolerance):
            best_val_loss = val_loss_avg
            counter = 0
            
        else:
            counter += 1
            if counter >= patience:
                print("Early stopping: Loss stagnated.")
                torch.save(model_ncf.state_dict(), "./best_model_ncf.pth")    
                break
            