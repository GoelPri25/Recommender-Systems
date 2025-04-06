import torch
import os
from NeuMF import NeuMF
from data_load import simple_load_data_rate, get_model_data
# from data_load_perpos import simple_load_data_rate, get_model_data

import random
import torch
import torch.nn as nn
import torch.multiprocessing as mp
mp.set_start_method('spawn', force=True)
import torch.optim as optim
import torch._dynamo
from torch.optim.lr_scheduler import ReduceLROnPlateau
from evaluation import model_evaluation_metric
import pandas as pd
from collections import defaultdict
torch._dynamo.config.suppress_errors = True

random.seed(1000)


#file_name = "/Users/priyanjaligoel/Documents/Recommender_systems/ratings.dat"


base_dir = os.getcwd()
name_rating_dir = "ratings.dat"
rating_data_file = os.path.join(base_dir, name_rating_dir)

layers = [[256, 128, 64, 32], [128, 64, 32], [64, 32], [32]]
results = {}
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
batch_size = 32
num_epochs = 50
for layer in layers:
    train_losses_ncf = []
    val_losses_ncf = []
    recalls_ncf = []
    ndcgs_ncf = []
    # f1s_rate3 = []
    # train_recalls_rate3 = []
    # train_ndcgs_rate3 = []
    patience = 20
    counter = 0
    best_val_loss = float('inf')
    print(f"Training with negative_sample_no_train={3}")
    
    train_dict, valid_dict, test_dict, movie_num, user_num, removed_users_info, _ = simple_load_data_rate(rating_data_file, negative_sample_no_train=1, negative_sample_no_valid=100, threshold=3)
    
    train_user_input, train_movie_input, train_labels = get_model_data(train_dict)
    valid_user_input, valid_movie_input, valid_labels = get_model_data(valid_dict)
    test_user_input, test_movie_input, test_labels = get_model_data(test_dict)

    model_ncf = NeuMF(
        num_users=user_num + 1,
        num_items=movie_num + 1,
        mf_dim=layer[-1],
        layers=layer
    ).to(device)

    optimizer = optim.Adam(
        model_ncf.parameters(),
        lr=0.001,
        weight_decay=1e-5,
        betas=(0.9, 0.999)
    )
    scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.2, patience=5, verbose=True)

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
    model_ncf = torch.compile(model_ncf)  
    model_ncf = torch.nn.DataParallel(model_ncf) 
    metrics = defaultdict(list)
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
        val_loss = 0.0
        with torch.inference_mode():
            with torch.cuda.amp.autocast():
                for batch_users, batch_items, batch_labels in val_dataloader:
                    batch_users = batch_users.to(device, non_blocking=True)
                    batch_items = batch_items.to(device, non_blocking=True)
                    batch_labels = batch_labels.to(device, non_blocking=True)

                    predictions = model_ncf(batch_users, batch_items)
                    loss = criterion(predictions, batch_labels.view(-1, 1))
                    val_loss += loss.item()
                val_loss_avg = val_loss / len(val_dataloader)
                scheduler.step(val_loss_avg) if len(val_dataloader) > 0 else 0.0
                print(f"Epoch {epoch + 1}, Validation Loss: {val_loss_avg}")
        
        with torch.no_grad():
            recall, ndcg = model_evaluation_metric(model_ncf, valid_dict, device, K=10)
            recalls_ncf.append(recall)
            ndcgs_ncf.append(ndcg)

        metrics['epoch'].append(epoch + 1)
        metrics['train_loss'].append(total_loss / len(dataloader))
        metrics['val_loss'].append(val_loss_avg)
        metrics['recall@10'].append(recall)
        metrics['ndcg@10'].append(ndcg)
        metrics['lr'].append(optimizer.param_groups[0]['lr'])

        df_metrics = pd.DataFrame(metrics)
        df_metrics.to_csv(f'./4_training_{layer[0]}.csv', index=False)

        tolerance = 0.001  
        if val_loss_avg < (best_val_loss - tolerance):
            best_val_loss = val_loss_avg
            counter = 0
            best_model = model_ncf
        else:
            counter += 1
            print(f"Early Stopping Counter: {counter}/{patience}")
            if counter >= patience:
                print("Early stopping: Loss stagnated.")
                torch.save(best_model, f"./4_training_{layer[0]}.pth") 
                break


    with torch.no_grad():
        test_recall, test_ndcg = model_evaluation_metric(best_model, test_dict, device, K=10)
        print(f"\n=== Test Recall@10: {test_recall:.4f}, Test NDCG@10: {test_ndcg:.4f} ===\n")
        
        metrics['test_recall@10'] = [test_recall] * len(metrics['epoch'])  
        metrics['test_ndcg@10'] = [test_ndcg] * len(metrics['epoch']) 
        
        df_metrics = pd.DataFrame(metrics)
        df_metrics.to_csv(f'./4_test_{layer[0]}.csv', index=False)

        results['name'] = {
            "test_recall": test_recall,
            "test_ndcg": test_ndcg
        }
        print(f"Test Results: {results['name']}")