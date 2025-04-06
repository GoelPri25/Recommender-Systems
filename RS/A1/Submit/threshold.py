import torch
import os
from NeuMF import NeuMF
from data_load import simple_load_data_rate, get_model_data

import random
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import ReduceLROnPlateau
from evaluation_f1 import model_evaluation_metric
import pandas as pd
from collections import defaultdict
torch._dynamo.config.suppress_errors = True

random.seed(1000)

base_dir = os.getcwd()
name_rating_dir = "ratings.dat"
rating_data_file = os.path.join(base_dir, name_rating_dir)
best_model = None

for threshold_value in [3, 4]:
    print(f"\nRunning with threshold = {threshold_value}")

    train_dict, valid_dict, test_dict, movie_num, user_num, removed_users_info, _ = simple_load_data_rate(
        rating_data_file, negative_sample_no_train=1, negative_sample_no_valid=100, threshold=threshold_value)

    train_user_input, train_movie_input, train_labels = get_model_data(train_dict)
    valid_user_input, valid_movie_input, valid_labels = get_model_data(valid_dict)
    test_user_input, test_movie_input, test_labels = get_model_data(test_dict)

    print(len(train_user_input), len(train_movie_input), len(train_labels))
    print(len(valid_user_input), len(valid_movie_input), len(valid_labels))
    print(len(test_user_input), len(test_movie_input), len(test_labels))

    print(removed_users_info)

    train_losses_ncf = []
    val_losses_ncf = []
    recalls_ncf = []
    ndcgs_ncf = []

    patience = 20
    counter = 0
    best_val_loss = float('inf')

    results = {}
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    batch_size = 32
    num_epochs = 50
    model_ncf = NeuMF(
        num_users=user_num + 1, 
        num_items=movie_num + 1,
        mf_dim=8,       
        layers=[16, 8],
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
            recall, ndcg, precision = model_evaluation_metric(model_ncf, valid_dict, device, K=10)

            if precision + recall > 0:
                f1_at_10 = 2 * (precision * recall) / (precision + recall)
            else:
                f1_at_10 = 0.0 

            metrics['epoch'].append(epoch + 1)
            metrics['train_loss'].append(total_loss / len(dataloader))
            metrics['val_loss'].append(val_loss_avg)
            metrics['f1@10'].append(f1_at_10)  
            metrics['lr'].append(optimizer.param_groups[0]['lr']) 

            df_metrics = pd.DataFrame(metrics)
            df_metrics.to_csv(f'./1_rating_threshold_{threshold_value}.csv', index=False)

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
                    torch.save(best_model, f"./1_rating_threshold_{threshold_value}.pth") 
                    break

    # Test phase
    test_dict = defaultdict(list)

    for user_id, movie_id, label in zip(test_user_input, test_movie_input, test_labels):
        test_dict[user_id].append((movie_id, label))
    test_dict = dict(test_dict)

    with torch.no_grad():
        test_recall, test_ndcg, test_precision = model_evaluation_metric(best_model, test_dict, device, K=10)

        if test_precision + test_recall > 0:
            test_f1_at_10 = 2 * (test_precision * test_recall) / (test_precision + test_recall)
        else:
            test_f1_at_10 = 0.0

        print(f"\n=== Test F1@10: {test_f1_at_10:.4f} ===\n")

        metrics['test_f1@10'] = [test_f1_at_10] * len(metrics['epoch']) 

        df_metrics = pd.DataFrame(metrics)
        df_metrics.to_csv(f'./1_rating_threshold_test_{threshold_value}.csv', index=False)

        results[f'threshold_{threshold_value}'] = {
            "test_f1": test_f1_at_10
        }
        print(f"Test Results for threshold {threshold_value}: {results[f'threshold_{threshold_value}']}")