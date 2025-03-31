import torch
import torch.nn as nn
from load_data import load_data_ignore3, load_data_rate, load_data_meanStd, load_data_rate_np
from load_data import load_data_rate_np, load_data_user_split, load_data_time_split
from data_process import get_non_interacted_movies, get_train_data, get_part_noninteract_validation, get_all_noninteract_test
from NeuMF import NeuMF
import torch.optim as optim
from collections import defaultdict
from evaluation import model_evaluation

loaders = {
    "timestamp_split": lambda: load_data_time_split('ratings.dat', threshold=3),
    "random_split": lambda: load_data_rate('ratings.dat', threshold=3),
    "pos_neg_split": lambda: load_data_rate_np('ratings.dat', threshold=3),
    "user_split": lambda: load_data_user_split('ratings.dat', threshold=3),
}

results = {}

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
negative_num = 10
batch_size = 128
num_epochs = 30
patience = 10

for name, loader in loaders.items():
    print(f"\n=== Training with {name} ===\n")
    
    train_dict, val_dict, test_dict, movie_num, user_num = loader()
    non_interacted_movies = get_non_interacted_movies(train_dict, val_dict, test_dict, movie_num)

    train_user_input, train_movie_input, train_labels, neg_sample_train = get_train_data(train_dict, 
                                                                                         non_interacted_movies, negative_num)

    val_user_input, val_movie_input, val_labels, neg_sample_val = get_part_noninteract_validation(val_dict, 
                                                                                                  non_interacted_movies, 
                                                                                                  neg_sample_train, pos_num=5, neg_num=499)
    val_dict = defaultdict(list)
    for user_id, movie_id, label in zip(val_user_input, val_movie_input, val_labels):
        val_dict[user_id].append((movie_id, label))
    val_dict = dict(val_dict)

    user_input = torch.tensor(train_user_input, dtype=torch.long).to(device)
    movie_input = torch.tensor(train_movie_input, dtype=torch.long).to(device)
    labels = torch.tensor(train_labels, dtype=torch.float32).to(device)

    dataset = torch.utils.data.TensorDataset(user_input, movie_input, labels)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True, drop_last=True)

    val_user_input = torch.tensor(val_user_input, dtype=torch.long).to(device)
    val_movie_input = torch.tensor(val_movie_input, dtype=torch.long).to(device)
    val_labels = torch.tensor(val_labels, dtype=torch.float32).to(device)

    val_dataset = torch.utils.data.TensorDataset(val_user_input, val_movie_input, val_labels)
    val_dataloader = torch.utils.data.DataLoader(val_dataset, batch_size=batch_size, shuffle=True, drop_last=True)

    model = NeuMF(user_num+1, movie_num+1, 10, [10, 16]).to(device)
    optimizer = optim.Adam(model.parameters(), lr=0.0001, weight_decay=1e-5)
    criterion = nn.BCEWithLogitsLoss()
    scaler = torch.amp.GradScaler('cuda')

    train_losses, val_losses, f1s = [], [], []
    best_val_loss = float('inf')
    counter = 0

    for epoch in range(num_epochs):
        model.train()
        total_loss = 0
        for batch_users, batch_items, batch_labels in dataloader:
            batch_users, batch_items, batch_labels = batch_users.to(device), batch_items.to(device), batch_labels.to(device)
            optimizer.zero_grad(set_to_none=True)
            with torch.amp.autocast('cuda'):
                predictions = model(batch_users, batch_items)
                loss = criterion(predictions, batch_labels.view(-1, 1))
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            total_loss += loss.item()
        
        train_loss = total_loss / len(dataloader)

        model.eval()
        val_loss = 0
        with torch.no_grad():
            for batch_users, batch_items, batch_labels in val_dataloader:
                batch_users, batch_items, batch_labels = batch_users.to(device), batch_items.to(device), batch_labels.to(device)
                predictions = model(batch_users, batch_items)
                loss = criterion(predictions, batch_labels.view(-1, 1))
                val_loss += loss.item()
        val_loss /= len(val_dataloader)

        _, _, f1 = model_evaluation(model, val_dict, device, K=10)

        print(f"Epoch {epoch+1}, Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}, F1: {f1:.4f}")

        train_losses.append(train_loss)
        val_losses.append(val_loss)
        f1s.append(f1)

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            counter = 0
            torch.save(model.state_dict(), f"./best_model_{name}.pth")
        else:
            counter += 1
            print(f"Early Stopping Counter: {counter}/{patience}")
            if counter >= patience:
                print("Early stopping triggered! Stopping training.")
                break

    test_user_input, test_movie_input, test_labels = get_all_noninteract_test(test_dict, 
                                                                              non_interacted_movies, 
                                                                              neg_sample_train, neg_sample_val)
    test_dict = defaultdict(list)
    for user_id, movie_id, label in zip(test_user_input, test_movie_input, test_labels):
        test_dict[user_id].append((movie_id, label))
    test_dict = dict(test_dict)

    with torch.no_grad():
        _, _, test_f1 = model_evaluation(model, test_dict, device, K=10)
    print(f"\n=== Test F1 Score for {name}: {test_f1:.4f} ===\n")

    results[name] = {
        "train_losses": train_losses,
        "val_losses": val_losses,
        "f1_scores": f1s,
        "test_f1": test_f1
    }

import json
with open("training_results.json", "w") as f:
    json.dump(results, f, indent=4)

print("All results saved to training_results.json")