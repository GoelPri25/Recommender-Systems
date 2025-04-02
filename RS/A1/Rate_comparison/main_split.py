import torch
import torch.nn as nn
from load_data import load_data_ignore3, load_data_rate, load_data_meanStd, load_data_rate_np
from load_data import load_data_rate_np, load_data_user_split, load_data_time_split
from data_process import get_non_interacted_movies, get_train_data, get_part_noninteract_validation, get_all_noninteract_test
from NeuMF import NeuMF
import torch.optim as optim
from collections import defaultdict
from evaluation import model_evaluation, model_evaluation_metric
import csv

loaders = {
    "pos_neg_split": lambda: load_data_rate_np('ratings.dat', threshold=3),
    "random_split": lambda: load_data_rate('ratings.dat', threshold=3),
    "timestamp_split": lambda: load_data_time_split('ratings.dat', threshold=3),
    "user_split": lambda: load_data_user_split('ratings.dat', threshold=3),
}

results = {}

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
negative_num = 1
batch_size = 32
num_epochs = 50
patience = 30
latent_dim = 8
layer = [16, 8]

# Open a CSV file for writing
with open('training_results.csv', mode='w', newline='') as file:
    writer = csv.writer(file)
    # Write the header row
    writer.writerow(["Dataset", "Epoch", "Train Loss", "Val Loss", "Recall@10", "NDCG@10"])

    for name, loader in loaders.items():
        print(f"\n=== Training with {name} ===\n")
        
        train_dict, val_dict, test_dict, movie_num, user_num = loader()
        non_interacted_movies = get_non_interacted_movies(train_dict, val_dict, test_dict, movie_num)

        train_user_input, train_movie_input, train_labels, neg_sample_train = get_train_data(train_dict, 
                                                                                             non_interacted_movies, negative_num)

        val_user_input, val_movie_input, val_labels, neg_sample_val = get_part_noninteract_validation(val_dict, 
                                                                                                      non_interacted_movies, 
                                                                                                      neg_sample_train, pos_num=5, neg_num=500)
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

        model = NeuMF(user_num+1, movie_num+1, latent_dim, layer).to(device)
        optimizer = optim.Adam(model.parameters(), lr=0.0001, weight_decay=1e-5)
        criterion = nn.BCEWithLogitsLoss()
        scaler = torch.amp.GradScaler('cuda')

        train_losses, val_losses, recalls, ndcgs = [], [], [], []
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

            recall, ndcg = model_evaluation_metric(model, val_dict, device, K=10)

            print(f"Epoch {epoch+1}, Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}, Recall@10: {recall:.4f}, NDCG@10: {ndcg:.4f}")

            train_losses.append(train_loss)
            val_losses.append(val_loss)
            recalls.append(recall)
            ndcgs.append(ndcg)

            # Write to CSV file after each epoch
            writer.writerow([name, epoch + 1, train_loss, val_loss, recall, ndcg])

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
            test_recall, test_ndcg = model_evaluation_metric(model, test_dict, device, K=10)
        print(f"\n=== Test Recall@10: {test_recall:.4f}, Test NDCG@10: {test_ndcg:.4f} ===\n")

        results[name] = {
            "train_losses": train_losses,
            "val_losses": val_losses,
            "recall_scores": recalls,
            "ndcg_scores": ndcgs,
            "test_recall": test_recall,
            "test_ndcg": test_ndcg
        }

print("All results saved to training_results.csv")