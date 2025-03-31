import torch
import torch.nn as nn
from load_data import load_data_ignore3, load_data_rate, load_data_meanStd
from data_process import get_non_interacted_movies, get_train_data, get_all_noninteract_validation, get_part_noninteract_validation, get_all_noninteract_test
from NeuMF import NeuMF
import torch.optim as optim
from collections import defaultdict
from evaluation import model_evaluation
train_losses = []
val_losses = []
train_recalls = []
train_ndcgs = []
recalls = []
ndcgs = []
f1s = []
patience = 10
counter = 0
best_val_loss = float('inf')
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
train_dict, val_dict, test_dict, movie_num, user_num = load_data_rate('ratings.dat', threshold=3)
negative_num = 10
non_interacted_movies = get_non_interacted_movies(train_dict, val_dict, test_dict, movie_num)

#user_input, movie_input, labels = get_input_data(train_dict, non_interacted_movies, negative_num)

train_user_input, train_movie_input, train_labels, neg_sample_train = get_train_data(train_dict, 
                                                                                     non_interacted_movies, negative_num)

# latent_dim = 8

batch_size = 128
num_epochs = 30
model = NeuMF(user_num+1, movie_num+1, 10, [10, 16]).to(device)
model = torch.compile(model)
optimizer = optim.Adam(model.parameters(), lr=0.0001, weight_decay=1e-5)
# scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.7, patience=3, verbose=True)

criterion = nn.BCEWithLogitsLoss()
scaler = torch.amp.GradScaler('cuda')

user_input = torch.tensor(train_user_input, dtype=torch.long).to(device)
movie_input = torch.tensor(train_movie_input, dtype=torch.long).to(device)
labels = torch.tensor(train_labels, dtype=torch.float32).to(device)

dataset = torch.utils.data.TensorDataset(user_input, movie_input, labels)
dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True, drop_last=True)


val_user_input, val_movie_input, val_labels, neg_sample_val = get_part_noninteract_validation(val_dict, non_interacted_movies, 
                                                                              neg_sample_train, pos_num=1, neg_num=99)

val_dict = defaultdict(list)

for user_id, movie_id, label in zip(val_user_input, val_movie_input, val_labels):
    val_dict[user_id].append((movie_id, label))

val_dict = dict(val_dict)

val_user_input = torch.tensor(val_user_input, dtype=torch.long).to(device)
val_movie_input = torch.tensor(val_movie_input, dtype=torch.long).to(device)
val_labels = torch.tensor(val_labels, dtype=torch.float32).to(device)

val_dataset = torch.utils.data.TensorDataset(val_user_input, val_movie_input, val_labels)
val_dataloader = torch.utils.data.DataLoader(val_dataset, batch_size=batch_size, shuffle=True, drop_last=True)

for epoch in range(num_epochs):
    model.train()
    total_loss = 0

    for batch_users, batch_items, batch_labels in dataloader:
        batch_users = batch_users.to(device)
        batch_items = batch_items.to(device)
        batch_labels = batch_labels.to(device)
        optimizer.zero_grad(set_to_none=True)
        with torch.amp.autocast('cuda'):
            predictions = model(batch_users, batch_items)
            loss = criterion(predictions, batch_labels.view(-1, 1))
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        total_loss += loss.item()

    print(f"Epoch {epoch + 1}, Training Loss: {total_loss / len(dataloader)}")

    model.eval()
    val_loss = 0
    num_samples = 0

    model.eval()
    with torch.no_grad():
        for batch_users, batch_items, batch_labels in val_dataloader:
            batch_users = batch_users.to(device)
            batch_items = batch_items.to(device)
            batch_labels = batch_labels.to(device)

            predictions = model(batch_users, batch_items)
            loss = criterion(predictions, batch_labels.view(-1, 1))  # BCE loss (averaged over batch)

            batch_size = batch_labels.size(0)
            val_loss += loss.item() * batch_size  # convert avg back to total loss for batch
            num_samples += batch_size

    val_loss_avg = val_loss / num_samples  # final average loss per sample
    # scheduler.step(val_loss_avg)

    print(f"Epoch {epoch + 1}, Validation Loss: {val_loss_avg:.4f}")

    with torch.no_grad():
        for batch_users, batch_items, batch_labels in val_dataloader:
            batch_users = batch_users.to(device)
            batch_items = batch_items.to(device)
            batch_labels = batch_labels.to(device)
            predictions = model(batch_users, batch_items)
            loss = criterion(predictions, batch_labels.view(-1, 1))
            val_loss += loss.item()
        val_loss_avg = val_loss / len(val_dataloader)
        # scheduler.step(val_loss_avg)
        print(f"Epoch {epoch + 1}, Validation Loss: {val_loss_avg}")
    train_losses.append(total_loss / len(dataloader))
    val_losses.append(val_loss_avg)

    # train_recall, train_ndcg = model_evaluation(model, train_dict, device, K=10)
    _, _, f1 = model_evaluation(model, val_dict, device, K=10)
    # train_recalls.append(train_recall)
    # train_ndcgs.append(train_ndcg)
    # recalls.append(recall)
    # ndcgs.append(ndcg)
    f1s.append(f1)

    # early stop
    if val_loss_avg < best_val_loss:
        best_val_loss = val_loss_avg
        counter = 0
        torch.save(model.state_dict(), "./best_model.pth")
    else:
        counter += 1
        print(f"Early Stopping Counter: {counter}/{patience}")
        if counter >= patience:
            print("Early stopping triggered! Stopping training.")
            break

print(len(f1))

test_user_input, test_movie_input, test_labels = get_all_noninteract_test(test_dict, non_interacted_movies, 
                                                                          neg_sample_train, neg_sample_val)
test_dict = defaultdict(list)

for user_id, movie_id, label in zip(test_user_input, test_movie_input, test_labels):
    test_dict[user_id].append((movie_id, label))

test_dict = dict(test_dict)
_, _, f1 = model_evaluation(model, test_dict, device, K=10)
print('test', f1)