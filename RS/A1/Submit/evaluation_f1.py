import numpy as np
import torch

def calculate_ndcg(pos_movies, top_k_items, K):
    K = min(K, len(top_k_items)) 

    # Compute DCG
    dcg = sum(1 / np.log2(i + 2) for i, item in enumerate(top_k_items[:K]) if item in pos_movies)

    # Compute IDCG 
    ideal_hits = min(K, len(pos_movies)) 
    idcg = sum(1 / np.log2(i + 2) for i in range(ideal_hits))

    return dcg / idcg if idcg > 0 else 0


def model_evaluation_metric(model, val_dict, device, K=10, batch_size=1024):
    model.to(device)
    model.eval()
    user_input = []
    movie_input = []
    labels = []

    for u, interactions in val_dict.items():
        for movie_id, label in interactions:
            user_input.append(u)
            movie_input.append(movie_id)
            labels.append(label)

    user_input = torch.tensor(user_input, dtype=torch.long, device=device)
    movie_input = torch.tensor(movie_input, dtype=torch.long, device=device)
    
    predictions = []
    with torch.no_grad():
        for i in range(0, len(user_input), batch_size):  
            batch_users = user_input[i:i+batch_size]
            batch_movies = movie_input[i:i+batch_size]
            batch_preds = model(batch_users, batch_movies).squeeze(-1).cpu().numpy()
            predictions.extend(batch_preds)
    
    predictions_dict = {}
    for u, m, score in zip(user_input.cpu().tolist(), movie_input.cpu().tolist(), predictions):
        if u not in predictions_dict:
            predictions_dict[u] = {}
        predictions_dict[u][m] = score
    
    recall_list, ndcg_list, precision_list = [], [], [] 
    for u, interactions in val_dict.items():
        pos_movies = {m for m, label in interactions if label == 1}
        if not pos_movies or u not in predictions_dict:
            continue
        
        pred_scores = predictions_dict[u]
        top_k_items = np.array(sorted(pred_scores.keys(), key=lambda x: pred_scores[x], reverse=True))[:K]
        
        # Calculate recall@10
        relevant_in_top_k = sum(1 for movie_id in top_k_items if movie_id in pos_movies)
        recall_at_10 = relevant_in_top_k / len(pos_movies)
        recall_list.append(recall_at_10)

        # Calculate precision@10
        precision_at_10 = relevant_in_top_k / K
        precision_list.append(precision_at_10)

        # Calculate ndcg@10
        ndcg_at_10 = calculate_ndcg(pos_movies, top_k_items, K)
        ndcg_list.append(ndcg_at_10)
    
    avg_recall_at_10 = np.mean(recall_list) if recall_list else 0
    avg_ndcg_at_10 = np.mean(ndcg_list) if ndcg_list else 0
    avg_precision_at_10 = np.mean(precision_list) if precision_list else 0 

    torch.cuda.empty_cache() 

    return avg_recall_at_10, avg_ndcg_at_10, avg_precision_at_10