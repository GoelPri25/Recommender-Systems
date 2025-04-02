import torch
import numpy as np
import random
def model_evaluation(model, val_dict, device, K=10):
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

    with torch.no_grad():
        predictions = model(user_input, movie_input).squeeze(-1).cpu().numpy()

    predictions_dict = {}
    for u, m, score in zip(user_input.cpu().tolist(), movie_input.cpu().tolist(), predictions):
        if u not in predictions_dict:
            predictions_dict[u] = {}
        predictions_dict[u][m] = score

    recall_list = []
    ndcg_list = []

    for u, interactions in val_dict.items():
        pos_movies = {m for m, label in interactions if label == 1}
        if not pos_movies:
            continue

        if u not in predictions_dict:
            continue
        pred_scores = predictions_dict[u]

        top_k_items = np.array(sorted(pred_scores.keys(), key=lambda x: pred_scores[x], reverse=True))[:K]

        recall = len(pos_movies.intersection(top_k_items)) / len(pos_movies)
        recall_list.append(recall)

        ndcg = calculate_ndcg(pos_movies, top_k_items, K)
        ndcg_list.append(ndcg)

    avg_recall = np.mean(recall_list) if recall_list else 0
    avg_ndcg = np.mean(ndcg_list) if ndcg_list else 0

    return avg_recall, avg_ndcg


def calculate_ndcg(pos_movies, top_k_items, K):
    """
    Calculate NDCG for the top-K recommended items.

    Args:
    - pos_movies: A set of relevant (ground truth) items for the user.
    - top_k_items: A list of the top-K recommended items.
    - K: The number of top items considered for evaluation.

    Returns:
    - NDCG score.
    """
    K = min(K, len(top_k_items))  # Adjust K to avoid overestimation

    # Compute DCG
    dcg = sum(1 / np.log2(i + 2) for i, item in enumerate(top_k_items[:K]) if item in pos_movies)

    # Compute IDCG (Ideal DCG)
    ideal_hits = min(K, len(pos_movies))  # Can't be more than positive items
    idcg = sum(1 / np.log2(i + 2) for i in range(ideal_hits))

    return dcg / idcg if idcg > 0 else 0

import math
def evaluate_ranking(model, val_dict, all_items, device, K=10, num_negatives=99):
    model.eval()
    hits, ndcgs = [], []

    for user, interactions in val_dict.items():
        # Get the first true positive item (no randomness)
        true_item = next((item for item, label in interactions if label == 1), None)
        if true_item is None:
            continue

        # Sample negatives not in user's val set
        user_items = set(item for item, _ in interactions)
        negatives = list(all_items - user_items)
        if len(negatives) < num_negatives:
            continue  # skip if not enough negatives

        sampled_negatives = random.sample(negatives, num_negatives)
        test_items = [true_item] + sampled_negatives

        user_tensor = torch.tensor([user] * len(test_items), dtype=torch.long).to(device)
        item_tensor = torch.tensor(test_items, dtype=torch.long).to(device)

        with torch.no_grad():
            scores = model(user_tensor, item_tensor).squeeze().cpu().numpy()

        ranked_items = [x for _, x in sorted(zip(scores, test_items), reverse=True)]

        # Hit@K
        hit = int(true_item in ranked_items[:K])
        hits.append(hit)

        # NDCG@K
        if true_item in ranked_items[:K]:
            rank = ranked_items.index(true_item)
            ndcg = 1 / math.log2(rank + 2)
        else:
            ndcg = 0
        ndcgs.append(ndcg)

    avg_hit = sum(hits) / len(hits) if hits else 0
    avg_ndcg = sum(ndcgs) / len(ndcgs) if ndcgs else 0

    return avg_hit, avg_ndcg

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
    
    recall_list, ndcg_list = [], []
    for u, interactions in val_dict.items():
        pos_movies = {m for m, label in interactions if label == 1}
        if not pos_movies or u not in predictions_dict:
            continue
        
        pred_scores = predictions_dict[u]
        top_k_items = np.array(sorted(pred_scores.keys(), key=lambda x: pred_scores[x], reverse=True))[:K]
        
        relevant_in_top_k = sum(1 for movie_id in top_k_items if movie_id in pos_movies)
        recall_at_10 = relevant_in_top_k / len(pos_movies)
        recall_list.append(recall_at_10)

        ndcg_at_10 = calculate_ndcg(pos_movies, top_k_items, K)
        ndcg_list.append(ndcg_at_10)
    
    avg_recall_at_10 = np.mean(recall_list) if recall_list else 0
    avg_ndcg_at_10 = np.mean(ndcg_list) if ndcg_list else 0

    torch.cuda.empty_cache() 

    return avg_recall_at_10, avg_ndcg_at_10