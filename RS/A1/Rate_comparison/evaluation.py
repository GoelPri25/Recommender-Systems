import numpy as np
import torch

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

    precision_list = []
    recall_list = []

    for u, interactions in val_dict.items():
        pos_movies = {m for m, label in interactions if label == 1}
        if not pos_movies:
            continue

        if u not in predictions_dict:
            continue
        pred_scores = predictions_dict[u]

        top_k_items = np.array(sorted(pred_scores.keys(), key=lambda x: pred_scores[x], reverse=True))[:K]

        # Calculate Precision@10
        relevant_in_top_k = sum(1 for movie_id in top_k_items if movie_id in pos_movies)
        precision_at_10 = relevant_in_top_k / K
        precision_list.append(precision_at_10)

        # Calculate Recall@10
        recall_at_10 = relevant_in_top_k / len(pos_movies)
        recall_list.append(recall_at_10)

    # Calculate average Precision@10 and Recall@10
    avg_precision_at_10 = np.mean(precision_list) if precision_list else 0
    avg_recall_at_10 = np.mean(recall_list) if recall_list else 0

    # Calculate F1@10
    if avg_precision_at_10 + avg_recall_at_10 > 0:
        f1_at_10 = 2 * (avg_precision_at_10 * avg_recall_at_10) / (avg_precision_at_10 + avg_recall_at_10)
    else:
        f1_at_10 = 0

    return avg_precision_at_10, avg_recall_at_10, f1_at_10


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