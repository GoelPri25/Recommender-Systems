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

torch._dynamo.config.suppress_errors = True

random.seed(1000) # to get same samples shuffled to have consistent training results


# (user_id: [(movie_id, label)])


# (user_id: [(movie_id, label)])
import random

def simple_load_data_rate(filename, negative_sample_no=10, threshold=4, train_ratio=0.7, test_ratio=0.15):
    """
    Load dataset and split data on a per-user basis, ensuring:
    - Validation has at least one positive sample
    - Users with no positives are removed
    - Returns information about removed users
    
    Args:
        filename (str): Path to the ratings file.
        train_ratio (float): Percentage of interactions used for training.
        test_ratio (float): Percentage of interactions used for testing.

    Returns:
        train_dict, val_dict, test_dict, movie_num, user_num, removed_users_info
    """
    # Initialize data structures
    user_ratings = {}
    movie_num = -1
    user_num = -1
    removed_users_info = {
        'total_removed': 0,
        'removed_user_ids': []
    }

    # First pass: collect all ratings
    with open(filename, "r", encoding="utf-8") as file:
        for line in file:
            user_id, movie_id, rating, _ = map(int, line.strip().split("::"))
            label = 1 if rating >= threshold else 0

            if user_id not in user_ratings:
                user_ratings[user_id] = []
            user_ratings[user_id].append((movie_id, label))

            # Update movie and user counts
            movie_num = max(movie_num, movie_id)
            user_num = max(user_num, user_id)

    # Add negative samples and filter users with no positives
    all_movies = set(range(1, movie_num + 1))
    filtered_user_ratings = {}
    
    for user_id, interactions in user_ratings.items():
        # Check if user has any positive interactions
        positives = [x for x in interactions if x[1] == 1]
        
        if not positives:
            removed_users_info['total_removed'] += 1
            removed_users_info['removed_user_ids'].append(user_id)
            continue  # Skip this user
        
        # Only proceed for users with positive interactions
        interacted_movies = {movie_id for movie_id, _ in interactions}
        non_interacted = list(all_movies - interacted_movies)
        
        # Sample negative examples
        if len(non_interacted) >= negative_sample_no:
            negatives = random.sample(non_interacted, negative_sample_no)
        else:
            negatives = non_interacted
            
        # Add negative samples to user's interactions
        filtered_interactions = interactions + [(movie_id, 0) for movie_id in negatives]
        filtered_user_ratings[user_id] = filtered_interactions

    # Now split the filtered users
    train_dict, val_dict, test_dict = {}, {}, {}

    for user_id, interactions in filtered_user_ratings.items():
        # Separate positive and negative interactions
        positives = [x for x in interactions if x[1] == 1]
        negatives = [x for x in interactions if x[1] == 0]
        
        # Shuffle both lists
        random.shuffle(positives)
        random.shuffle(negatives)
        
        # Split positive samples (guarantee at least 1 in validation)
        total_pos = len(positives)
        train_pos_end = int(train_ratio * total_pos)
        val_pos_end = train_pos_end + max(1, int(test_ratio * total_pos))
        
        train_pos = positives[:train_pos_end]
        val_pos = positives[train_pos_end:val_pos_end]
        test_pos = positives[val_pos_end:]
        
        # Split negative samples proportionally
        total_neg = len(negatives)
        train_neg_end = int(train_ratio * total_neg)
        val_neg_end = train_neg_end + int(test_ratio * total_neg)
        
        train_neg = negatives[:train_neg_end]
        val_neg = negatives[train_neg_end:val_neg_end]
        test_neg = negatives[val_neg_end:]
        
        # Combine positive and negative samples
        train_dict[user_id] = train_pos + train_neg
        val_dict[user_id] = val_pos + val_neg
        test_dict[user_id] = test_pos + test_neg

    return train_dict, val_dict, test_dict, movie_num, user_num, removed_users_info




def get_model_data(train_dict):
    """
    For the training set, extract positive interactions (rating = 1) and negative samples
    (randomly sampled from non-interacted movies).

    Args:
        train_dict (dict): Training set data for users.
        non_interacted_movies (dict): Non-interacted movies for each user.
        negative_num (int): Number of negative samples to generate per user.

    Returns:
        user_input, movie_input, labels (Lists): Input data and labels (positive and negative samples).
        used_train_negatives (dict): A record of negative samples used for training for each user.
    """
    user_input, movie_input, labels = [], [], []
   
    for u, rate_list in train_dict.items():
        # Positive samples in train set
        for movie_id, label in rate_list:
            user_input.append(u)
            movie_input.append(movie_id)
            labels.append(label)
    return user_input, movie_input, labels


if __name__ == "__main__":
    file_name = "/Users/priyanjaligoel/Documents/Recommender_systems/ratings.dat"

    
    #train_dict, val_dict, test_dict, non_interacted_movies, movie_num, user_num = load_data_rate(file_name)
    
    train_dict, valid_dict, test_dict, movie_num, user_num, removed_users_info= simple_load_data_rate(file_name)
    
    train_user_input, train_movie_input, train_labels = get_model_data(train_dict)
    valid_user_input, valid_movie_input, valid_labels = get_model_data(valid_dict)
    test_user_input, test_movie_input, test_labels = get_model_data(test_dict)
    
    print(len(train_user_input), len(train_movie_input), len(train_labels ))
    print(len(valid_user_input), len(valid_movie_input), len(valid_labels ))
    print(len(test_user_input), len(test_movie_input), len(test_labels ))
    
    print(removed_users_info)
    