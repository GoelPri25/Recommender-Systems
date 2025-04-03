import random
import torch
import torch.multiprocessing as mp
mp.set_start_method('spawn', force=True)
import torch._dynamo
torch._dynamo.config.suppress_errors = True

random.seed(1000) # to get same samples shuffled to have consistent training results


def simple_load_data_rate(filename, negative_sample_no_train=1, negative_sample_no_valid=100, threshold=3, train_ratio=0.7, test_ratio=0.15):
    """
    Load dataset and split data on a per-user basis, ensuring:
    - Validation has at least one positive sample
    - Users with fewer than 5 positives are removed
    - Users with no enough negative samples are removed
    - Returns information about removed users
    
    Args:
        filename (str): Path to the ratings file.
        negative_sample_no_train (int): Number of negative samples for each positive sample in training.
        negative_sample_no_valid (int): Number of negative samples for validation.
        threshold (int): Rating threshold for positive samples.
        train_ratio (float): Percentage of interactions used for training.
        test_ratio (float): Percentage of interactions used for testing.

    Returns:
        train_dict, val_dict, test_dict, movie_num, user_num, removed_users_info
    """
    user_ratings = {}
    movie_num, user_num = -1, -1
    removed_users_info = {'total_removed': 0, 'removed_user_ids': []}

    # Read file and collect interactions
    with open(filename, "r", encoding="utf-8") as file:
        for line in file:
            user_id, movie_id, rating, _ = map(int, line.strip().split("::"))
            label = 1 if rating >= threshold else 0
            user_ratings.setdefault(user_id, []).append((movie_id, label))
            movie_num, user_num = max(movie_num, movie_id), max(user_num, user_id)
    
    all_movies = set(range(1, movie_num + 1))
    train_dict, val_dict, test_dict = {}, {}, {}
    
    # remove < 5
    for user_id, interactions in user_ratings.items():
        positives = [(m, l) for m, l in interactions if l == 1]
        if len(positives) < 5:
            removed_users_info['total_removed'] += 1
            removed_users_info['removed_user_ids'].append(user_id)
            continue
        
        # negtively interacted movies
        negatives = [(m, 0) for m, l in interactions if l == 0]
        interacted_movies = {m for m, _ in interactions}

        # non-interacted
        non_interacted = list(all_movies - interacted_movies)

        if not non_interacted:
            removed_users_info['total_removed'] += 1
            removed_users_info['removed_user_ids'].append(user_id)
            continue

        random.shuffle(non_interacted)

        # Train negative samples: For each positive sample, sample negative samples from non-interacted movies
        train_neg = []
        for movie_id, label in positives:
            neg_samples = random.sample(non_interacted, negative_sample_no_train)
            train_neg.extend([(m, 0) for m in neg_samples])

        val_neg = negatives[:negative_sample_no_valid] 
        remaining_needed = negative_sample_no_valid - len(val_neg)

        if remaining_needed > 0: 
            additional_neg = [(m, 0) for m in non_interacted[negative_sample_no_train:negative_sample_no_train + remaining_needed]]
            val_neg += additional_neg

        if not val_neg or len(val_neg) < negative_sample_no_valid:
            removed_users_info['total_removed'] += 1
            removed_users_info['removed_user_ids'].append(user_id)
            continue

        # Shuffle and split positives
        random.shuffle(positives)
        total_pos = len(positives)
        train_pos_end = int(train_ratio * total_pos)
        val_pos_end = train_pos_end + int(test_ratio * total_pos)
        train_pos, val_pos, test_pos = positives[:train_pos_end], positives[train_pos_end:val_pos_end] or positives[:1], positives[val_pos_end:]
        
        # Shuffle and split negatives
        random.shuffle(negatives)
        total_neg = len(negatives)
        train_neg_end = int(train_ratio * total_neg)
        val_neg_end = train_neg_end + int(test_ratio * total_neg)
        
        test_neg = negatives[val_neg_end:]
        test_neg += [(m, 0) for m in non_interacted[negative_sample_no_train + remaining_needed:]]
        
        # Merge and shuffle
        train_dict[user_id] = train_pos + train_neg
        val_dict[user_id] = val_pos[:5] + val_neg
        test_dict[user_id] = test_pos + test_neg
        
        random.shuffle(train_dict[user_id])
        random.shuffle(val_dict[user_id])
        random.shuffle(test_dict[user_id])
    
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
    file_name = "ratings.dat"

    
    #train_dict, val_dict, test_dict, non_interacted_movies, movie_num, user_num = load_data_rate(file_name)
    
    train_dict, valid_dict, test_dict, movie_num, user_num, removed_users_info= simple_load_data_rate(file_name)
    
    train_user_input, train_movie_input, train_labels = get_model_data(train_dict)
    valid_user_input, valid_movie_input, valid_labels = get_model_data(valid_dict)
    test_user_input, test_movie_input, test_labels = get_model_data(test_dict)
    
    print(len(train_user_input), len(train_movie_input), len(train_labels ))
    print(len(valid_user_input), len(valid_movie_input), len(valid_labels ))
    print(len(test_user_input), len(test_movie_input), len(test_labels ))
    
    print(removed_users_info)