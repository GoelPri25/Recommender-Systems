import random
import torch
import torch.multiprocessing as mp
mp.set_start_method('spawn', force=True)
import torch._dynamo
torch._dynamo.config.suppress_errors = True

random.seed(1000) # to get same samples shuffled to have consistent training results


def simple_load_data_rate(filename, negative_sample_no_train=1, negative_sample_no_valid=100, threshold=4, filter=False, train_ratio=0.7, test_ratio=0.15):
    """
    Load dataset and split data on a per-user basis, ensuring:
    - Validation has at least one positive sample
    - Users with fewer than 5 positives are removed
    - Users with not enough negative samples are removed
    - Returns information about removed users
    
    Args:
        filename (str): Path to the ratings file.
        negative_sample_no_train (int): Number of negative samples for training.
        negative_sample_no_valid (int): Number of negative samples for validation.
        threshold (int): Rating threshold for positive samples.
        filter (bool): Whether to filter ratings so that only 4,5 are positive and 1,2 are negative.
        train_ratio (float): Percentage of interactions used for training.
        test_ratio (float): Percentage of interactions used for testing.

    Returns:
        train_dict, val_dict, test_dict, movie_num, user_num, removed_users_info, user_ratings
    """
    user_ratings = {}
    movie_num, user_num = -1, -1
    removed_users_info = {'total_removed': 0, 'removed_user_ids': []}

    # Read file and collect interactions
    with open(filename, "r", encoding="utf-8") as file:
        for line in file:
            user_id, movie_id, rating, _ = map(int, line.strip().split("::"))

            # Apply filtering condition
            if filter:
                if rating in [4, 5]:
                    label = 1
                elif rating in [1, 2]:
                    label = 0
                else:
                    continue  # Skip rating=3
            else:
                label = 1 if rating >= threshold else 0

            user_ratings.setdefault(user_id, []).append((movie_id, label))
            movie_num, user_num = max(movie_num, movie_id), max(user_num, user_id)
    
    all_movies = set(range(1, movie_num + 1))
    train_dict, val_dict, test_dict = {}, {}, {}
    
    # Remove users with fewer than 5 positive interactions
    for user_id, interactions in user_ratings.items():
        positives = [(m, l) for m, l in interactions if l == 1]
        if len(positives) < 5:
            removed_users_info['total_removed'] += 1
            removed_users_info['removed_user_ids'].append(user_id)
            continue
        
        # Negative interactions (rating == 0)
        negatives = [(m, 0) for m, l in interactions if l == 0]
        interacted_movies = {m for m, _ in interactions}
        # Non-interacted movies (to sample negative examples from)
        non_interacted = list(all_movies - interacted_movies)

        if not non_interacted:
            removed_users_info['total_removed'] += 1
            removed_users_info['removed_user_ids'].append(user_id)
            continue

        # Now we handle splitting the interactions into train, validation, and test sets
        total_samples = len(positives) + len(negatives)
        all_samples = positives + negatives
        
        # Shuffle all samples to avoid bias in splitting
        random.shuffle(all_samples)
        
        # Split into train, validation, and test
        train_end = int(train_ratio * total_samples)
        val_end = train_end + int(test_ratio * total_samples)
        
        train_set = all_samples[:train_end]
        val_set = all_samples[train_end:val_end]
        test_set = all_samples[val_end:]
        
        val_neg_existing = [(m, l) for m, l in val_set if l == 0]

        remaining_needed = max(0, negative_sample_no_valid - len(val_neg_existing))

        val_neg_additional = [(m, 0) for m in non_interacted[:remaining_needed]]

        val_neg = val_neg_existing + val_neg_additional
        # Add negative samples to training and validation sets
        random.shuffle(non_interacted)

        # **Changes here**: For each positive sample in training set, sample negative samples
        train_neg = []
        start_idx = remaining_needed
        for _, label in train_set:
            if label == 1:
                end_idx = start_idx + negative_sample_no_train
                neg_samples = [(m, 0) for m in non_interacted[start_idx:end_idx]]
                train_neg.extend(neg_samples)
                start_idx = end_idx
        test_neg = [(m, 0) for m in non_interacted[start_idx+negative_sample_no_valid:]]

        # Ensure we have at least 5 positive samples in the validation set
        if len(val_neg) < negative_sample_no_valid:
            removed_users_info['total_removed'] += 1
            removed_users_info['removed_user_ids'].append(user_id)
            continue

        # Add the negative samples to the sets
        train_dict[user_id] = train_set + train_neg
        val_set_filtered = [sample for sample in val_set if sample not in val_neg_existing]
        val_dict[user_id] = val_set_filtered + val_neg
        test_dict[user_id] = test_set + test_neg

        # Shuffle the sets to ensure randomness
        random.shuffle(train_dict[user_id])
        random.shuffle(val_dict[user_id])
        random.shuffle(test_dict[user_id])
    
    return train_dict, val_dict, test_dict, movie_num, user_num, removed_users_info, user_ratings
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


def analyze_data(user_ratings, movie_num, removed_users_info):
    """
    Analyze the dataset and return some key statistics.
    
    Args:
        user_ratings (dict): The user-item interaction data.
        movie_num (int): Total number of movies.
        removed_users_info (dict): Information about removed users.

    Returns:
        dict: Key statistics including sparsity, positive/negative interactions, and others.
    """
    total_interactions = sum(len(ratings) for ratings in user_ratings.values())
    total_users = len(user_ratings)
    total_items = movie_num
    sparsity = 1 - (total_interactions / (total_users * total_items))

    positive_interactions = sum(len([1 for _, label in ratings if label == 1]) for ratings in user_ratings.values())
    negative_interactions = total_interactions - positive_interactions

    user_interaction_counts = [len(ratings) for ratings in user_ratings.values()]
    item_interaction_counts = [sum(1 for ratings in user_ratings.values() if movie_id in [m for m, _ in ratings]) for movie_id in range(1, total_items + 1)]

    removed_users = removed_users_info['total_removed']

    # Print the key statistics
    print(f"Total interactions: {total_interactions}")
    print(f"Total users: {total_users}")
    print(f"Total items: {total_items}")
    print(f"Sparsity: {sparsity:.4f}")
    print(f"Total positive interactions: {positive_interactions}")
    print(f"Total negative interactions: {negative_interactions}")
    # print(f"User interaction count distribution (first 10 users): {user_interaction_counts[:10]}...")  # Show the first 10 for example
    # print(f"Item interaction count distribution (first 10 items): {item_interaction_counts[:10]}...")  # Show the first 10 for example
    print(f"Number of removed users: {removed_users}")


if __name__ == "__main__":
    file_name = "ratings.dat"
    
    # Load the data and split it
    train_dict, valid_dict, test_dict, movie_num, user_num, removed_users_info, user_ratings = simple_load_data_rate(file_name)
    
    # Analyze the data
    analyze_data(user_ratings, movie_num, removed_users_info)

    # Further processing for model data (optional)
    train_user_input, train_movie_input, train_labels = get_model_data(train_dict)
    valid_user_input, valid_movie_input, valid_labels = get_model_data(valid_dict)
    test_user_input, test_movie_input, test_labels = get_model_data(test_dict)
    
    print(len(train_user_input), len(train_movie_input), len(train_labels))
    print(len(valid_user_input), len(valid_movie_input), len(valid_labels))
    print(len(test_user_input), len(test_movie_input), len(test_labels))

    print(removed_users_info)