import numpy as np
import random
random.seed(1000)
def load_data_ignore3(filename, threshold=4, train_ratio=0.7, test_ratio=0.15):
    """
    Load dataset and split data on a per-user basis.

    Args:
        filename (str): Path to the ratings file.
        train_ratio (float): Percentage of interactions used for training.
        test_ratio (float): Percentage of interactions used for testing.

    Returns:
        train_dict, val_dict, test_dict, movie_num, user_num
    """
    user_ratings = {}  # Store each user's interactions (user_id: [(movie_id, label), ...])
    movie_num = -1
    user_num = -1

    with open(filename, "r", encoding="utf-8") as file:
        for line in file:
            user_id, movie_id, rating, _ = map(int, line.strip().split("::"))

            # Ignore rating 3
            if rating == 3:
                continue

            # Map ratings >=4 to 1, ratings 1 or 2 to 0
            label = 1 if rating >= threshold else 0

            if user_id not in user_ratings:
                user_ratings[user_id] = []
            user_ratings[user_id].append((movie_id, label))

            # movie and user number
            movie_num = max(movie_num, movie_id)
            user_num = max(user_num, user_id)

    train_dict, val_dict, test_dict = {}, {}, {}

    ######### divide by users? cold start #######
    # Divide each user's movie interactions by proportion
    for user_id, interactions in user_ratings.items():
        random.shuffle(interactions)  # shuffle

        total_interactions = len(interactions)
        train_end = int(train_ratio * total_interactions)
        val_end = int((train_ratio + test_ratio) * total_interactions)

        train_dict[user_id] = interactions[:train_end]
        val_dict[user_id] = interactions[train_end:val_end]
        test_dict[user_id] = interactions[val_end:]

    return train_dict, val_dict, test_dict, movie_num, user_num


def load_data_rate(filename, threshold=3, train_ratio=0.7, test_ratio=0.15):
    """
    Load dataset and split data on a per-user basis.

    Args:
        filename (str): Path to the ratings file.
        train_ratio (float): Percentage of interactions used for training.
        test_ratio (float): Percentage of interactions used for testing.

    Returns:
        train_dict, val_dict, test_dict, movie_num, user_num
    """
    user_ratings = {}  # Store each user's interactions (user_id: [(movie_id, label), ...])
    movie_num = -1
    user_num = -1

    with open(filename, "r", encoding="utf-8") as file:
        for line in file:
            user_id, movie_id, rating, _ = map(int, line.strip().split("::"))

            # # Ignore rating 3
            # if rating == 3:
            #     continue

            # Map ratings >=4 to 1, ratings 1 or 2 to 0
            label = 1 if rating >= threshold else 0

            if user_id not in user_ratings:
                user_ratings[user_id] = []
            user_ratings[user_id].append((movie_id, label))

            # movie and user number
            movie_num = max(movie_num, movie_id)
            user_num = max(user_num, user_id)

    train_dict, val_dict, test_dict = {}, {}, {}

    ######### divide by users? cold start #######
    # Divide each user's movie interactions by proportion
    for user_id, interactions in user_ratings.items():
        random.shuffle(interactions)  # shuffle

        total_interactions = len(interactions)
        train_end = int(train_ratio * total_interactions)
        val_end = int((train_ratio + test_ratio) * total_interactions)

        train_dict[user_id] = interactions[:train_end]
        val_dict[user_id] = interactions[train_end:val_end]
        test_dict[user_id] = interactions[val_end:]

    return train_dict, val_dict, test_dict, movie_num, user_num

def load_data_rate_np(filename, threshold=3, train_ratio=0.7, test_ratio=0.15):
    """
    Load dataset and split data on a per-user basis, ensuring positive and negative samples are split separately.

    Args:
        filename (str): Path to the ratings file.
        threshold (int): Rating threshold to classify positive samples.
        train_ratio (float): Percentage of interactions used for training.
        test_ratio (float): Percentage of interactions used for testing.

    Returns:
        train_dict, val_dict, test_dict, movie_num, user_num
    """
    user_ratings = {}  # Store each user's interactions (user_id: [(movie_id, label), ...])
    movie_num = -1
    user_num = -1

    with open(filename, "r", encoding="utf-8") as file:
        for line in file:
            user_id, movie_id, rating, _ = map(int, line.strip().split("::"))
            
            # Map ratings >= threshold to 1, otherwise 0
            label = 1 if rating >= threshold else 0

            if user_id not in user_ratings:
                user_ratings[user_id] = []
            user_ratings[user_id].append((movie_id, label))

            # Track max user and movie IDs
            movie_num = max(movie_num, movie_id)
            user_num = max(user_num, user_id)

    train_dict, val_dict, test_dict = {}, {}, {}

    # Split data for each user
    for user_id, interactions in user_ratings.items():
        # Separate positives and negatives
        positives = [x for x in interactions if x[1] == 1]
        negatives = [x for x in interactions if x[1] == 0]
        
        # Shuffle both lists
        random.shuffle(positives)
        random.shuffle(negatives)

        def split_data(data):
            total = len(data)
            train_end = int(train_ratio * total)
            val_end = int((train_ratio + test_ratio) * total)
            return data[:train_end], data[train_end:val_end], data[val_end:]
        
        # Apply split separately
        train_pos, val_pos, test_pos = split_data(positives)
        train_neg, val_neg, test_neg = split_data(negatives)
        
        # Merge positive and negative samples
        train_dict[user_id] = train_pos + train_neg
        val_dict[user_id] = val_pos + val_neg
        test_dict[user_id] = test_pos + test_neg
        
        # Shuffle again to mix positives and negatives
        random.shuffle(train_dict[user_id])
        random.shuffle(val_dict[user_id])
        random.shuffle(test_dict[user_id])

    return train_dict, val_dict, test_dict, movie_num, user_num

def load_data_meanStd(filename, train_ratio=0.7, test_ratio=0.15):
    """
    Load dataset and split data on a per-user basis using a dynamic threshold
    based on user-specific mean (μ_u) and standard deviation (σ_u).

    Args:
        filename (str): Path to the ratings file.
        train_ratio (float): Percentage of interactions used for training.
        test_ratio (float): Percentage of interactions used for testing.

    Returns:
        train_dict, val_dict, test_dict, movie_num, user_num
    """
    user_ratings = {}  # Store each user's interactions (user_id: [(movie_id, rating), ...])
    movie_num = -1
    user_num = -1

    # Step 1: Read data and store user ratings
    with open(filename, "r", encoding="utf-8") as file:
        for line in file:
            user_id, movie_id, rating, _ = map(int, line.strip().split("::"))

            if user_id not in user_ratings:
                user_ratings[user_id] = []
            user_ratings[user_id].append((movie_id, rating))

            # Update max movie and user IDs
            movie_num = max(movie_num, movie_id)
            user_num = max(user_num, user_id)

    # Step 2: Compute user-specific mean and standard deviation
    train_dict, val_dict, test_dict = {}, {}, {}

    for user_id, interactions in user_ratings.items():
        ratings = np.array([r for _, r in interactions])
        mu_u = np.mean(ratings)
        sigma_u = np.std(ratings)

        # Step 3: Convert ratings to binary labels based on user-specific threshold
        labeled_interactions = []
        for movie_id, rating in interactions:
            if rating >= mu_u:
                label = 1  # Positive
            elif rating < mu_u - sigma_u:
                label = 0  # Negative
            else:
                continue  # Ignore ratings in the middle range

            labeled_interactions.append((movie_id, label))

        # Step 4: Shuffle and split into train/val/test sets
        random.shuffle(labeled_interactions)
        total_interactions = len(labeled_interactions)
        train_end = int(train_ratio * total_interactions)
        val_end = int((train_ratio + test_ratio) * total_interactions)

        train_dict[user_id] = labeled_interactions[:train_end]
        val_dict[user_id] = labeled_interactions[train_end:val_end]
        test_dict[user_id] = labeled_interactions[val_end:]

    return train_dict, val_dict, test_dict, movie_num, user_num


