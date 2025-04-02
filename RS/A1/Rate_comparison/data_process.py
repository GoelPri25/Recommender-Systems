import random
random.seed(1000)

def get_non_interacted_movies(train_dict, val_dict, test_dict, movie_num):
    non_interacted_movies = {}

    for u in train_dict:
        # Get the movies that the user has interacted with (including train, val, test)
        interacted_movies = set(movie_id for movie_id, _ in train_dict.get(u, []))
        if u in val_dict:
            interacted_movies.update(movie_id for movie_id, _ in val_dict.get(u, []))
        if u in test_dict:
            interacted_movies.update(movie_id for movie_id, _ in test_dict.get(u, []))

        # Get the movies that the user has not interacted with
        all_movies = set(range(1, movie_num + 1))
        non_interacted_movies[u] = list(all_movies - interacted_movies)

    return non_interacted_movies

def get_train_data(train_dict, non_interacted_movies, negative_num):
    user_input, movie_input, labels = [], [], []
    neg_sample_train = {}

    for u, rate_list in train_dict.items():
        positive_samples = []
        negative_candidates = []

        for movie_id, label in rate_list:
            if label == 1:
                positive_samples.append(movie_id)
            else:
                negative_candidates.append(movie_id)

        for movie_id in positive_samples:
            user_input.append(u)
            movie_input.append(movie_id)
            labels.append(1)

            non_interacted_items = non_interacted_movies.get(u, [])
            available_negatives = negative_candidates + list(non_interacted_items)

            if len(available_negatives) >= negative_num:
                negative_samples = random.sample(available_negatives, negative_num)
            else:
                negative_samples = available_negatives + random.choices(
                    available_negatives, k=negative_num - len(available_negatives)
                )

            for neg_movie_id in negative_samples:
                user_input.append(u)
                movie_input.append(neg_movie_id)
                labels.append(0)

        neg_sample_train[u] = set(negative_samples)

    # --------------- Shuffle ---------------
    data = list(zip(user_input, movie_input, labels))
    random.shuffle(data)  
    user_input, movie_input, labels = zip(*data) 

    return list(user_input), list(movie_input), list(labels), neg_sample_train

def get_all_noninteract_validation(val_dict, non_interacted_movies, neg_sample_train):
    user_input, movie_input, labels = [], [], []
    neg_sample_val = {} 

    for u, rate_list in val_dict.items():
        for movie_id, label in rate_list:
            user_input.append(u)
            movie_input.append(movie_id)
            labels.append(label)

        non_interacted_items = set(non_interacted_movies.get(u, []))
        excluded_neg_samples = neg_sample_train.get(u, set())
        valid_neg_samples = non_interacted_items - excluded_neg_samples

        neg_sample_val[u] = set(valid_neg_samples)

        for movie_id in valid_neg_samples:
            user_input.append(u)
            movie_input.append(movie_id)
            labels.append(0)

    return user_input, movie_input, labels, neg_sample_val



def get_part_noninteract_validation(val_dict, non_interacted_movies, neg_sample_train, pos_num=5, neg_num=500):
    user_discard = []
    user_input, movie_input, labels = [], [], []
    neg_sample_val = {}

    for u, rate_list in val_dict.items():
        pos_samples = [(movie_id, label) for movie_id, label in rate_list if label == 1]

        if len(pos_samples) < pos_num:
            continue
        
        pos_samples = pos_samples[:pos_num]

        interacted_neg_samples = {movie_id for movie_id, label in rate_list if label == 0}
        non_interacted_items = set(non_interacted_movies.get(u, []))
        excluded_neg_samples = neg_sample_train.get(u, set())

        valid_neg_samples = (interacted_neg_samples | non_interacted_items) - excluded_neg_samples
        valid_neg_samples = list(valid_neg_samples)

        if len(valid_neg_samples) < pos_num * neg_num:
            user_discard.append(u)
            continue
        
        neg_sample_val[u] = set(valid_neg_samples)

        for movie_id, label in pos_samples:
            user_input.append(u)
            movie_input.append(movie_id)
            labels.append(label)

            sampled_negatives = random.sample(valid_neg_samples, neg_num)
            for neg_movie in sampled_negatives:
                user_input.append(u)
                movie_input.append(neg_movie)
                labels.append(0)
    
    # --------------- Shuffle ---------------
    data = list(zip(user_input, movie_input, labels))
    random.shuffle(data)  
    user_input, movie_input, labels = zip(*data)  

    return list(user_input), list(movie_input), list(labels), neg_sample_val, len(user_discard)


def get_all_noninteract_test(test_dict, non_interacted_movies, neg_sample_train, neg_sample_val):
    user_input, movie_input, labels = [], [], []
    
    for u, rate_list in test_dict.items():
        for movie_id, label in rate_list:
            user_input.append(u)
            movie_input.append(movie_id)
            labels.append(label)

        non_interacted_items = set(non_interacted_movies.get(u, []))
        excluded_neg_samples_train = neg_sample_train.get(u, set())
        excluded_neg_samples_val = neg_sample_val.get(u, set())
        
        excluded_neg_samples = excluded_neg_samples_train.union(excluded_neg_samples_val)
        valid_neg_samples = non_interacted_items - excluded_neg_samples

        neg_sample_val[u] = set(valid_neg_samples)

        for movie_id in valid_neg_samples:
            user_input.append(u)
            movie_input.append(movie_id)
            labels.append(0)

    # --------------- Shuffle ---------------
    data = list(zip(user_input, movie_input, labels))
    random.shuffle(data)  
    user_input, movie_input, labels = zip(*data) 

    return list(user_input), list(movie_input), list(labels)