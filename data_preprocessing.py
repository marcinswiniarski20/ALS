import numpy as np
import re
import os
from timeit import default_timer as timer

def get_records_from_file(file):        
    reading_start = timer()
    with open('amazon-meta.txt', 'r', encoding='utf8') as file:
        data = file.read()
    print(f"Time needed to read amazon data: {timer() - reading_start}")
    splitting_start = timer()
    records = data.split('\n\n')
    print(f"Time needed to split amazon data: {timer() - splitting_start}")
    
    records = records[2:]
    return records


def extract_product_from_record(record):
    id = int(re.findall(r"Id:   (\d*)", record)[0])
    return id


def extract_data_from_record(record, file_to_write=None):
    product_id = extract_product_from_record(record)
    ratings = re.findall(r"cutomer:\s*(.*?)  votes", record)
    category = re.findall(r"group:\s(.*?)\n", record)
    users_ratings = []
    if ratings is None:
        return
    if len(category) > 0:
        category = category[0]
    for rating in ratings:
        user_id = rating.split('  ')[0].replace(' ', '')
        user_rate = rating.split(': ')[1].replace(' ', '')
        users_ratings.append((user_id, int(user_rate)))

    if file_to_write is not None:
        with open(file_to_write, "a+") as file:
            for user_rating in users_ratings:
                file.write(f"{user_rating[0]},{product_id},{user_rating[1]},{category}\n")

    return product_id, users_ratings

def create_csv(file_path):
    with open(file_path, "w") as file:
        file.write("user_id,product_id,user_rating,category\n")

def split_data(ratings, spliting_ratio=0.8, seed=None):
    print(f"Diving data into training and test set with splitiing ratio: {spliting_ratio}")
    non_zero_idxs = np.argwhere(ratings != 0)
    nb_of_ratings = len(non_zero_idxs)
    test_set_size = int((1-spliting_ratio)*nb_of_ratings)
    test_ratings = np.zeros(ratings.shape)
    training_ratings = np.copy(ratings)
    nb_tests = 0
    for i, user_ratings in enumerate(ratings):
        non_zero_user_ratings = np.argwhere(user_ratings != 0)
        if len(non_zero_user_ratings) > 1:
            if seed is not None:
                np.random.seed(seed)
            np.random.shuffle(non_zero_user_ratings)
            for non_zero_user_rating in non_zero_user_ratings:
                non_zero_product_ratings = np.argwhere(training_ratings[:, non_zero_user_rating])
                if len(non_zero_product_ratings) > 1:
                    test_ratings[i, non_zero_user_rating] = training_ratings[i, non_zero_user_rating]
                    training_ratings[i, non_zero_user_rating] = 0
                    nb_tests += 1
                    break
            if nb_tests >= test_set_size:
                break
            
    print(f"Test items: {nb_tests}")    
    return training_ratings, test_ratings.astype(np.int32), nb_tests

if __name__ == "__main__":
    product_start = 0
    product_stop = 10000
    nb_of_products = product_stop - product_start
    file_to_write = f"ratings_{nb_of_products}.csv"    
    start_extrating = timer()
    records = get_records_from_file("amazon-meta.txt")
    create_csv(file_to_write)
    
    for i, record in enumerate(records[product_start:product_stop]):
        extract_data_from_record(record, file_to_write)
        print(f"Product processed: {i}")
    stop_extracting = timer()
    time_of_extraction = stop_extracting - start_extrating
    print(f"Time needed extracted whole amazon data: {time_of_extraction}")