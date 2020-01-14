# %%
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from data_preprocessing import split_data
from gauss import partial_pivot_gauss
from timeit import default_timer as timer

import utils

# Initial values
gauss = np.linalg.solve
d = 5
lambd = 0.05
nb_iter = 100
nb_products = 30
category = "Video"
verbose = 2

# For printing whole array
np.set_printoptions(linewidth=np.inf)

# Reading ratings 
dataframe = pd.read_csv(f'ratings_10000.csv', sep=',')
dataframe = dataframe[dataframe['category'] == category]
dataframe = dataframe[['user_id', 'product_id', 'user_rating']]
dataframe = utils.extract_nb_of_products(dataframe, nb_products)
data = dataframe.values

# Creating pivot table
rows, row_pos = np.unique(data[:, 0], return_inverse=True)
cols, col_pos = np.unique(data[:, 1], return_inverse=True)
pivot_table = np.zeros((len(rows), len(cols)), dtype=np.float64)
pivot_table[row_pos, col_pos] = data[:, 2]
ratings = pivot_table

print(f"Number of ratings: {len(np.argwhere(ratings != 0))}")
print(f"Number of users: {len(rows)}")
print(f"Number of products: {len(cols)}")
nb_products=len(cols)

# Initialize U (users coeffs), P (products coeffs) matricies
np.random.seed(42)
U = 5*np.random.rand(d, len(rows), dtype=np.float64)
P = 5*np.random.rand(d, len(cols), dtype=np.float64)

# Spliting data into test and train set
training_ratings, test_ratings, nb_tests = split_data(ratings, spliting_ratio=0.8, seed=42)

training_losses = []
test_losses = []
start_als = timer()

for n in range(nb_iter):
    # ALS algorithm
    for u in range(len(rows)):
        I_u = np.argwhere(training_ratings[u] != 0).flatten()
        P_I_u = P[:, I_u]
        P_I_u_T = np.transpose(P_I_u)
        E = np.eye(d)
        A_u = np.dot(P_I_u, P_I_u_T) + lambd*E
        V_u = 0
        for i in I_u:
            V_u += training_ratings[u, i]*P[:, i]

        U_u = gauss(A_u, V_u)
        U[:, u] = U_u

    for p in range(len(cols)):
        I_p = np.argwhere(training_ratings[:, p] != 0).flatten()
        U_I_p = U[:, I_p]
        U_I_p_T = np.transpose(U_I_p)
        E = np.eye(d)
        B_p = np.dot(U_I_p, U_I_p_T) + lambd*E
        W_p = 0
        for i in I_p:
            W_p += training_ratings[i, p]*U[:, i]

        P_p = gauss(B_p, W_p)
        P[:, p] = P_p

    training_loss = 0
    test_loss = 0
    nb_test_ratings = 1
    R = np.dot(np.transpose(U), P)
    for i in range(training_ratings.shape[0]):
        for j in range(training_ratings.shape[1]):
            if training_ratings[i, j] != 0:
                training_loss += (training_ratings[i, j] - R[i, j])**2 + lambd*(np.sqrt(np.sum(U[:, i]**2)) + np.sqrt(np.sum(P[:, j]**2)))            
            if test_ratings[i, j] != 0:
                test_loss += (test_ratings[i, j] - R[i, j])**2
                nb_test_ratings += 1

    test_loss = test_loss/nb_test_ratings
    # Save best matrix for test set
    if n == 0:
        R_best = R
        best_iter = n
        best_test_loss = test_loss
    elif test_loss <= test_losses[n-1]:
        R_best = R
        best_iter = n
        best_test_loss = test_loss

    if verbose > 0:
        print(f"Iter: {n} train loss: {training_loss}")

    training_losses.append(training_loss)
    test_losses.append(test_loss)

R = R_best
stop_als = timer()
print(f"Lambda: {lambd}, d: {d}, products: {nb_products}, iterations: {nb_iter}")
print(f"Training loss: {training_loss}")
print(f"Test MSE loss: {best_test_loss}")
print(f"Time needed to compute: {stop_als-start_als}")
ratings_predicted = np.rint(R).astype(np.int32)
ratings_predicted = utils.normalize_data(ratings_predicted)

ground_truth_ratings = ""
predicted_ratings_string = ""
avg_err = 0
for i in range(test_ratings.shape[0]):
        for j in range(test_ratings.shape[1]):
            if test_ratings[i, j] != 0:
                avg_err += np.abs(test_ratings[i, j] - ratings_predicted[i,j])
                ground_truth_ratings += f"{test_ratings[i, j]}, "
                predicted_ratings_string += f"{ratings_predicted[i,j]}, "
avg_err = avg_err/len(np.argwhere(test_ratings != 0))

print(f"Average error per rating: {avg_err}")
print(f"Test ratings: {ground_truth_ratings}")
print(f"Predicted ratings: {predicted_ratings_string}")
print()
print("All predicted ratings")
print(ratings_predicted)

if verbose > 1:
    utils.plot_train_test_curves(training_losses, test_losses, nb_products, lambd, d, nb_iter)
else:
    utils.plot_train_curve(training_losses, nb_products, lambd, d, nb_iter)
