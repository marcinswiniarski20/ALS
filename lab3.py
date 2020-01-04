# %%
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from data_preprocessing import split_data
from gauss import partial_pivot_gauss

gauss = np.linalg.solve

d = 20
lambd = 0.01
nb_iter = 8000
products = 'example_data'
np.set_printoptions(linewidth=np.inf)
dataframe = pd.read_csv('example_data.csv', sep=';')
dataframe = dataframe[['id_user', 'id_product', 'rating']]

data = dataframe.values

rows, row_pos = np.unique(data[:, 0], return_inverse=True)
cols, col_pos = np.unique(data[:, 1], return_inverse=True)


pivot_table = np.zeros((len(rows), len(cols)), dtype=data.dtype)
pivot_table[row_pos, col_pos] = data[:, 2]
ratings = pivot_table

# Initialize U (users coeffs), P (products coeffs) matricies
U = 5*np.random.rand(d, len(rows))
P = 5*np.random.rand(d, len(cols))


print(f"Rating matrix:")
print(ratings)
training_ratings, test_ratings, nb_tests = split_data(ratings, spliting_ratio=0.8, seed=51)
print("Train matrix: ")
print(training_ratings)
print("Test matrix: ")
print(test_ratings)

training_losses = []
test_losses = []
for n in range(nb_iter):
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
    R = np.dot(np.transpose(U), P)
    for i in range(training_ratings.shape[0]):
        for j in range(training_ratings.shape[1]):
            if training_ratings[i, j] != 0:
                training_loss += (training_ratings[i, j] - R[i, j])**2 + lambd*(np.sqrt(np.sum(U[:, i]**2)) + np.sqrt(np.sum(P[:, j]**2)))
                # nb_train += 1            
            if test_ratings[i, j] != 0:
                test_loss += (test_ratings[i, j] - R[i, j])**2
                nb_test += 1
                # err += np.abs(test_ratings[i, j] - ratings_predicted[i,j])
    # training_loss = training_loss/nb_train
    test_loss = test_loss/nb_test
    print(f"Iter: {n} train loss: {training_loss} test loss: {test_loss}")
    training_losses.append(training_loss)
    test_losses.append(test_loss)

print(f"Lambda: {lambd}, number of iterations: {nb_iter}")
print(f"Training loss: {training_loss}")
print(f"Test loss: {test_loss}")

fig, axs = plt.subplots(nrows=1, ncols=2)
axs[0].plot(training_losses, label="train")
axs[0].set_ylabel("Loss function")
axs[0].set_xlabel("Iteration")
axs[1].plot(test_losses, label="test", color="orange")
axs[1].set_xlabel("Iteration")
plt.title(f"Loss functions lambda: {lambd}, d: {d}, iterations: {nb_iter}")
plt.legend()
plt.show()

# %%
print("After ALS algorithm")
print("R matrix")
print(R)
print(np.rint(R).astype(np.int32))
print("Test matrix")
print(test_ratings)
print("Train matrix")
print(training_ratings)


ratings_predicted = np.rint(R).astype(np.int32)
err = 0

for i in range(test_ratings.shape[0]):
    for j in range(test_ratings.shape[1]):
        if test_ratings[i, j] != 0:
            err += np.abs(test_ratings[i, j] - ratings_predicted[i,j])
# err = err/nb_tests
err = err/len(np.argwhere(test_ratings != 0))
print(f"Average error of prediction: {err}")
# %%
# calc_ratings = np.dot(np.transpose(U), P).astype(int)
# rmse = np.sqrt(np.mean((ratings - calc_ratings)**2))
# print(rmse)
# print(np.mean(np.abs(ratings - calc_ratings)))
