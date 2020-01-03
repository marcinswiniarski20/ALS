# %%
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from data_preprocessing import split_data
from gauss import partial_pivot_gauss

gauss = partial_pivot_gauss
d = 20
lambd = 0.001

np.set_printoptions(linewidth=np.inf)
dataframe = pd.read_csv('ratings_100.csv', sep=',')
dataframe = dataframe[['user_id', 'product_id', 'user_rating']]
data = dataframe.values

rows, row_pos = np.unique(data[:, 0], return_inverse=True)
cols, col_pos = np.unique(data[:, 1], return_inverse=True)

pivot_table = np.zeros((len(rows), len(cols)), dtype=data.dtype)
pivot_table[row_pos, col_pos] = data[:, 2]
ratings = pivot_table
print(ratings)
print(ratings.shape)
print(f"Number of ratings: {len(np.argwhere(ratings != 0))}")

# Initialize U (users coeffs), P (products coeffs) matricies
U = 5*np.random.rand(d, len(rows))
P = 5*np.random.rand(d, len(cols))


print(f"Rating matrix:")
print(ratings)
training_ratings, test_ratings, nb_tests = split_data(ratings, spliting_ratio=0.8)
print("TRAIN")
print(training_ratings)
print("TEST")
print(test_ratings)
print(training_ratings.shape)
print(ratings.shape)

all_losses = []
for n in range(10000):
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

    # Poprawka
    f = 0
    R = np.dot(np.transpose(U), P)
    for i in range(training_ratings.shape[0]):
        for j in range(training_ratings.shape[1]):
            if training_ratings[i, j] != 0:
                f += (training_ratings[i, j] - R[i, j])**2 + lambd*(np.sqrt(np.sum(U[:, i]**2)) + np.sqrt(np.sum(P[:, j]**2)))

    loss = f
    # to bylo źle
    # f1 = np.sum((ratings - np.dot(np.transpose(U), P))**2)
    # f2 = np.sum(np.sqrt(np.sum(U**2, axis=1)))
    # f3 = np.sum(np.sqrt(np.sum(P**2, axis=1)))
    # loss = f1 + lambd*(f2+f3)
    print(f"Iter: {n} Loss: {loss}")
    all_losses.append(loss)

plt.plot(all_losses)
plt.xlabel('Iteration')
plt.ylabel('Loss')
plt.show()



# %%
ratings_predicted = np.rint(R).astype(np.int32)
err = 0

for i in range(test_ratings.shape[0]):
    for j in range(test_ratings.shape[1]):
        if test_ratings[i, j] != 0:
            err += np.abs(test_ratings[i, j] - ratings_predicted[i,j])
# err = err/nb_tests
err = err/len(np.argwhere(test_ratings != 0))
print(f"Average error of prediction: {err}")