import matplotlib.pyplot as plt

def plot_train_curve(train_data, nb_products, lambd, d, nb_iter):
    plt.plot(train_data, label="train")
    plt.xlabel("Iteration")
    plt.ylabel("Train loss")
    plt.title(f"Loss functions for products: {nb_products}, lambda: {lambd}, d: {d}, iterations: {nb_iter}")
    plt.legend()
    plt.show()

def plot_train_test_curves(train_data, test_data, nb_products, lambd, d, nb_iter):
    fig, axs = plt.subplots(nrows=1, ncols=2)
    axs[0].plot(train_data, label="train")
    axs[0].set_ylabel("Loss function")
    axs[0].set_xlabel("Iteration")
    axs[1].plot(test_data, label="test", color="orange")
    axs[1].set_xlabel("Iteration")
    plt.title(f"Loss functions for products: {nb_products}, lambda: {lambd}, d: {d}, iterations: {nb_iter}")
    plt.legend()
    plt.show()
    
def normalize_data(data, low=1, high=5):
    data[data>high] = high
    data[data<low] = low
    return data