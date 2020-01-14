import numpy as np

def partial_pivot_gauss(A, b, eps=1e-6):
    n = len(b)
    X = np.zeros(n, np.float64)
    for k in range(0, n-1):
        if np.abs(A[k, k]) < eps:
            for i in range(k+1, n):
                if np.abs(A[i, k]) > np.abs(A[k, k]):
                    A[[k, i]] = A[[i, k]]
                    b[[k, i]] = b[[i, k]]
                    break
        # Gauss elimination
        for i in range(k+1, n):
            if A[k, k] == 0:
                continue
            coeff = A[i, k] / A[k, k]
            for j in range(k, n):
                A[i, j] = A[i, j] - coeff*A[k, j]
            b[i] = b[i] - coeff*b[k]
    
    # Reverse Gauss procedure
    X[n-1] = b[n-1]/A[n-1, n-1]
    for i in reversed(range(0, n-1)):
       sum_ax = 0
       for j in range(i+1, n):
           sum_ax += A[i, j] * X[j]
       X[i] = (b[i] - sum_ax)/A[i, i]
    
    return X


def test():        
    a = np.array([[0, 31, -1, 3, 1, -15],
                [0, 32, 0, -11, 70, -3],
                [61, 2, 22, -12, -1, 22],
                [-2, 17, 24, 0, 2, -6],
                [3, 0, 14, -27, 1, -5],
                [62, 31, -4, 5, 2, 0]], dtype=np.float64)
    b = np.array([55, 47, 22, 3, 4, 8], dtype=np.float64)

    solution = partial_pivot_gauss(a, b)
    print(f"Partial pivot solution : {solution}")
    numpy_solution = np.linalg.solve(a, b)
    print(f"Numpy linalg solution: {numpy_solution}")
    avg_err = np.mean(np.abs(numpy_solution - solution))
    print(f"Average error between solutions: {avg_err}")


if __name__ == "__main__":
    test()
