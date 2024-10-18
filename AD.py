import numpy as np
import time

# Function for naive matrix multiplication
def naive_matrix_multiplication(A, B):
    n = A.shape[0]
    C = np.zeros((n, n))
    for i in range(n):
        for j in range(n):
            for k in range(n):
                C[i][j] += A[i][k] * B[k][j]
    return C

# Function for cache-friendly matrix multiplication (loop reordering)
def cache_friendly_matrix_multiplication(A, B):
    n = A.shape[0]
    C = np.zeros((n, n))
    for i in range(n):
        for k in range(n):
            for j in range(n):
                C[i][j] += A[i][k] * B[k][j]
    return C

# Testing the performance of both implementations
def main():
    # Create two random square matrices of size 500x500
    n = 500
    A = np.random.rand(n, n)
    B = np.random.rand(n, n)

    # Naive matrix multiplication
    start_naive = time.time()
    C_naive = naive_matrix_multiplication(A, B)
    end_naive = time.time()
    naive_time = end_naive - start_naive
    print(f"Naive Matrix Multiplication Time: {naive_time:.4f} seconds")

    # Cache-friendly matrix multiplication
    start_cache_friendly = time.time()
    C_cache_friendly = cache_friendly_matrix_multiplication(A, B)
    end_cache_friendly = time.time()
    cache_friendly_time = end_cache_friendly - start_cache_friendly
    print(f"Cache-Friendly Matrix Multiplication Time: {cache_friendly_time:.4f} seconds")

    # Comparing the result (both should be nearly identical)
    print(f"Result difference (should be close to 0): {np.sum(np.abs(C_naive - C_cache_friendly)):.6f}")

if __name__ == "__main__":
    main()
