#include <iostream>
#include <cstdlib>
#include <cmath>
#include <omp.h>

#define N 1000  // Розмірність матриці

void gram_schmidt(double* A, double* Q, double* R, int n) {
    int i, j, k;
    double sum;

    #pragma omp parallel for private(i, j, k, sum) shared(A, Q, R, n)
    for (j = 0; j < n; j++) {
        for (i = 0; i < j; i++) {
            sum = 0.0;
            for (k = 0; k < n; k++) {
                sum += Q[k*n + i] * A[k*n + j];
            }
            R[i*n + j] = sum;
            #pragma omp parallel for
            for (k = 0; k < n; k++) {
                A[k*n + j] -= sum * Q[k*n + i];
            }
        }
        sum = 0.0;
        for (k = 0; k < n; k++) {
            sum += A[k*n + j] * A[k*n + j];
        }
        R[j*n + j] = std::sqrt(sum);
        #pragma omp parallel for
        for (k = 0; k < n; k++) {
            Q[k*n + j] = A[k*n + j] / R[j*n + j];
        }
    }
}

int main() {
    int n = N;
    double* A = new double[n*n];
    double* Q = new double[n*n];
    double* R = new double[n*n];

    // Ініціалізація матриці A випадковими значеннями
    #pragma omp parallel for
    for (int i = 0; i < n; i++) {
        for (int j = 0; j < n; j++) {
            A[i*n + j] = std::rand() / double(RAND_MAX);
        }
    }

    // Виклик функції ортогоналізації
    gram_schmidt(A, Q, R, n);

    // Виведення результатів
    std::cout << "Матриця Q:" << std::endl;
    for (int i = 0; i < n; i++) {
        for (int j = 0; j < n; j++) {
            std::cout << Q[i*n + j] << " ";
        }
        std::cout << std::endl;
    }

    std::cout << "Матриця R:" << std::endl;
    for (int i = 0; i < n; i++) {
        for (int j = 0; j < n; j++) {
            std::cout << R[i*n + j] << " ";
        }
        std::cout << std::endl;
    }

    delete[] A;
    delete[] Q;
    delete[] R;

    return 0;
}
