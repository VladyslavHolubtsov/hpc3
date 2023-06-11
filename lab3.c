#include <stdio.h>
#include <stdlib.h>
#include <omp.h>
#include <math.h>

#define MATRIX_SIZE 1000

void gram_schmidt(double matrix[MATRIX_SIZE][MATRIX_SIZE], double q[MATRIX_SIZE][MATRIX_SIZE], double r[MATRIX_SIZE][MATRIX_SIZE]) {
    int i, j, k;
    double sum;

    // Обчислення першого стовпця Q і матриці R
    sum = 0.0;
    for (i = 0; i < MATRIX_SIZE; i++) {
        q[i][0] = matrix[i][0];
        sum += q[i][0] * q[i][0];
    }
    r[0][0] = sqrt(sum);

    for (i = 0; i < MATRIX_SIZE; i++) {
        q[i][0] /= r[0][0];
    }

    // Обчислення решти стовпців Q і матриці R
    for (j = 1; j < MATRIX_SIZE; j++) {
        for (i = 0; i < MATRIX_SIZE; i++) {
            sum = 0.0;
            for (k = 0; k < j; k++) {
                sum += q[i][k] * matrix[i][j];
            }
            r[k][j] = sum;

            for (k = 0; k < j; k++) {
                matrix[i][j] -= r[k][j] * q[i][k];
            }
            r[j][j] = sqrt(matrix[i][j] * matrix[i][j] + sum * sum);
            q[i][j] = matrix[i][j] / r[j][j];
        }
    }
}

int main() {
    int num_threads;
    double start_time, end_time;
    double matrix[MATRIX_SIZE][MATRIX_SIZE];
    double q[MATRIX_SIZE][MATRIX_SIZE];
    double r[MATRIX_SIZE][MATRIX_SIZE];
    double acceleration;

    srand(0);

    // Ініціалізація матриці
    for (int i = 0; i < MATRIX_SIZE; i++) {
        for (int j = 0; j < MATRIX_SIZE; j++) {
            matrix[i][j] = (double)rand() / RAND_MAX;
        }
    }

    printf("Кількість потоків\tЧас виконання (сек)\tПрискорення\n");

    for (num_threads = 1; num_threads <= 8; num_threads++) {
        omp_set_num_threads(num_threads);
        start_time = omp_get_wtime();

        #pragma omp parallel shared(matrix, q, r)
        {
            #pragma omp single
            gram_schmidt(matrix, q, r);
        }

        end_time = omp_get_wtime();
        acceleration = (end_time - start_time) / (end_time / num_threads);
        printf("%d\t\t\t%.6f\t\t\t%.2f\n", num_threads, end_time - start_time, acceleration);
    }

    return 0;
}
