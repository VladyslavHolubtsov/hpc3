#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <omp.h>

#define MATRIX_SIZE 1000
#define NUM_THREADS 8

void gram_schmidt(double** matrix, int size) {
    int i, j, k;
    double dot_product;
    
    for (k = 0; k < size; k++) {
        #pragma omp parallel for shared(matrix) private(i, j, dot_product) num_threads(NUM_THREADS)
        for (i = 0; i < k; i++) {
            dot_product = 0.0;
            for (j = 0; j < size; j++) {
                dot_product += matrix[k][j] * matrix[i][j];
            }
            for (j = 0; j < size; j++) {
                matrix[k][j] -= dot_product * matrix[i][j];
            }
        }
        dot_product = 0.0;
        #pragma omp parallel for shared(matrix) private(j) reduction(+:dot_product) num_threads(NUM_THREADS)
        for (j = 0; j < size; j++) {
            dot_product += matrix[k][j] * matrix[k][j];
        }
        double norm = sqrt(dot_product);
        #pragma omp parallel for shared(matrix, norm) private(j) num_threads(NUM_THREADS)
        for (j = 0; j < size; j++) {
            matrix[k][j] /= norm;
        }
    }
}

int main() {
    int i, j;
    double** matrix = (double**)malloc(MATRIX_SIZE * sizeof(double*));
    for (i = 0; i < MATRIX_SIZE; i++) {
        matrix[i] = (double*)malloc(MATRIX_SIZE * sizeof(double));
    }
    
    // Заповнення матриці випадковими значеннями
    for (i = 0; i < MATRIX_SIZE; i++) {
        for (j = 0; j < MATRIX_SIZE; j++) {
            matrix[i][j] = (double)rand() / RAND_MAX;
        }
    }
    
    double start_time = omp_get_wtime();
    
    // Виклик функції ортогоналізації
    gram_schmidt(matrix, MATRIX_SIZE);
    
    double end_time = omp_get_wtime();
    
    // Виведення результату
    printf("Orthogonalized matrix:\n");
    for (i = 0; i < MATRIX_SIZE; i++) {
        for (j = 0; j < MATRIX_SIZE; j++) {
            printf("%f ", matrix[i][j]);
        }
        printf("\n");
    }
    
    // Звільнення пам'яті
    for (i = 0; i < MATRIX_SIZE; i++) {
        free(matrix[i]);
    }
    free(matrix);
    
    double execution_time = end_time - start_time;
    printf("Execution time: %f seconds\n", execution_time);
    
    int num_processors = omp_get_num_procs();
    double speedup = execution_time / (execution_time / NUM_THREADS + (1.0 - 1.0 / NUM_THREADS));
    double efficiency = speedup / NUM_THREADS * 100.0;
    
    printf("Number of processors: %d\n", NUM_THREADS);
    printf("Speedup: %f\n", speedup);
    printf("Efficiency: %.2f%%\n", efficiency);
    
    return 0;
}
