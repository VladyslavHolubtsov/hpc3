#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <omp.h>

// Функція для обчислення векторного добутку двох векторів
void vector_cross_product(double *a, double *b, double *result) {
    result[0] = a[1] * b[2] - a[2] * b[1];
    result[1] = a[2] * b[0] - a[0] * b[2];
    result[2] = a[0] * b[1] - a[1] * b[0];
}

// Функція для нормалізації вектора
void vector_normalize(double *v) {
    double length = sqrt(v[0] * v[0] + v[1] * v[1] + v[2] * v[2]);
    v[0] /= length;
    v[1] /= length;
    v[2] /= length;
}

// Функція для ортогоналізації стовпців матриці методом Грама-Шмідта
void gram_schmidt(double **matrix, int rows, int cols) {
    int i, j, k;
    double *temp = malloc(cols * sizeof(double));
    
    for (i = 1; i < cols; i++) {
        for (j = 0; j < i; j++) {
            // Обчислення проекції стовпця на попередні ортогональні стовпці
            double projection = 0;
            for (k = 0; k < rows; k++) {
                projection += matrix[k][i] * matrix[k][j];
            }
            
            // Віднімання проекції від поточного стовпця
            for (k = 0; k < rows; k++) {
                temp[k] = matrix[k][i] - projection * matrix[k][j];
            }
            
            // Оновлення стовпця матриці
            for (k = 0; k < rows; k++) {
                matrix[k][i] = temp[k];
            }
        }
        
        // Нормалізація стовпця
        vector_normalize(matrix[:, i]);
    }
    
    free(temp);
}

int main() {
    int rows = 3;
    int cols = 3;
    double **matrix = malloc(rows * sizeof(double *));
    int i, j;
    
    // Ініціалізація матриці випадковими значеннями
    for (i = 0; i < rows; i++) {
        matrix[i] = malloc(cols * sizeof(double));
        for (j = 0; j < cols; j++) {
            matrix[i][j] = (double)rand() / RAND_MAX; // Випадкове число від 0 до 1
        }
    }
    
    // Виконання ортогоналізації стовпців
    gram_schmidt(matrix, rows, cols);
    
    // Виведення результату
    printf("Результат ортогоналізації:\n");
    for (i = 0; i < rows; i++) {
        for (j = 0; j < cols; j++) {
            printf("%f ", matrix[i][j]);
        }
        printf("\n");
    }
    
    // Звільнення пам'яті
    for (i = 0; i < rows; i++) {
        free(matrix[i]);
    }
    free(matrix);
    
    return 0;
}
