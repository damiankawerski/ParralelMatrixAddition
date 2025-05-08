#include <iostream>
#include <omp.h>
#include <cstdlib>
#include <ctime>
#include <vector>

// Constants
int size = 25000;


// Global matrices
std::vector<std::vector<int>> matrix1(size, std::vector<int>(size));
std::vector<std::vector<int>> matrix2(size, std::vector<int>(size));
std::vector<std::vector<int>> result(size, std::vector<int>(size));

// Function to fill matrix with randomized values from -10000 to 10000
void random_fill_matrix(std::vector<std::vector<int>>& matrix) {
    for (int i = 0; i < size; ++i) {
        for (int j = 0; j < size; ++j) {
            matrix[i][j] = rand() % 20001 - 10000;
        }
    }
}

// Function to add two matrices
void add_matrices(const std::vector<std::vector<int>>& matrix1, const std::vector<std::vector<int>>& matrix2, std::vector<std::vector<int>>& result) {
    for (int i = 0; i < size; ++i) {
        for (int j = 0; j < size; ++j) {
            result[i][j] = matrix1[i][j] + matrix2[i][j];
        }
    }
}

int main() {
    srand(time(nullptr));

    random_fill_matrix(matrix1);
    random_fill_matrix(matrix2);

    double start_time = omp_get_wtime();
    add_matrices(matrix1, matrix2, result);
    double end_time = omp_get_wtime();

    std::cout << "Time taken for matrix addition: " << (end_time - start_time) << " seconds " << "for size: " << size <<  std::endl;
}


// Compile with: g++ -fopenmp sequence_addition.cpp -o sequence_addition
// Time taken for matrix addition: 0.415228 seconds for size: 5000
// ime taken for matrix addition: 219.82 seconds for size: 100000
