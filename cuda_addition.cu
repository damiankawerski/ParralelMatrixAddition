#include <iostream>
#include <cstdlib>
#include <ctime>
#include <cuda_runtime.h>

const int size = 100000; 
const long long total_elements = static_cast<long long>(size) * size;
const int TILE_SIZE = 2000;  // Zmniejszony rozmiar kafelka dla lepszego wykorzystania VRAM

// CUDA kernel do dodawania macierzy
__global__ void add_matrices(int* a, int* b, int* c, int tile_width, int tile_height, int full_size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int idy = blockIdx.y * blockDim.y + threadIdx.y;

    if (idx < tile_width && idy < tile_height) {
        // Używamy prostego indeksowania w obrębie kafelka
        int index = idy * tile_width + idx;
        c[index] = a[index] + b[index];
    }
}

// Funkcja do wypełniania fragmentu macierzy losowymi liczbami
void random_fill_matrix_tile(int* matrix, long long start_idx, long long elements) {
    for (long long i = 0; i < elements; ++i) {
        matrix[i] = rand() % 20001 - 10000;
    }
}

int main() {
    srand(time(nullptr));
    
    std::cout << "Starting matrix addition on RTX 4070 Ti (12GB VRAM)" << std::endl;
    std::cout << "Matrix size: " << size << "x" << size << " (" << (static_cast<double>(total_elements) * sizeof(int) / (1024.0 * 1024.0 * 1024.0)) << " GB per matrix)" << std::endl;
    std::cout << "Using tile size: " << TILE_SIZE << "x" << TILE_SIZE << std::endl;

    // Do pomiaru łącznego czasu dodawania macierzy
    double total_kernel_time = 0.0;
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    // Iteracja po kafelkach
    for (int y = 0; y < size; y += TILE_SIZE) {
        for (int x = 0; x < size; x += TILE_SIZE) {
            int current_tile_size_x = std::min(TILE_SIZE, size - x);
            int current_tile_size_y = std::min(TILE_SIZE, size - y);
            long long current_tile_elements = static_cast<long long>(current_tile_size_x) * current_tile_size_y;
            
            std::cout << "Processing tile at (" << x << "," << y << ") with size " 
                     << current_tile_size_x << "x" << current_tile_size_y << std::endl;

            // Alokuj kafelki na CPU
            int* h_tile1 = new int[current_tile_elements];
            int* h_tile2 = new int[current_tile_elements];
            int* h_result = new int[current_tile_elements];

            // Wypełnij kafelki danymi (poza pomiarem czasu)
            random_fill_matrix_tile(h_tile1, 0, current_tile_elements);
            random_fill_matrix_tile(h_tile2, 0, current_tile_elements);

            // Alokuj kafelki na GPU
            int* d_tile1;
            int* d_tile2;
            int* d_result;
            cudaMalloc(&d_tile1, current_tile_elements * sizeof(int));
            cudaMalloc(&d_tile2, current_tile_elements * sizeof(int));
            cudaMalloc(&d_result, current_tile_elements * sizeof(int));

            // Kopiuj kafelki do GPU (poza pomiarem czasu)
            cudaMemcpy(d_tile1, h_tile1, current_tile_elements * sizeof(int), cudaMemcpyHostToDevice);
            cudaMemcpy(d_tile2, h_tile2, current_tile_elements * sizeof(int), cudaMemcpyHostToDevice);

            // Konfiguracja kernela
            dim3 threadsPerBlock(32, 32);
            dim3 numBlocks(
                (current_tile_size_x + threadsPerBlock.x - 1) / threadsPerBlock.x,
                (current_tile_size_y + threadsPerBlock.y - 1) / threadsPerBlock.y
            );

            // Rozpocznij pomiar czasu dla kernela
            cudaEventRecord(start);

            // Odpal kernel
            add_matrices<<<numBlocks, threadsPerBlock>>>(d_tile1, d_tile2, d_result, 
                                                       current_tile_size_x, current_tile_size_y, size);
            
            // Zakończ pomiar czasu dla kernela
            cudaEventRecord(stop);
            cudaEventSynchronize(stop);
            
            // Sprawdź błędy CUDA
            cudaError_t err = cudaGetLastError();
            if (err != cudaSuccess) {
                std::cerr << "CUDA Error: " << cudaGetErrorString(err) << std::endl;
                return -1;
            }

            // Oblicz czas wykonania kernela
            float kernel_milliseconds = 0;
            cudaEventElapsedTime(&kernel_milliseconds, start, stop);
            total_kernel_time += kernel_milliseconds / 1000.0; // Konwersja z ms na sekundy
            
            // Kopiuj wynik z powrotem do CPU (poza pomiarem czasu)
            cudaMemcpy(h_result, d_result, current_tile_elements * sizeof(int), cudaMemcpyDeviceToHost);
            
            // Zwolnij zasoby GPU
            cudaFree(d_tile1);
            cudaFree(d_tile2);
            cudaFree(d_result);
            
            // Zwolnij zasoby CPU
            delete[] h_tile1;
            delete[] h_tile2;
            delete[] h_result;
        }
    }

    // Zniszcz wydarzenia CUDA
    cudaEventDestroy(start);
    cudaEventDestroy(stop);

    std::cout << "Total kernel execution time: " << total_kernel_time << " seconds" << std::endl;
    std::cout << "Kernel throughput: " << (static_cast<double>(total_elements) * 2 * sizeof(int) / (1024.0 * 1024.0 * 1024.0)) / total_kernel_time 
              << " GB/s" << std::endl;

    return 0;
}

// Total kernel execution time: 2.23779 seconds
// Kernel throughput: 33.2944 GB/s