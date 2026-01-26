#include <iostream>
#include <cuda_runtime.h>
#include <device_launch_parameters.h>


__constant__ int Gx[3][3] = {{-1, 0, 1}, {-2, 0, 2}, {-1, 0, 1}};
__constant__ int Gy[3][3] = {{-1, -2, -1}, {0, 0, 0}, {1, 2, 1}};

__global__ void init_mat(float* A, int width, int height) {
    int dx = blockIdx.x * blockDim.x + threadIdx.x;
    int dy = blockIdx.y * blockDim.y + threadIdx.y;

    if (dx < width && dy < height) {
        // Desen 32x32'lik kareler (Satranç tahtası gibi)
        A[dy * width + dx] = (float)(((dx / 32) + (dy / 32)) % 2 == 0 ? 200 : 50);
    }
}

__global__ void sobel_edge_det(const float* A, float* Result, int width, int height) {
    int dx = blockIdx.x * blockDim.x + threadIdx.x;
    int dy = blockIdx.y * blockDim.y + threadIdx.y;

    unsigned int index = dy * width + dx;

    // Kenarlardan 1 piksel içeride çalış (3x3 penceresi taşmasın)
    if (dx > 0 && dx < width - 1 && dy > 0 && dy < height - 1) {
        float sumX = 0, sumY = 0;

        for (int i = -1; i <= 1; i++) {
            for (int j = -1; j <= 1; j++) {
                // Doğrudan Global Memory erişimi
                float pixel = A[(dy + i) * width + (dx + j)];
                sumX += pixel * Gx[i + 1][j + 1];
                sumY += pixel * Gy[i + 1][j + 1];
            }
        }

        Result[index] = sqrtf(sumX * sumX + sumY * sumY);
    }
}

int main() {
    const int width = 512;
    const int height = 512;
    size_t size = width * height * sizeof(float);

    // Device Memory
    float *d_A, *d_Result;
    cudaMalloc(&d_A, size);
    cudaMalloc(&d_Result, size);

    // 2D Grid Yapısı
    dim3 threads_per_block(16, 16);
    dim3 blocks_per_grid((width + threads_per_block.x - 1) / threads_per_block.x,
                         (height + threads_per_block.y - 1) / threads_per_block.y);

    // Kernel: Görüntüyü oluştur
    init_mat<<<blocks_per_grid, threads_per_block>>>(d_A, width, height);

    // Kernel: Kenarları tespit et
    sobel_edge_det<<<blocks_per_grid, threads_per_block>>>(d_A, d_Result, width, height);

    cudaDeviceSynchronize();

    // Host Memory
    float *h_A = (float*)malloc(size);
    float *h_Result = (float*)malloc(size);

    // Verileri CPU'ya çek
    cudaMemcpy(h_A, d_A, size, cudaMemcpyDeviceToHost);
    cudaMemcpy(h_Result, d_Result, size, cudaMemcpyDeviceToHost);

    // Dosyaları Kaydet
    //  kaynak_A.pgm
    //  sonuc_sobel.pgm

    // Belleği temizle
    free(h_A);
    free(h_Result);
    cudaFree(d_A);
    cudaFree(d_Result);

    return 0;
}