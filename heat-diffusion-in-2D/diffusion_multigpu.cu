// compile with: nvcc -O2 -Xcompiler -fopenmp -o diffusion_multigpu diffusion_multigpu.cu

#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>
#include <math.h>
#include <chrono>
#include <string.h>

#define N 1024
#define HALF (N / 2)
#define MAX_ITER 100000
#define TOLERANCE 1e-6f

#define TOP_TEMP 400.0f
#define OTHER_TEMP 273.0f

__global__ void jacobi_kernel(float* input, float* output, int rows, int* diff) {
    int i = blockIdx.y * blockDim.y + threadIdx.y + 1;
    int j = blockIdx.x * blockDim.x + threadIdx.x + 1;

    if (i >= 1 && i < rows - 1 && j >= 1 && j < N - 1) {
        float new_val = 0.25f * (
            input[(i - 1) * N + j] + input[(i + 1) * N + j] +
            input[i * N + (j - 1)] + input[i * N + (j + 1)]
        );
        output[i * N + j] = new_val;
        float local_diff = fabsf(new_val - input[i * N + j]);
        int int_val;
        memcpy(&int_val, &local_diff, sizeof(float));
        atomicMax(diff, int_val);
    }
}

__global__ void set_row(float* grid, int row, float value) {
    int j = blockIdx.x * blockDim.x + threadIdx.x;
    if (j < N)
        grid[row * N + j] = value;
}

__global__ void set_col(float* grid, int col, int rows, float value) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < rows)
        grid[i * N + col] = value;
}

void initialize(float* grid) {
    #pragma omp parallel for collapse(2)
    for (int i = 0; i < N; ++i)
        for (int j = 0; j < N; ++j)
            grid[i * N + j] = (i == 0) ? TOP_TEMP : OTHER_TEMP;
}

void checkCuda(cudaError_t err, const char* msg) {
    if (err != cudaSuccess) {
        fprintf(stderr, "CUDA Error: %s: %s\n", msg, cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }
}

int main() {
    float *h_grid = (float*)malloc(N * N * sizeof(float));
    float *h_result = (float*)malloc(N * N * sizeof(float));
    initialize(h_grid);

    float *d_in[2], *d_out[2];
    int* d_diff[2];
    int rows = HALF + 2;

    for (int dev = 0; dev < 2; ++dev) {
        checkCuda(cudaSetDevice(dev), "set device");
        checkCuda(cudaMalloc(&d_in[dev], rows * N * sizeof(float)), "malloc d_in");
        checkCuda(cudaMalloc(&d_out[dev], rows * N * sizeof(float)), "malloc d_out");
        checkCuda(cudaMalloc(&d_diff[dev], sizeof(int)), "malloc d_diff");
    }

    checkCuda(cudaSetDevice(0), "set device 0");
    checkCuda(cudaMemcpy(d_in[0], h_grid, (HALF + 1) * N * sizeof(float), cudaMemcpyHostToDevice), "copy to device 0");

    checkCuda(cudaSetDevice(1), "set device 1");
    checkCuda(cudaMemcpy(d_in[1] + N, h_grid + HALF * N, (HALF + 1) * N * sizeof(float), cudaMemcpyHostToDevice), "copy to device 1");

    dim3 threads(4, 4);
    dim3 blocks(N / threads.x, HALF / threads.y);
    dim3 tpb(256);
    dim3 b_row((N + tpb.x - 1) / tpb.x);
    dim3 b_col((rows + tpb.x - 1) / tpb.x);

    printf("block size : %d * %d\n" , 4 , 4) ;
    int iter = 0;
    float max_diff = 0;

    cudaDeviceEnablePeerAccess(0, 0);
    cudaDeviceEnablePeerAccess(1, 0);

    auto t1 = std::chrono::high_resolution_clock::now();

    do {
        for (int dev = 0; dev < 2; ++dev) {
            checkCuda(cudaSetDevice(dev), "set device in loop");
            int zero;
            float zero_f = 0.0f;
            memcpy(&zero, &zero_f, sizeof(float));
            checkCuda(cudaMemcpy(d_diff[dev], &zero, sizeof(int), cudaMemcpyHostToDevice), "reset diff");
            jacobi_kernel<<<blocks, threads>>>(d_in[dev], d_out[dev], rows, d_diff[dev]);
        }

        checkCuda(cudaMemcpyPeer(d_in[0] + (HALF) * N, 0, d_out[1] + N, 1, N * sizeof(float)), "sync GPU1 to GPU0");
        checkCuda(cudaMemcpyPeer(d_in[1], 1, d_out[0] + (HALF - 1) * N, 0, N * sizeof(float)), "sync GPU0 to GPU1");

        for (int dev = 0; dev < 2; ++dev) {
            checkCuda(cudaSetDevice(dev), "swap device");
            std::swap(d_in[dev], d_out[dev]);
        }

        checkCuda(cudaSetDevice(0), "set device for GPU 0 boundaries");
        set_row<<<b_row, tpb>>>(d_in[0], 0, TOP_TEMP);
	set_row<<<b_row, tpb>>>(d_in[0], rows - 1, OTHER_TEMP);
        set_col<<<b_col, tpb>>>(d_in[0], 0, rows, OTHER_TEMP);
        set_col<<<b_col, tpb>>>(d_in[0], N - 1, rows, OTHER_TEMP);

        checkCuda(cudaSetDevice(1), "set device for GPU 1 boundaries");
	set_row<<<b_row, tpb>>>(d_in[1], 0, OTHER_TEMP);
        set_row<<<b_row, tpb>>>(d_in[1], rows - 1, OTHER_TEMP);
        set_col<<<b_col, tpb>>>(d_in[1], 0, rows, OTHER_TEMP);
        set_col<<<b_col, tpb>>>(d_in[1], N - 1, rows, OTHER_TEMP);

        int diff0_raw, diff1_raw;
        checkCuda(cudaMemcpy(&diff0_raw, d_diff[0], sizeof(int), cudaMemcpyDeviceToHost), "copy diff0");
        checkCuda(cudaMemcpy(&diff1_raw, d_diff[1], sizeof(int), cudaMemcpyDeviceToHost), "copy diff1");

        float diff0, diff1;
        memcpy(&diff0, &diff0_raw, sizeof(float));
        memcpy(&diff1, &diff1_raw, sizeof(float));
        max_diff = fmaxf(diff0, diff1);

        iter++;
    } while (max_diff > TOLERANCE && iter < MAX_ITER);

    auto t2 = std::chrono::high_resolution_clock::now();

    checkCuda(cudaSetDevice(0), "final copy GPU 0");
    checkCuda(cudaMemcpy(h_result, d_in[0], HALF * N * sizeof(float), cudaMemcpyDeviceToHost), "copy back GPU 0");

    checkCuda(cudaSetDevice(1), "final copy GPU 1");
    checkCuda(cudaMemcpy(h_result + HALF * N, d_in[1] + N, HALF * N * sizeof(float), cudaMemcpyDeviceToHost), "copy back GPU 1");

    printf("Multi-GPU Jacobi completed in %d iterations\n", iter);
    printf("Multi-GPU Time: %.3f sec\n", std::chrono::duration<double>(t2 - t1).count());

    float min_temp = h_result[0];
    #pragma omp parallel for reduction(min:min_temp)
    for (int i = 1; i < N * N; ++i)
        if (h_result[i] < min_temp) min_temp = h_result[i];
    printf("Min temperature in output: %f\n", min_temp);

    FILE* fp = fopen("gpu_multigpu_output.txt", "w");
    for (int i = 0; i < N * N; ++i)
        fprintf(fp, "%f\n", h_result[i]);
    fclose(fp);

    for (int dev = 0; dev < 2; ++dev) {
        checkCuda(cudaSetDevice(dev), "final free device");
        cudaFree(d_in[dev]);
        cudaFree(d_out[dev]);
        cudaFree(d_diff[dev]);
    }

    free(h_grid);
    free(h_result);
    return 0;
}

