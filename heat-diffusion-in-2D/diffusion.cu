// compile by : nvcc -O2 -o diffusion diffusion.cu

#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>
#include <math.h>
#include <chrono>

#define N 1024
#define MAX_ITER 100000
#define TOLERANCE 1e-6

#define TOP_TEMP 400.0f
#define OTHER_TEMP 273.0f

void initialize(float* grid) {
    for (int i = 0; i < N; ++i) {
        for (int j = 0; j < N; ++j) {
            if (i == 0)
                grid[i * N + j] = TOP_TEMP;
            else
                grid[i * N + j] = OTHER_TEMP;
        }
    }
}

void cpu_jacobi(float* grid, float* result) {
    float* prev = (float*)malloc(N * N * sizeof(float));
    memcpy(prev, grid, N * N * sizeof(float));

    int iter = 0;
    float diff;
    do {
        diff = 0.0f;
        for (int i = 1; i < N - 1; ++i) {
            for (int j = 1; j < N - 1; ++j) {
                float new_val = 0.25f * (
                    prev[(i - 1) * N + j] + prev[(i + 1) * N + j] +
                    prev[i * N + (j - 1)] + prev[i * N + (j + 1)]
                );
                diff = fmaxf(diff, fabsf(new_val - prev[i * N + j]));
                result[i * N + j] = new_val;
            }
        }
        float* temp = prev;
        prev = result;
        result = temp;

	for (int i = 0; i < N; ++i) {
           result[0 * N + i] = TOP_TEMP;          // top row
           result[(N - 1) * N + i] = OTHER_TEMP;  // bottom row
           result[i * N + 0] = OTHER_TEMP;        // left col
           result[i * N + (N - 1)] = OTHER_TEMP;  // right col
        }



        iter++;
    } while (diff > TOLERANCE && iter < MAX_ITER);

    memcpy(result, prev, N * N * sizeof(float));
    free(prev);
    printf("CPU completed in %d iterations\n", iter);
}

__global__ void jacobi_kernel(float* input, float* output , float* diff) {
    int i = blockIdx.y * blockDim.y + threadIdx.y;
    int j = blockIdx.x * blockDim.x + threadIdx.x;

    if (i > 0 && i < N - 1 && j > 0 && j < N - 1) {
        float new_val = 0.25f * (
            input[(i - 1) * N + j] + input[(i + 1) * N + j] +
            input[i * N + (j - 1)] + input[i * N + (j + 1)]
        );
        output[i * N + j] = new_val;
        float local_diff = fabsf(new_val - input[i * N + j]);
        atomicMax((int*)diff, __float_as_int(local_diff));
    }
}

void gpu_jacobi(float* h_grid, float* h_result) {
    float *d_in, *d_out, *d_diff;
    cudaMalloc(&d_in, N * N * sizeof(float));
    cudaMalloc(&d_out, N * N * sizeof(float));
    cudaMalloc(&d_diff, sizeof(float));
    cudaMemcpy(d_in, h_grid, N * N * sizeof(float), cudaMemcpyHostToDevice);

    dim3 threads(32,32);
    dim3 blocks(N / threads.x, N / threads.y);
	
    printf("block size(single GPU) %d * %d\n" , 32 , 32) ;	

    int iter = 0;
    float diff;

    do {
        diff = 0.0f;
        cudaMemcpy(d_diff, &diff, sizeof(float), cudaMemcpyHostToDevice);

        jacobi_kernel<<<blocks, threads>>>(d_in, d_out , d_diff);
        cudaDeviceSynchronize();

        cudaMemcpy(&diff, d_diff, sizeof(float), cudaMemcpyDeviceToHost);
        std::swap(d_in, d_out);

        cudaMemcpy(h_grid, d_in, N * N * sizeof(float), cudaMemcpyDeviceToHost);
	for (int i = 0; i < N; ++i) {
    	    h_grid[0 * N + i] = TOP_TEMP;               // Top row
            h_grid[(N - 1) * N + i] = OTHER_TEMP;       // Bottom row
            h_grid[i * N + 0] = OTHER_TEMP;             // Left column
            h_grid[i * N + (N - 1)] = OTHER_TEMP;       // Right column
        }
	cudaMemcpy(d_in, h_grid, N * N * sizeof(float), cudaMemcpyHostToDevice);
        iter++;
    } while (diff > TOLERANCE && iter < MAX_ITER);

    cudaMemcpy(h_result, d_in, N * N * sizeof(float), cudaMemcpyDeviceToHost);
    printf("GPU completed in %d iterations\n", iter);

    cudaFree(d_in); cudaFree(d_out); cudaFree(d_diff);
}

int main() {
    float *grid = (float*)malloc(N * N * sizeof(float));
    float *cpu_result = (float*)malloc(N * N * sizeof(float));
    float *gpu_result = (float*)malloc(N * N * sizeof(float));

    initialize(grid);
    auto t1 = std::chrono::high_resolution_clock::now();
    cpu_jacobi(grid, cpu_result);
    auto t2 = std::chrono::high_resolution_clock::now();
    printf("CPU Time: %.3f sec\n", std::chrono::duration<double>(t2 - t1).count());

    initialize(grid);
    t1 = std::chrono::high_resolution_clock::now();
    gpu_jacobi(grid, gpu_result);
    t2 = std::chrono::high_resolution_clock::now();
    printf("GPU Time: %.3f sec\n", std::chrono::duration<double>(t2 - t1).count());

    FILE* fp = fopen("gpu_output.txt", "w");
    for (int i = 0; i < N * N; ++i)
        fprintf(fp, "%f\n", gpu_result[i]);
    fclose(fp);

    FILE* fp2 = fopen("cpu_output.txt" , "w" ) ;
    for (int i = 0 ; i < N * N ; ++i)
	fprintf(fp2 ,"%f\n", cpu_result[i]) ;
    fclose(fp2) ;


    free(grid); free(cpu_result); free(gpu_result);
    return 0;
}

