// compile by : nvcc -Xcompiler -fopenmp -O3 -arch=sm_61 -o multiGPU_vecDot multiGPU_vecDot.cu


#include <stdio.h>
#include <stdlib.h>
#include <omp.h>
#include <cuda_runtime.h>
#include <math.h>

float* h_A;
float* h_B;
float* h_C; // partial sums from each GPU

void RandomInit(float* data, int n) {
    for (int i = 0; i < n; ++i)
        data[i] = 2.0 * rand() / (float)RAND_MAX - 1.0;
}

__global__ void VecDotKernel(const float* A, const float* B, float* C, int N) {
    extern __shared__ float cache[];
    int i = blockDim.x * blockIdx.x + threadIdx.x;
    int cacheIndex = threadIdx.x;
    float temp = 0.0;

    while (i < N) {
        temp += A[i] * B[i];
        i += blockDim.x * gridDim.x;
    }

    cache[cacheIndex] = temp;
    __syncthreads();

    int ib = blockDim.x / 2;
    while (ib != 0) {
        if (cacheIndex < ib)
            cache[cacheIndex] += cache[cacheIndex + ib];
        __syncthreads();
        ib /= 2;
    }

    if (cacheIndex == 0)
        C[blockIdx.x] = cache[0];
}

int main() {
    printf("Vector Dot Product with multiple GPUs\n");

    int NGPU, N;
    printf("Enter the number of GPUs: ");
    scanf("%d", &NGPU);
    printf("%d\n", NGPU);

    int* Dev = (int*)malloc(sizeof(int) * NGPU);
    printf("GPU device number: ");
    for (int i = 0; i < NGPU; ++i) {
        scanf("%d", &Dev[i]);
        printf("%d ", Dev[i]);
        if (getchar() == '\n') break;
    }
    printf("\n");

    N = 40960000 ;

    int threadsPerBlock = 512 ;

    if (threadsPerBlock > 1024) {
        printf("Threads per block must be <= 1024.\n");
        return 1;
    }

    int blocksPerGrid = 256;
    int sm = threadsPerBlock * sizeof(float);
    size_t vec_bytes = (N / NGPU) * sizeof(float);
    size_t part_bytes = blocksPerGrid * sizeof(float);

    printf("threadsPerBlock = %d , blocksPerGrid = %d\n" , threadsPerBlock , blocksPerGrid);

    h_A = (float*)malloc(N * sizeof(float));
    h_B = (float*)malloc(N * sizeof(float));
    h_C = (float*)malloc(NGPU * blocksPerGrid * sizeof(float));

    RandomInit(h_A, N);
    RandomInit(h_B, N);

    omp_set_num_threads(NGPU);

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    float Intime = 0, gputime = 0, Outime = 0;

#pragma omp parallel
    {
        int tid = omp_get_thread_num();
        cudaSetDevice(Dev[tid]);

        float *d_A, *d_B, *d_C;
        cudaMalloc(&d_A, vec_bytes);
        cudaMalloc(&d_B, vec_bytes);
        cudaMalloc(&d_C, part_bytes);

        if (tid == 0) cudaEventRecord(start, 0);

        cudaMemcpy(d_A, h_A + (N / NGPU) * tid, vec_bytes, cudaMemcpyHostToDevice);
        cudaMemcpy(d_B, h_B + (N / NGPU) * tid, vec_bytes, cudaMemcpyHostToDevice);

#pragma omp barrier
        if (tid == 0) {
            cudaEventRecord(stop, 0);
            cudaEventSynchronize(stop);
            cudaEventElapsedTime(&Intime, start, stop);
            printf("Input time: %f (ms)\n", Intime);
            cudaEventRecord(start, 0);
        }

        VecDotKernel<<<blocksPerGrid, threadsPerBlock, sm>>>(d_A, d_B, d_C, N / NGPU);
        cudaDeviceSynchronize();

#pragma omp barrier
        if (tid == 0) {
            cudaEventRecord(stop, 0);
            cudaEventSynchronize(stop);
            cudaEventElapsedTime(&gputime, start, stop);
            printf("GPU processing time: %f (ms)\n", gputime);
            printf("GPU GFLOPS: %f\n", 2.0 * N / (1000000.0 * gputime));
            cudaEventRecord(start, 0);
        }

        cudaMemcpy(h_C + tid * blocksPerGrid, d_C, part_bytes, cudaMemcpyDeviceToHost);
        cudaFree(d_A); cudaFree(d_B); cudaFree(d_C);

#pragma omp barrier
        if (tid == 0) {
            cudaEventRecord(stop, 0);
            cudaEventSynchronize(stop);
            cudaEventElapsedTime(&Outime, start, stop);
            printf("Output time: %f (ms)\n", Outime);
        }
    }

    float total = 0.0;
    for (int g = 0; g < NGPU; ++g) {
        for (int i = 0; i < blocksPerGrid; ++i)
            total += h_C[g * blocksPerGrid + i];
    }

    float gputime_total = Intime + gputime + Outime;
    printf("Total GPU time: %f (ms)\n", gputime_total);

    cudaEventRecord(start, 0);
    double cpu_result = 0.0;
    for (int i = 0; i < N; ++i)
        cpu_result += (double)h_A[i] * h_B[i];
    cudaEventRecord(stop, 0);
    cudaEventSynchronize(stop);
    float cputime;
    cudaEventElapsedTime(&cputime, start, stop);

    printf("CPU time: %f (ms)\n", cputime);
    printf("CPU GFLOPS: %f\n", 2.0 * N / (1000000.0 * cputime));
    printf("Speedup: %f\n", cputime / gputime_total);

    double diff = fabs((cpu_result - total) / cpu_result);
    printf("Relative error = %.15e\n", diff);
    printf("GPU result = %.15e\n", total);
    printf("CPU result = %.15e\n", cpu_result);

    free(h_A); free(h_B); free(h_C);
    cudaEventDestroy(start);
    cudaEventDestroy(stop);

    for (int i = 0; i < NGPU; ++i) {
        cudaSetDevice(i);
        cudaDeviceReset();
    }

    return 0;
}

