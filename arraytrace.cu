// Matrix Trace Computation using GPU
// compile with:
//
// nvcc -arch=compute_61 -code=sm_61,sm_61 -O3 -m64 -o arraytrace arraytrace.cu

#include <stdio.h>
#include <stdlib.h>

// Variables
float* h_Matrix;  // host matrix
float* h_C;       // host partial sums
float* d_Matrix;  // device matrix
float* d_C;       // device partial sums

// Functions
void RandomInit(float*, int);

// Device code
__global__ void TraceReduction(const float* Matrix, float* C, int N)
{
    extern __shared__ float cache[];

    int tid = threadIdx.x;
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    float temp = 0.0f;

    while (idx < N) {
        temp += Matrix[idx * N + idx]; // access diagonal elements
        idx += blockDim.x * gridDim.x;
    }

    cache[tid] = temp;
    __syncthreads();

    // Reduction
    int ib = blockDim.x / 2;
    while (ib != 0) {
        if (tid < ib)
            cache[tid] += cache[tid + ib];
        __syncthreads();
        ib /= 2;
    }

    if (tid == 0)
        C[blockIdx.x] = cache[0];
}

// Host code
int main(void)
{
    int gid;
    cudaError_t err = cudaSuccess;

    printf("Enter the GPU ID: ");
    scanf("%d", &gid);
    printf("%d\n", gid);

    err = cudaSetDevice(gid);
    if (err != cudaSuccess) {
        printf("!!! Cannot select GPU with device ID = %d\n", gid);
        exit(1);
    }
    printf("Set GPU with device ID = %d\n", gid);

    int N;
    printf("Enter the matrix size N (N x N): ");
    scanf("%d", &N);
    printf("%d\n", N);

    int threadsPerBlock;
    printf("Enter the number (2^m) of threads per block: ");
    scanf("%d", &threadsPerBlock);
    printf("%d\n", threadsPerBlock);

    if (threadsPerBlock > 1024) {
        printf("The number of threads per block must be less than 1024!\n");
        exit(0);
    }

    int blocksPerGrid;
    printf("Enter the number of blocks per grid: ");
    scanf("%d", &blocksPerGrid);
    printf("%d\n", blocksPerGrid);

    if (blocksPerGrid > 2147483647) {
        printf("The number of blocks must be less than 2147483647!\n");
        exit(0);
    }

    int size = N * N * sizeof(float);
    int sb = blocksPerGrid * sizeof(float);

    h_Matrix = (float*)malloc(size);
    h_C = (float*)malloc(sb);

    RandomInit(h_Matrix, N * N);

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    cudaEventRecord(start, 0);
    cudaMalloc((void**)&d_Matrix, size);
    cudaMalloc((void**)&d_C, sb);

    cudaMemcpy(d_Matrix, h_Matrix, size, cudaMemcpyHostToDevice);
    cudaEventRecord(stop, 0);
    cudaEventSynchronize(stop);

    float Intime;
    cudaEventElapsedTime(&Intime, start, stop);
    printf("Input time for GPU: %f (ms)\n", Intime);

    cudaEventRecord(start, 0);
    int sm = threadsPerBlock * sizeof(float);
    TraceReduction<<<blocksPerGrid, threadsPerBlock, sm>>>(d_Matrix, d_C, N);
    cudaEventRecord(stop, 0);
    cudaEventSynchronize(stop);

    float gputime;
    cudaEventElapsedTime(&gputime, start, stop);
    printf("Processing time for GPU: %f (ms)\n", gputime);
    printf("GPU Gflops: %f\n", N / (1000000.0 * gputime));

    cudaEventRecord(start, 0);
    cudaMemcpy(h_C, d_C, sb, cudaMemcpyDeviceToHost);
    cudaFree(d_Matrix);
    cudaFree(d_C);

    double h_G = 0.0;
    for (int i = 0; i < blocksPerGrid; i++)
        h_G += (double)h_C[i];
    cudaEventRecord(stop, 0);
    cudaEventSynchronize(stop);

    float Outime;
    cudaEventElapsedTime(&Outime, start, stop);
    printf("Output time for GPU: %f (ms)\n", Outime);

    float gputime_tot = Intime + gputime + Outime;
    printf("Total time for GPU: %f (ms)\n", gputime_tot);

    cudaEventRecord(start, 0);
    double h_D = 0.0;
    for (int i = 0; i < N; i++)
        h_D += (double)h_Matrix[i * N + i];
    cudaEventRecord(stop, 0);
    cudaEventSynchronize(stop);

    float cputime;
    cudaEventElapsedTime(&cputime, start, stop);
    printf("Processing time for CPU: %f (ms)\n", cputime);
    printf("CPU Gflops: %f\n", N / (1000000.0 * cputime));
    printf("Speed up of GPU = %f\n", cputime / gputime_tot);

    cudaEventDestroy(start);
    cudaEventDestroy(stop);

    printf("Check result:\n");
    double diff = abs((h_D - h_G) / h_D);
    printf("|(h_G - h_D)/h_D|=%20.15e\n", diff);
    printf("h_G =%20.15e\n", h_G);
    printf("h_D =%20.15e\n", h_D);
    printf("\n");

    free(h_Matrix);
    free(h_C);

    cudaDeviceReset();
}

void RandomInit(float* data, int n)
{
    for (int i = 0; i < n; ++i)
        data[i] = 2.0f * rand() / (float)RAND_MAX - 1.0f;
    // data[i] = 1.0f; // for debugging
}

