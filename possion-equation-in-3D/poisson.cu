
// compile : nvcc -O2 -o poisson poisson.cu

#include <stdio.h>
#include <stdlib.h>
#include <math.h>

#define IDX(x,y,z,L) ((z)*(L)*(L)+(y)*(L)+(x))

__global__ void jacobi_step(double* new_phi, double* phi, double* rho, int L) {
    int x = blockIdx.x * blockDim.x + threadIdx.x + 1;
    int y = blockIdx.y * blockDim.y + threadIdx.y + 1;
    int z = blockIdx.z * blockDim.z + threadIdx.z + 1;

    if (x < L - 1 && y < L - 1 && z < L - 1) {
        int idx = IDX(x, y, z, L);
        new_phi[idx] = (1.0/6.0) * (
            phi[IDX(x+1, y, z, L)] +
            phi[IDX(x-1, y, z, L)] +
            phi[IDX(x, y+1, z, L)] +
            phi[IDX(x, y-1, z, L)] +
            phi[IDX(x, y, z+1, L)] +
            phi[IDX(x, y, z-1, L)] +
            rho[idx]
        );
    }
}

void initialize(double* rho, double* phi, int L) {
    int center = L / 2;
    for (int z = 0; z < L; ++z) {
        for (int y = 0; y < L; ++y) {
            for (int x = 0; x < L; ++x) {
                int idx = IDX(x, y, z, L);
                phi[idx] = 0.0;
                rho[idx] = (x == center && y == center && z == center) ? 1.0 : 0.0;
            }
        }
    }
}

void save_potential(double* phi, int L) {
    char filename[64];
    sprintf(filename, "potential_L%d.txt", L);
    FILE* f = fopen(filename, "w");
    int cx = L / 2, cy = L / 2, cz = L / 2;
    for (int z = 0; z < L; ++z) {
        for (int y = 0; y < L; ++y) {
            for (int x = 0; x < L; ++x) {
                int idx = IDX(x, y, z, L);
                double r = sqrt((x-cx)*(x-cx) + (y-cy)*(y-cy) + (z-cz)*(z-cz));
                fprintf(f, "%f %f\n", r, phi[idx]);
            }
        }
    }
    fclose(f);
}

int main(int argc, char** argv) {
    int L ;
    scanf("%d" , &L) ;
    printf("L = %d\n" , L) ;
    if (argc > 1) L = atoi(argv[1]);
    int size = L * L * L;
    size_t bytes = size * sizeof(double);

    double *h_phi = (double*)malloc(bytes);
    double *h_rho = (double*)malloc(bytes);

    initialize(h_rho, h_phi, L);

    double *d_phi, *d_new_phi, *d_rho;
    cudaMalloc(&d_phi, bytes);
    cudaMalloc(&d_new_phi, bytes);
    cudaMalloc(&d_rho, bytes);

    cudaMemcpy(d_phi, h_phi, bytes, cudaMemcpyHostToDevice);
    cudaMemcpy(d_rho, h_rho, bytes, cudaMemcpyHostToDevice);

    dim3 block(8, 8, 8);
    dim3 grid((L-2 + block.x - 1)/block.x, (L-2 + block.y - 1)/block.y, (L-2 + block.z - 1)/block.z);

    int max_iter = 10000;
    for (int iter = 0; iter < max_iter; ++iter) {
        jacobi_step<<<grid, block>>>(d_new_phi, d_phi, d_rho, L);
        std::swap(d_phi, d_new_phi);
    }

    cudaMemcpy(h_phi, d_phi, bytes, cudaMemcpyDeviceToHost);
    
    int cx = L / 2, cy = L / 2, cz = L / 2;
    for (int z = 0; z < L; ++z) {
        for (int y = 0; y < L; ++y) {
            for (int x = 0; x < L; ++x) {
                int idx = IDX(x, y, z, L);
                double r = sqrt((x - cx) * (x - cx) + (y - cy) * (y - cy) + (z - cz) * (z - cz));
                printf("%f %f\n", r, h_phi[idx]);
            }
        }
    }


    cudaFree(d_phi);
    cudaFree(d_new_phi);
    cudaFree(d_rho);
    free(h_phi);
    free(h_rho);

    return 0;
}

