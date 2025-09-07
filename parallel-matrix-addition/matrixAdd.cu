// matrix addition c(i , j) = 1 / a(i , j) + 1 / b(i , j)
// compile by : nvcc -arch=compute_61 -code=sm_61,sm_61 -O2 -m64 -o matrixAdd matrixAdd.cu

#include<stdio.h>
#include<stdlib.h>

#define BLOCKSIZE 2

// matrix in host memory
double *h_A , *h_B , *h_C ;

// matrix in device memory
double *d_A , *d_B , *d_C ;


// random initialization
void randomInit (double* matrix , int size)
{
	for(int i = 0 ; i < size ; i++) {
		for(int j = 0 ; j < size ; j++) {
			*(matrix + i * size + j) = 0.1 + rand() / (double) RAND_MAX ;
		}
	}
}

// kernel function : run in device
__global__ void matrixAdd(const double* A , const double* B , double* C , int size ) {
	int i = blockDim.y * blockIdx.y + threadIdx.y ;
	int j = blockDim.x * blockIdx.x + threadIdx.x ;

	if ( i < size && j < size  )
		*(C + i * size + j) = 1.0 / *(A + i * size + j) + 1.0 / *(B + i * size + j) ;

}


// main function : run in CPU (host)

int main(void) 
{
	int gid ;

	// Error state storing
	cudaError_t err = cudaSuccess ;

	printf("Enter the GPU ID : ") ;
	scanf("%d" , &gid) ;
	
	printf("%d\n" , gid) ;
	err = cudaSetDevice(gid) ;
	if (err != cudaSuccess) {
		printf("!!!cannot select GPU with device ID = %d\n" , gid) ;
		exit(1) ;
	}
	printf("Set GPu with device ID = %d\n" , gid) ;


	cudaSetDevice(gid) ;
	printf("Matrix Addition : c(i , j) = 1 / a(i , j) + 1 / b(i , j)\n ") ;
	
	int N ;

	printf("Enter the size of the vectors : ") ;
	scanf("%d" , &N) ;
	
	printf("%d\n" ,N) ;
	
	int size = N * N * sizeof(double) ;
	
	
	// Allocate input vectors h_A , h_B , h_c
	h_A = (double*) malloc(size) ;
	h_B = (double*) malloc(size) ;
	h_C = (double*) malloc(size) ;
	
	// Initialization
	randomInit(h_A , N) ;
	randomInit(h_B , N) ;

	

	// Set the size of blocks and threads 
	dim3 threadsPerBlock(BLOCKSIZE , BLOCKSIZE);
	dim3 blocksPerGrid((N + threadsPerBlock.x - 1) / threadsPerBlock.x,
                   (N + threadsPerBlock.y - 1) / threadsPerBlock.y);

	
	
	// create timer
	cudaEvent_t start ,stop ;
	cudaEventCreate(&start) ;
	cudaEventCreate(&stop) ;


	// start the timer
	cudaEventRecord(start , 0) ;

	// Allocate matrixs  in device memory
	cudaMalloc( (void**)&d_A , size) ;
	cudaMalloc( (void**)&d_B , size ) ;
	cudaMalloc( (void**)&d_C , size ) ;

	// copy matrixs to device memory
	cudaMemcpy( d_A , h_A , size  , cudaMemcpyHostToDevice ) ;
	cudaMemcpy(d_B , h_B , size  , cudaMemcpyHostToDevice) ;


	// stop timer
	cudaEventRecord(stop , 0) ;
	cudaEventSynchronize(stop) ;


	float Intime ;
	cudaEventElapsedTime(&Intime , start , stop) ;
	printf("Input time for GPU : %f (ms) \n" , Intime ) ;

	
	// start timer 
	cudaEventRecord(start , 0) ;

	matrixAdd<<<blocksPerGrid , threadsPerBlock>>> (d_A , d_B , d_C , N ) ;

	err = cudaGetLastError();  // check the GPU running process
	if (err != cudaSuccess) {
    		printf("CUDA Kernel launch failed: %s\n", cudaGetErrorString(err));
    		exit(1);
	}

	cudaDeviceSynchronize() ;

	// stop timer 
	cudaEventRecord(stop , 0) ;
	cudaEventSynchronize(stop) ;

	float gputime ;
	cudaEventElapsedTime( &gputime , start , stop ) ;
	if (gputime > 0) {
   		 printf("GPU Gflops : %f\n", 3 * N * N / (1000000 * gputime));
	}
       	else {
    		printf("GPU Gflops : N/A (gputime = 0)\n");
	}


	printf("GPU Running time : %f\n" , gputime ) ;

	
	// start timer 
	cudaEventRecord(start,0);

	// copy result to host memory
	cudaMemcpy(h_C , d_C , size , cudaMemcpyDeviceToHost) ;

	cudaFree(d_A ) ;
	cudaFree(d_B) ;
	cudaFree(d_C) ;
	
	// stop timer 
	cudaEventRecord(stop,0);
    	cudaEventSynchronize(stop);

	float Outtime ;
	cudaEventElapsedTime( &Outtime , start , stop ) ;
	printf("Output time for GPU: %f (ms) \n",Outtime);


	double gpuTimeTotal = Intime + gputime + Outtime ;
	printf("Total time for GPU : %f (ms) \n" , gpuTimeTotal) ;

	

	// start running CPU 
	cudaEventRecord(start , 0) ;
	double* h_D = (double*) malloc(size) ;
	for(int i = 0 ; i < N ;i++) {
		for(int j = 0 ; j < N ; j++) {
			*(h_D + i * N + j) = 1.0 / *(h_A + i * N + j) + 1.0/ *(h_B + i * N + j) ;
		}
	}

	// stop timer 
	cudaEventRecord(stop , 0) ;
	cudaEventSynchronize(stop) ;

	float cputime ;
	cudaEventElapsedTime(&cputime , start , stop) ;
	
	float gputime_tot = Intime + Outtime + gputime ;
	printf("Processing time for CPU: %f (ms) \n",cputime);
   	printf("CPU Gflops: %f\n",3*N * N  /(1000000.0*cputime));
    	printf("Speed up of GPU = %f\n", cputime/(gputime_tot));

	
	// destroy timer 
	cudaEventDestroy(start) ;
	cudaEventDestroy(stop) ;


	// check result 
	printf("Check result : \n") ;
	double sum = 0 ;
	double diff ;
	for(int i = 0 ; i < N ; i++) {
		for(int j = 0 ; j < N ; j++) {
			diff = fabs( *(h_D + i * N + j) - *(h_C + i * N + j) ) ;
			sum += diff * diff ;
		}
	}

	sum = sqrt(sum) ;
	printf("norm(h_C - h_D) = %20.15e\n\n" , sum ) ;
	cudaDeviceReset() ;

	free(h_A) ;
	free(h_B) ;
	free(h_C) ;
	free(h_D) ;

	return 0 ;
}

