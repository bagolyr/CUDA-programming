
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <stdio.h>

#include <iostream>
#include <stdlib.h>

#define CFG_MATRIX_MULT_SHARED_MEM false

// Thread block size
#define BLOCK_SIZE 16

#if(CFG_MATRIX_MULT_SHARED_MEM == false)
// Matrices are stored in row-major order:
// M(row, col) = *(M.elements + row * M.width + col)
typedef struct {
	int width;
	int height;
	float* elements;
} Matrix;

// Forward declaration of the matrix multiplication kernel
__global__ void MatMulKernel(const Matrix, const Matrix, Matrix);

// Matrix multiplication - Host code
// Matrix dimensions are assumed to be multiples of BLOCK_SIZE
void MatMul(const Matrix A, const Matrix B, Matrix C)
{
	// Load A and B to device memory
	Matrix d_A;
	d_A.width = A.width; d_A.height = A.height;
	size_t size = A.width * A.height * sizeof(float);
	cudaMalloc(&d_A.elements, size);
	cudaMemcpy(d_A.elements, A.elements, size,
		cudaMemcpyHostToDevice);
	Matrix d_B;
	d_B.width = B.width; d_B.height = B.height;
	size = B.width * B.height * sizeof(float);
	cudaMalloc(&d_B.elements, size);
	cudaMemcpy(d_B.elements, B.elements, size,
		cudaMemcpyHostToDevice);

	// Allocate C in device memory
	Matrix d_C;
	d_C.width = C.width; d_C.height = C.height;
	size = C.width * C.height * sizeof(float);
	cudaMalloc(&d_C.elements, size);

	// Invoke kernel
	dim3 dimBlock(BLOCK_SIZE, BLOCK_SIZE);
	dim3 dimGrid(B.width / dimBlock.x, A.height / dimBlock.y);
	MatMulKernel << <dimGrid, dimBlock >> > (d_A, d_B, d_C);

	// Read C from device memory
	cudaMemcpy(C.elements, d_C.elements, size,
		cudaMemcpyDeviceToHost);

	// Free device memory
	cudaFree(d_A.elements);
	cudaFree(d_B.elements);
	cudaFree(d_C.elements);
}

// Matrix multiplication kernel called by MatMul()
__global__ void MatMulKernel(Matrix A, Matrix B, Matrix C)
{
	// Each thread computes one element of C
	// by accumulating results into Cvalue
	float Cvalue = 0;
	int row = blockIdx.y * blockDim.y + threadIdx.y;
	int col = blockIdx.x * blockDim.x + threadIdx.x;
	for (int e = 0; e < A.width; ++e)
		Cvalue += A.elements[row * A.width + e]
		* B.elements[e * B.width + col];
	C.elements[row * C.width + col] = Cvalue;
}
#else
// Matrices are stored in row-major order:
// M(row, col) = *(M.elements + row * M.stride + col)
typedef struct {
	int width;
	int height;
	int stride;
	float* elements;
} Matrix;

// Get a matrix element
__device__ float GetElement(const Matrix A, int row, int col)
{
	return A.elements[row * A.stride + col];
}

// Set a matrix element
__device__ void SetElement(Matrix A, int row, int col,
	float value)
{
	A.elements[row * A.stride + col] = value;
}

// Get the BLOCK_SIZExBLOCK_SIZE sub-matrix Asub of A that is
// located col sub-matrices to the right and row sub-matrices down
// from the upper-left corner of A
__device__ Matrix GetSubMatrix(Matrix A, int row, int col)
{
	Matrix Asub;
	Asub.width = BLOCK_SIZE;
	Asub.height = BLOCK_SIZE;
	Asub.stride = A.stride;
	Asub.elements = &A.elements[A.stride * BLOCK_SIZE * row
		+ BLOCK_SIZE * col];
	return Asub;
}

// Thread block size
#define BLOCK_SIZE 16

// Forward declaration of the matrix multiplication kernel
__global__ void MatMulKernel(const Matrix, const Matrix, Matrix);

// Matrix multiplication - Host code
// Matrix dimensions are assumed to be multiples of BLOCK_SIZE
void MatMul(const Matrix A, const Matrix B, Matrix C)
{
	// Load A and B to device memory
	Matrix d_A;
	d_A.width = d_A.stride = A.width; d_A.height = A.height;
	size_t size = A.width * A.height * sizeof(float);
	cudaMalloc(&d_A.elements, size);
	cudaMemcpy(d_A.elements, A.elements, size,
		cudaMemcpyHostToDevice);
	Matrix d_B;
	d_B.width = d_B.stride = B.width; d_B.height = B.height;
	size = B.width * B.height * sizeof(float);
	cudaMalloc(&d_B.elements, size);
	cudaMemcpy(d_B.elements, B.elements, size,
		cudaMemcpyHostToDevice);

	// Allocate C in device memory
	Matrix d_C;
	d_C.width = d_C.stride = C.width; d_C.height = C.height;
	size = C.width * C.height * sizeof(float);
	cudaMalloc(&d_C.elements, size);

	// Invoke kernel
	dim3 dimBlock(BLOCK_SIZE, BLOCK_SIZE);
	dim3 dimGrid(B.width / dimBlock.x, A.height / dimBlock.y);
	MatMulKernel << <dimGrid, dimBlock >> > (d_A, d_B, d_C);

	// Read C from device memory
	cudaMemcpy(C.elements, d_C.elements, size,
		cudaMemcpyDeviceToHost);

	// Free device memory
	cudaFree(d_A.elements);
	cudaFree(d_B.elements);
	cudaFree(d_C.elements);
}

// Matrix multiplication kernel called by MatMul()
__global__ void MatMulKernel(Matrix A, Matrix B, Matrix C)
{
	// Block row and column
	int blockRow = blockIdx.y;
	int blockCol = blockIdx.x;

	// Each thread block computes one sub-matrix Csub of C
	Matrix Csub = GetSubMatrix(C, blockRow, blockCol);

	// Each thread computes one element of Csub
	// by accumulating results into Cvalue
	float Cvalue = 0;

	// Thread row and column within Csub
	int row = threadIdx.y;
	int col = threadIdx.x;

	// Loop over all the sub-matrices of A and B that are
	// required to compute Csub
	// Multiply each pair of sub-matrices together
	// and accumulate the results
	for (int m = 0; m < (A.width / BLOCK_SIZE); ++m) {

		// Get sub-matrix Asub of A
		Matrix Asub = GetSubMatrix(A, blockRow, m);

		// Get sub-matrix Bsub of B
		Matrix Bsub = GetSubMatrix(B, m, blockCol);

		// Shared memory used to store Asub and Bsub respectively
		__shared__ float As[BLOCK_SIZE][BLOCK_SIZE];
		__shared__ float Bs[BLOCK_SIZE][BLOCK_SIZE];

		// Load Asub and Bsub from device memory to shared memory
		// Each thread loads one element of each sub-matrix
		As[row][col] = GetElement(Asub, row, col);
		Bs[row][col] = GetElement(Bsub, row, col);

		// Synchronize to make sure the sub-matrices are loaded
		// before starting the computation
		__syncthreads();
		// Multiply Asub and Bsub together
		for (int e = 0; e < BLOCK_SIZE; ++e)
			Cvalue += As[row][e] * Bs[e][col];

		// Synchronize to make sure that the preceding
		// computation is done before loading two new
		// sub-matrices of A and B in the next iteration
		__syncthreads();
	}

	// Write Csub to device memory
	// Each thread writes one element
	SetElement(Csub, row, col, Cvalue);
}
#endif

__global__ void histogram_kernel(int* d_a, int* d_counters, int N, int K) {
	// compute global thread index
	int tid = blockIdx.x * blockDim.x + threadIdx.x;
	// check indexes
	if (tid >= N)
		return;

	int idx = d_a[tid];
	if (idx >= 0 && idx < K)
		atomicAdd(&d_counters[idx], 1); // 2nd parameter is the value what we would like to add -> this is to avoid conflict
}

__global__ void histogram_kernel_sharedmem(int* d_a, int* d_counters, int N, int K) {
	// compute global thread index
	int tid = blockIdx.x * blockDim.x + threadIdx.x;
	int i = threadIdx.x; //index in the block (0 - 255)

	//__shared__ int sh_counters[100];
	extern __shared__ int sh_counters[];

	if (i < K)
		sh_counters[i] = d_counters[i];
	__syncthreads();	// barrier synchronisation for all threads in the block

	// check indexes
	if (tid >= N)
		return;

	int idx = d_a[tid];
	if (idx >= 0 && idx < K)
		atomicAdd(&sh_counters[idx], 1); // 2nd parameter is the value what we would like to add -> this is to avoid conflict

	__syncthreads();   // barrier synchronisation for all threads in the block
	// save and sum local histograms to device memory
	if (i < K)
		atomicAdd(&d_counters[i], sh_counters[i]);
}

void histogram_CPU() {
	const int N = 1024;
	int a[N];
	int counters[100];
	// init
	for (int i = 0; i < N; i++) {
		a[i] = (int)(99 * ((double)rand()) / RAND_MAX);
	}
	for (int i = 0; i < 100; i++) {
		counters[i] = 0;
	}
	// compute histogram
	for (int i = 0; i < N; i++) {
		int idx = a[i];
		counters[idx]++;
	}
	for (int i = 0; i < 10; i++) {
		std::cout << "hist " << i << " : " << counters[i] << "\n";
	}
}

void histogram_GPU() {
	const int N = 1024;
	int a[N];
	int counters[100];
	// init host variables
	for (int i = 0; i < N; i++) {
		a[i] = (int)(99 * ((double)rand()) / RAND_MAX);
	}
	for (int i = 0; i < 100; i++) {
		counters[i] = 0;
	}

	// declare device variables
	int* d_a;
	int* d_counters;

	// allocate (global) memory on the device
	cudaMalloc((void**)&d_a, N * sizeof(int));
	cudaMalloc((void**)&d_counters, 100 * sizeof(int));

	// copy host data to device
	cudaMemcpy(d_a, a, N * sizeof(int), cudaMemcpyHostToDevice);
	cudaMemcpy(d_counters, counters, 100 * sizeof(int), cudaMemcpyHostToDevice);

	// compute histogram on the device
	int block_size = 256;
	int grid_size = N / block_size;
	//histogram_kernel<<<grid_size, block_size>>>(d_a, d_counters, N, 100);
	histogram_kernel_sharedmem << <grid_size, block_size, 100 * sizeof(int), 0 >> > (d_a, d_counters, N, 100); // sizeof shared dynamic memory and default stream
	//histogram_kernel_sharedmem << <grid_size, block_size >> > (d_a, d_counters, N, 100);

	// copy the result back from device
	cudaMemcpy(counters, d_counters, 100 * sizeof(int), cudaMemcpyDeviceToHost);
	// print
	for (int i = 0; i < 10; i++) {
		std::cout << "hist " << i << " : " << counters[i] << "\n";
	}
}

int main() {
	
	int size = 256;

	for (int j = 0; j < 5; j++) {

		Matrix A, B, C;
		A.width = A.height = size; A.elements = (float*)malloc(sizeof(float) * size * size);
		B.width = B.height = size; B.elements = (float*)malloc(sizeof(float) * size * size);
		C.width = C.height = size; C.elements = (float*)malloc(sizeof(float) * size * size);
		float toggle = 1.0;
		for (int i = 0; i < size * size; i++) {
			A.elements[i] = toggle;
			B.elements[i] = toggle;
			C.elements[i] = 0;
			if (i % 2 == 0)
				toggle = 1.0;
			else
				toggle = 2.0;
		}
		time_t t; time(&t);
		char str[26]; ctime_s(str, 26, &t);
		printf("- size: %d - ", size);
		clock_t begin = clock();
		/*
		Runtimes without shared memory:
		- size: 256 - 220 ms
		- size: 512 - 23 ms
		- size: 1024 - 158 ms
		- size: 2048 - 1117 ms
		- size: 4096 - 9642 ms (sreen shutdown)
		Runtimes with shared memory:
		- size: 256 - 225 ms
		- size: 512 - 27 ms
		- size: 1024 - 196 ms
		- size: 2048 - 2535 ms
		- size: 4096 - 7460 ms
		*/
		MatMul(A, B, C);

		clock_t end = clock();
		double time_spent = (double)(end - begin) / CLOCKS_PER_SEC;
		printf("%.0f ms\n", time_spent * 1000);

		free(A.elements); A.elements = NULL;
		free(B.elements); B.elements = NULL;
		free(C.elements); C.elements = NULL;
		size *= 2;
	}

	//histogram_CPU();
	//histogram_GPU();

	return 0;
}