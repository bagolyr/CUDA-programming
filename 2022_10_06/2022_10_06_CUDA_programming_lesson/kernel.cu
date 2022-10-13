
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <stdio.h>

#include <iostream>
#include <stdlib.h>

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

	histogram_CPU();
	histogram_GPU();
	return 0;
}