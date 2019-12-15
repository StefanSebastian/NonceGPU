
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <string.h>
#include <sys/timeb.h>
#include <curand.h>
#include <curand_kernel.h>

#include "config.h"
#include "util.h"

int found = 0;
char* sol;


__device__ void copy_str(char* dest, char* src) {
	int i = 0;
	while (src[i] != '\0') {
		dest[i] = src[i];
		i++;
	}
	dest[i] = '\0';
}

__device__ int str_len(char* str) {
	int size = 0;
	while (str[size] != '\0') {
		size++;
	}
	return size;
}

__device__ int get_random_char(curandState* state) {
	int idx = threadIdx.x + blockIdx.x * blockDim.x;
	float myrandf = curand_uniform(&(state[idx]));
	myrandf *= (90 - 65 + 0.999999);
	myrandf += 65;
	int myrand = (int)truncf(myrandf);
}

__global__ void nonceKernel(char* d_str, char* d_pattern, char* d_result, int* found, int* maxiter, char* d_strings, curandState* state) {
	if (*found == 1) {
		return;
	}

	int idx = threadIdx.x + blockIdx.x * blockDim.x;
	char* new_str = d_strings + idx * *maxiter;
	copy_str(new_str, d_str);
	int new_sz = str_len(new_str);

	for (int i = 0; i < *maxiter; i++) {
		if (*found == 1) {
			break;
		}

		new_str[new_sz] = 'A';//get_random_char(state);
		new_sz++;
		new_str[new_sz] = '\0';
	}

	//copy_str(d_result, new_str);
	if (idx == 1) {
		for (int i = *maxiter; i < *maxiter * 2; i++) {
			d_result[i] = d_strings[i];
		}
	}
}

__global__ void setup_kernel(curandState *state) {
	int idx = threadIdx.x + blockDim.x * blockIdx.x;
	curand_init(1234, idx, 0, &state[idx]);
}


cudaError_t do_iteration() {
	char* d_input;
	char* d_pattern;
	char* d_result;
	int* d_found;
	int* d_maxiter;
	char* d_strings;
	curandState *d_state;
	cudaError_t cudaStatus;

	char* input = (char*)malloc(MAX_ITERATION );
	strcpy(input, INPUT_STRING);
	char* pattern = (char*)malloc(MAX_ITERATION);
	strcpy(pattern, PATTERN);

	// choose GPU
	cudaStatus = cudaSetDevice(0);
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaSetDevice failed!  Do you have a CUDA-capable GPU installed?");
		goto Error;
	}

	// allocate buffers
	cudaStatus = cudaMalloc((void **)&d_found, sizeof(int));
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMalloc failed!");
		goto Error;
	}

	cudaStatus = cudaMalloc((void **)&d_input, MAX_ITERATION * sizeof(char));
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMalloc failed!");
		goto Error;
	}

	cudaStatus = cudaMalloc((void **)&d_pattern, MAX_ITERATION * sizeof(char));
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMalloc failed!");
		goto Error;
	}

	cudaStatus = cudaMalloc((void **)&d_result, MAX_ITERATION * sizeof(char));
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMalloc failed!");
		goto Error;
	}

	cudaStatus = cudaMalloc((void **)&d_maxiter, sizeof(int));
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMalloc failed!");
		goto Error;
	}

	cudaStatus = cudaMalloc((void **)&d_strings, (BLOCKS * THREADS_PER_BLOCK * MAX_ITERATION + 1) * sizeof(char));
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMalloc failed!");
		goto Error;
	}

	cudaStatus = cudaMalloc((void **)&d_state, sizeof(curandState));
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMalloc failed!");
		goto Error;
	}

	// copy to device
	cudaStatus = cudaMemcpy(d_input, input, MAX_ITERATION * sizeof(char), cudaMemcpyHostToDevice);
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMemcpy1 failed!");
		goto Error;
	}

	cudaStatus = cudaMemcpy(d_pattern, pattern, MAX_ITERATION * sizeof(char), cudaMemcpyHostToDevice);
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMemcpy2 failed!");
		goto Error;
	}

	int found = 0;
	cudaStatus = cudaMemcpy(d_found, &found, sizeof(int), cudaMemcpyHostToDevice);
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMemcpyplm failed!");
		goto Error;
	}

	int maxiter = MAX_ITERATION;
	cudaStatus = cudaMemcpy(d_maxiter, &maxiter, sizeof(int), cudaMemcpyHostToDevice);
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMemcpyplm failed!");
		goto Error;
	}

	// launch kernel 
	setup_kernel << <BLOCKS, THREADS_PER_BLOCK >> > (d_state);
	nonceKernel << < BLOCKS, THREADS_PER_BLOCK >> > (d_input, d_pattern, d_result, d_found, d_maxiter, d_strings, d_state);
	cudaStatus = cudaGetLastError();
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "fractalKernel launch failed: %s\n", cudaGetErrorString(cudaStatus));
		goto Error;
	}

	// wait for kernel to finish
	cudaStatus = cudaDeviceSynchronize();
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaDeviceSynchronize returned error code %d after launching fractalKernel!\n", cudaStatus);
		goto Error;
	}

	// copy result back to host 
	cudaStatus = cudaMemcpy(sol, d_result, MAX_ITERATION * sizeof(char), cudaMemcpyDeviceToHost);
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMemcpy res failed!");
		goto Error;
	}

Error:
	free(input);
	free(pattern);
	cudaFree(d_maxiter);
	cudaFree(d_input);
	cudaFree(d_pattern);
	cudaFree(d_result);
	cudaFree(d_found);
	cudaFree(d_strings);
	cudaFree(d_state);
	return cudaStatus;
}

int main()
{
	double times[REPEAT];
	struct timeb start, end;

	for (int r = 0; r < REPEAT; r++) {
		ftime(&start);

		sol = (char*)malloc(MAX_ITERATION);
		do_iteration();

		ftime(&end);
		times[r] = end.time = start.time + ((double)end.millitm - (double)start.millitm) / 1000.0;
		progress("gpu", r, times[r]);
	}

	report("gpu", times);
	FILE* pfile = fopen(OUTPUT, "w");
	fprintf(pfile, "%s", sol);
	fclose(pfile);

	printf("result : %s\n", sol);

    return 0;
}

int maintest() {
	char* dest = (char*)malloc(MAX_ITERATION);
	char* src = (char*)malloc(MAX_ITERATION);
	strcpy(src, INPUT_STRING);
	int i = 0;
	while (src[i] != '\0') {
		dest[i] = src[i];
		i++;
	}
	dest[i] = '\0';

	return 0;
}
