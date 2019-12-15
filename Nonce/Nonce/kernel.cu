
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

#include <stdint.h>

#include "custom_sprintf.c"

int found = 0;
char* sol;


__device__ int sha1digest(uint8_t *digest, char *hexdigest, const uint8_t *data, size_t databytes) {
#define SHA1ROTATELEFT(value, bits) (((value) << (bits)) | ((value) >> (32 - (bits))))

	uint32_t W[80];
	uint32_t H[] = { 0x67452301,
		0xEFCDAB89,
		0x98BADCFE,
		0x10325476,
		0xC3D2E1F0 };
	uint32_t a;
	uint32_t b;
	uint32_t c;
	uint32_t d;
	uint32_t e;
	uint32_t f = 0;
	uint32_t k = 0;

	uint32_t idx;
	uint32_t lidx;
	uint32_t widx;
	uint32_t didx = 0;

	int32_t wcount;
	uint32_t temp;
	uint64_t databits = ((uint64_t)databytes) * 8;
	uint32_t loopcount = (databytes + 8) / 64 + 1;
	uint32_t tailbytes = 64 * loopcount - databytes;
	uint8_t datatail[128] = { 0 };

	if (!digest && !hexdigest)
		return -1;

	if (!data)
		return -1;

	/* Pre-processing of data tail (includes padding to fill out 512-bit chunk):
	Add bit '1' to end of message (big-endian)
	Add 64-bit message length in bits at very end (big-endian) */
	datatail[0] = 0x80;
	datatail[tailbytes - 8] = (uint8_t)(databits >> 56 & 0xFF);
	datatail[tailbytes - 7] = (uint8_t)(databits >> 48 & 0xFF);
	datatail[tailbytes - 6] = (uint8_t)(databits >> 40 & 0xFF);
	datatail[tailbytes - 5] = (uint8_t)(databits >> 32 & 0xFF);
	datatail[tailbytes - 4] = (uint8_t)(databits >> 24 & 0xFF);
	datatail[tailbytes - 3] = (uint8_t)(databits >> 16 & 0xFF);
	datatail[tailbytes - 2] = (uint8_t)(databits >> 8 & 0xFF);
	datatail[tailbytes - 1] = (uint8_t)(databits >> 0 & 0xFF);

	/* Process each 512-bit chunk */
	for (lidx = 0; lidx < loopcount; lidx++)
	{
		/* Compute all elements in W */
		memset(W, 0, 80 * sizeof(uint32_t));

		/* Break 512-bit chunk into sixteen 32-bit, big endian words */
		for (widx = 0; widx <= 15; widx++)
		{
			wcount = 24;

			/* Copy byte-per byte from specified buffer */
			while (didx < databytes && wcount >= 0)
			{
				W[widx] += (((uint32_t)data[didx]) << wcount);
				didx++;
				wcount -= 8;
			}
			/* Fill out W with padding as needed */
			while (wcount >= 0)
			{
				W[widx] += (((uint32_t)datatail[didx - databytes]) << wcount);
				didx++;
				wcount -= 8;
			}
		}

		/* Extend the sixteen 32-bit words into eighty 32-bit words, with potential optimization from:
		"Improving the Performance of the Secure Hash Algorithm (SHA-1)" by Max Locktyukhin */
		for (widx = 16; widx <= 31; widx++)
		{
			W[widx] = SHA1ROTATELEFT((W[widx - 3] ^ W[widx - 8] ^ W[widx - 14] ^ W[widx - 16]), 1);
		}
		for (widx = 32; widx <= 79; widx++)
		{
			W[widx] = SHA1ROTATELEFT((W[widx - 6] ^ W[widx - 16] ^ W[widx - 28] ^ W[widx - 32]), 2);
		}

		/* Main loop */
		a = H[0];
		b = H[1];
		c = H[2];
		d = H[3];
		e = H[4];

		for (idx = 0; idx <= 79; idx++)
		{
			if (idx <= 19)
			{
				f = (b & c) | ((~b) & d);
				k = 0x5A827999;
			}
			else if (idx >= 20 && idx <= 39)
			{
				f = b ^ c ^ d;
				k = 0x6ED9EBA1;
			}
			else if (idx >= 40 && idx <= 59)
			{
				f = (b & c) | (b & d) | (c & d);
				k = 0x8F1BBCDC;
			}
			else if (idx >= 60 && idx <= 79)
			{
				f = b ^ c ^ d;
				k = 0xCA62C1D6;
			}
			temp = SHA1ROTATELEFT(a, 5) + f + e + k + W[idx];
			e = d;
			d = c;
			c = SHA1ROTATELEFT(b, 30);
			b = a;
			a = temp;
		}

		H[0] += a;
		H[1] += b;
		H[2] += c;
		H[3] += d;
		H[4] += e;
	}

	/* Store binary digest in supplied buffer */
	if (digest)
	{
		for (idx = 0; idx < 5; idx++)
		{
			digest[idx * 4 + 0] = (uint8_t)(H[idx] >> 24);
			digest[idx * 4 + 1] = (uint8_t)(H[idx] >> 16);
			digest[idx * 4 + 2] = (uint8_t)(H[idx] >> 8);
			digest[idx * 4 + 3] = (uint8_t)(H[idx]);
		}
	}

	/* Store hex version of digest in supplied buffer */
	if (hexdigest)
	{
		simple_sprintf(hexdigest, "%08x%08x%08x%08x%08x",
			H[0], H[1], H[2], H[3], H[4]);
	}

	return 0;
}



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

__device__ int str_cmp(char string1[], char string2[])
{
	for (int i = 0; ; i++)
	{
		if (string1[i] != string2[i])
		{
			return string1[i] < string2[i] ? -1 : 1;
		}

		if (string1[i] == '\0')
		{
			return 0;
		}
	}
}


__device__ int string_ends_with( char * str,  char * suffix)
{
	int str_leng = str_len(str);
	int suffix_len = str_len(suffix);

	return
		(str_leng >= suffix_len) &&
		(0 == str_cmp(str + (str_leng - suffix_len), suffix));
}

__device__ int get_random_char(curandState* state) {
	int idx = threadIdx.x + blockIdx.x * blockDim.x;
	float myrandf = curand_uniform(&(state[idx]));
	myrandf *= (90 - 65 + 0.999999);
	myrandf += 65;
	int myrand = (int)truncf(myrandf);
}


__device__ int check_str(char* str, char* pattern) {
	uint8_t digest[20]; char hexdigest[41];
	sha1digest(digest, hexdigest, (uint8_t*)str, str_len(str));
	return string_ends_with(hexdigest, pattern);
}

__global__ void nonceKernel(char* d_str, char* d_pattern, char* d_result, int* found, int* maxiter, char* d_strings, curandState* state, int pitch) {
	if (*found == 1) {
		return;
	}

	int idx = threadIdx.x + blockIdx.x * blockDim.x;
	char* new_str = d_strings + idx * pitch;
	copy_str(new_str, d_str);
	int new_sz = str_len(new_str);

	while (new_sz < *maxiter - 1) {
		if (*found == 1) {
			break;
		}

		new_str[new_sz] = get_random_char(state);
		new_sz++;
		new_str[new_sz] = '\0';

		if (check_str(new_str, d_pattern)) {
			*found = 1;
			copy_str(d_result, new_str);
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
	size_t pitch;
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

	cudaStatus = cudaMallocPitch((void **)&d_strings, &pitch, (MAX_ITERATION + 1) * sizeof(char), BLOCKS * THREADS_PER_BLOCK);
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
	nonceKernel << < BLOCKS, THREADS_PER_BLOCK >> > (d_input, d_pattern, d_result, d_found, d_maxiter, d_strings, d_state, pitch);
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

int main1() {
	char* dest = (char*)malloc(MAX_ITERATION);
	dest = "abdefc";

	uint8_t digest[20]; char hexdigest[41];
//	sha1digest(digest, hexdigest, (uint8_t*)dest, strlen(dest));

	printf("hash : %s", hexdigest);

	return 0;
}
