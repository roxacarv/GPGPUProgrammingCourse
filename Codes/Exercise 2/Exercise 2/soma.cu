
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <conio.h>

#define N 10

__global__ void add(int *a, int *b, int *c) {
	int tid = blockIdx.x;
	if (tid < N)
		c[tid] = a[tid] + b[tid];
}

int main(void) {
	
	int a[N], b[N], c[N], userInput, numToAdd;
	int *dev_a, *dev_b, *dev_c;
	srand((unsigned)time(NULL));
	
	cudaMalloc((void**)&dev_a, N * sizeof(int));
	cudaMalloc((void**)&dev_b, N * sizeof(int));
	cudaMalloc((void**)&dev_c, N * sizeof(int));
	
	//receive input from user
	printf("Choose the numbers for the arrays\n");
	printf("1 to fill automatically or 2 to fill manually: ");
	scanf("%d", &userInput);
	
	//process user input
	if (userInput == 1) {
		for (int i = 0; i < N; i++) {
			a[i] = rand() % 20;
			b[i] = rand() % 10;
		}
	} else {
		for (int i = 0; i < N; i++) {
			printf("Choose a number %d for A: ", i+1);
			scanf("%d", &numToAdd);
			a[i] = numToAdd;
			printf("Choose a number %d for B: ", i+1);
			scanf("%d", &numToAdd);
			b[i] = numToAdd;
		}
	}

	cudaMemcpy(dev_a, a, N * sizeof(int), cudaMemcpyHostToDevice);
	cudaMemcpy(dev_b, b, N * sizeof(int), cudaMemcpyHostToDevice);

	add<<<N, 1 >>>(dev_a, dev_b, dev_c);
	
	cudaMemcpy(c, dev_c, N * sizeof(int), cudaMemcpyDeviceToHost);

	for (int i = 0; i < N; i++)
		printf("%d + %d = %d\n", a[i], b[i], c[i]);

	getch();

	//free unused memory
	cudaFree(dev_a);
	cudaFree(dev_b);
	cudaFree(dev_c);
}
