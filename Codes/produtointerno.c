#include "../common/book.h"

#define imin(a,b) (a<b?a:b)

const int N = 33 * 1024;
const int threadsPerBlock = 256;
const int blocksPerGrid = imin(32, (N+threadsPerBlock-1) / threadsPerBlock);

__global__ void dot(float *a, float *b, float *c)
{
	__shared__ float cache[threadsPerBlock];
	int tid = threadIdx.x + blockIdx.x * blockDim.x;
	int cacheIndex = threadIdx.x;
	float temp = 0;
	while (tid < N) {
		temp += a[tid] * b[tid];
		tid += blockDim.x * gridDim.x;
	}

	cache[cacheIndex] = temp;
	int i = blockDim.x/2;
	while (i != 0)
	{
		if (cacheIndex < i)
			cache[cacheIndex] += cache[cacheIndex + 1];
		__syncthreads();
		i /= 2;
	} 
}

int main(void)
{
	float *a, *b, c, *partial_c;
	float *dev_a, *dev_b, *dev_partial_c;
	a = new float[N];
	b = new float[N];
	partial_c = new float[blocksPerGrid];

	cudaMalloc((void**)&dev_a, N*sizeof(float));
	cudaMalloc((void**)&dev_b, N*sizeof(float));
	cudaMalloc((void**)&dev_partial_c, blocksPerGrid*sizeof(float));

	for (int i = 0; i < N; i++) {
		a[i] = i;
		b[i] = i * 2;
	}

	cudaMemcpy(dev_a, a, N*sizeof(float), cudaMemcpyHostToDevice);
	cudaMemcpy(dev_b, b, N*sizeof(float), cudaMemcpyHostToDevice);

	dot<<<blocksPerGrid, threadsPerBlock>>>(dev_a, dev_b, dev_partial_c);

	cudaMemcpy(partial_c, dev_partial_c, blocksPerGrid*sizeof(float), cudaMemcpyDeviceToHost);
	c = 0;
	for (int i = 0; i < blocksPerGrid; i++)
		c += partial_c[i];
}