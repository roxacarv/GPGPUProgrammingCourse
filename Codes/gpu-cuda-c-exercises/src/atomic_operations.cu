#include "../common/book.h"
#define SIZE (100*1024*1024)

__global__ void histo_kernel( unsigned char *buffer,
								long size,
								unsigned int *histo ) {
	__shared__ unsigned int temp[256];
	temp[threadIdx.x] = 0;
	__syncthreads();

	int i = threadIdx.x + blockIdx.x * blockDim.x;
	int stride = blockDim.x * gridDim.x;
	while (i < size) {
		atomicAdd( &temp[buffer[i]], 1 );
		i += stride;
	}
	__syncthreads();
}

int main( void ) {
	unsigned char *buffer = (unsigned char*)big_random_block( SIZE );
	cudaEvent_t start, stop;
	cudaEventCreate( &start);
	cudaEventCreate( &stop);
	cudaEventRecord( start, 0);
	unsigned char *dev_buffer;
	unsigned int *dev_histo;
	cudaMalloc( (void**)&dev_buffer, SIZE);
	cudaMemcpy( dev_buffer, buffer, SIZE,
				cudaMemcpyHostToDevice);
	cudaMalloc( (void**)&dev_histo,
				256 * sizeof( int ));
	cudaMemset( dev_histo, 0,
				256 * sizeof( int ));

	unsigned int histo[256];
	cudaMemcpy( histo, dev_histo,
				256 * sizeof( int ),
				cudaMemcpyDeviceToHost);

	cudaEventRecord( stop, 0 );
	cudaEventSynchronize( stop );
	float elapsedTime;
	cudaEventElapsedTime( &elapsedTime,
						start, stop ) );
	printf( "Time to generate: %3.1f ms\n", elapsedTime );

	long histoCount = 0;
	for (int i=0; i<256; i++) {
		histoCount += histo[i];
	}
	printf( "Histogram Sum: %ld\n", histoCount );

	for (int i=0; i<SIZE; i++)
		histo[buffer[i]]--;
	for (int i=0; i<256; i++) {
		if (histo[i] != 0)
			printf( "Failure at %d! Off by %d\n", i, histo[i] );
		}

	cudaEventDestroy( start);
	cudaEventDestroy( stop );
	cudaFree( dev_histo );
	cudaFree( dev_buffer );
	free( buffer );

	cudaDeviceProp prop;
	cudaGetDeviceProperties( &prop, 0 );
	int blocks = prop.multiProcessorCount;
	histo_kernel<<<blocks*2,256>>>( dev_buffer, SIZE, dev_histo );

	return 0;
}

