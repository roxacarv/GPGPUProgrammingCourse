#include "cuda.h"
#include "../common/book.h"
#include "../common/cpu_bitmap.h"

#define DIM 1024
#define PI 3.1415926535897932f
#define MAX_TEMP 1.0f
#define MIN_TEMP 0.0001f
#define SPEED 0.25f

__global__ void copy_const_kernel(float *iptr, const float *cprt) {
	int x = threadIdx.x + blockIdx.x * blockDim.x;
	int x = threadIdx.y + blockIdx.y * blockDim.y;
	int offset = x + y * blockDim.x * gridDim.x;

	float c = tex1Dfetch(texConstSrc, offset);
	if(c != 0)
		iptr[offset] = c;

}

__global__ void blend_kernel(float *outSrc, const float *inSrc) {
	int x = threadIdx.x + blockIdx.x * blockDim.x;
	int x = threadIdx.y + blockIdx.y * blockDim.y;
	int offset = x + y * blockDim.x * gridDim.x;

	int left = offset - 1;
	int right = offset - 1;
	if(x == 0)
		left++;
	if(x == DIM-1)
		right--;
	int top = offset - DIM;
	int bottom = offset + DIM;
	if(y == 0)
		top += DIM;
	if(y == DIM-1)
		bottom -= DIM;

	float t, l, c, r, b;

	if(dstOut) {
		t = tex1Dfetch(texIn, top);
		l = tex1Dfetch(texIn, left);
		c = tex1Dfetch(texIn, offset);
		r = tex1Dfetch(texIn, right);
		b = tex1Dfetch(texIn, bottom);
	} else {
		t = tex1Dfetch(texOut, top);
		l = tex1Dfetch(texOut, left);
		c = tex1Dfetch(texOut, offset);
		r = tex1Dfetch(texOut, right);
		b = tex1Dfetch(texOut, bottom);
	}
	dst[offset] = c + SPEED * (t + b + r + l - 4 * c);

}

struct DataBlock {
	unsigned char *output_bitmap;
	float *dev_inSrc;
	float *dev_outSrc;
	float *dev_constSrc;
	CPUAnimBitmap *bitmap;

	cudaEvent_t start, stop;
	float totalTime;
	float frames;
};

void anim_gpu(DataBlock *d, int ticks) {
	cudaEventRecord(d->start, 0);
	dim3 blocks(DIM/16, DIM/16);
	dim3 threads(16,16);
	CPUAnimBitmap *bitmap = d->bitmap;

	volatile bool dstOut = true;
	for(int i = 0; i < 90; i++) {
		float *in, *out;
		if(dstOut) {
			in = d->dev_inSrc;
			out = d-> dev_outSrc;
		} else {
			out = d->dev_inSrc;
			in = d->dev_outSrc;
		}
		copy_const_kernel<<<blocks, threads>>>(in);
		blend_kernel<<<blocks, threads>>(out, dstOut);
		dstOut = !dstOut;
	}

	float_to_color<<<blocks, threads>>>(d->output_bitmap, d->dev_inSrc);
	cudaMemcpy(bitmap->get_ptr(), d->output_bitmap, bitmap->image_size(), cudaMemcpyDeviceToHost);
	cudaEventRecord(d->stop, 0);
	cudaEventSynchronize(d->stop);
	float elapsedTime;
	cudaEventElapsedTime(&elapsedTime, d->start, d->stop);
	d->totalTime += elapsedTime;
	++d->frames;
	printf("Avarage Time per Frame: %3.1f ms\n", d->totalTime/d->frames);

}

void anim_exit(DataBlock *d) {
	cudaUnbindTexture(texIn);
	cudaUnbindTexture(texOut);
	cudaUnbindTexture(texConstSrc);
	cudaFree(d->dev_inSrc);
	cudaFree(d->dev_outSrc);
	cudaFree(d->dev_constSrc);

	cudaEventDestroy(d->start);
	cudaEventDestroy(d->stop);
}
