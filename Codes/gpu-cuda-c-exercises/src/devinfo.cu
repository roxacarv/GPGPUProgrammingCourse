#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <stdio.h>
#include <ostream>

void ShowGPUProps(int DeviceID)
{
	cudaDeviceProp DeviceProp;
	cudaGetDeviceProperties(&DeviceProp, DeviceID);
	printf("--- Card: %s --- Device Number: %d --- Integrated: %s ---\n", DeviceProp.name, DeviceID, DeviceProp.integrated ? "True" : "False");
	printf("---   Major Revision is %d and Minor Revision is %d   ---\n", DeviceProp.major, DeviceProp.minor);
	printf("---------------------------------------------------------\n");
	printf("Total Global Memory: %d\n", DeviceProp.totalGlobalMem);
	printf("Total Constant Memory: %d\n", DeviceProp.totalConstMem);
	printf("Maximum Threads Per Block: %d\n", DeviceProp.maxThreadsPerBlock);
	printf("Maximum Grid Size: %d\n", DeviceProp.maxGridSize);
	printf("Multi Processors: %d\n", DeviceProp.multiProcessorCount);
	printf("Maximum Texture Dimensions;\n");
	printf("1D: %d\n2D: %d\n3D: %d\n", DeviceProp.maxTexture1D, DeviceProp.maxTexture2D, DeviceProp.maxTexture3D);
	printf("Warp Size: %d\n", DeviceProp.warpSize);
	printf("Registers Per Block: %d\n", DeviceProp.regsPerBlock);
	printf("---------------------------------------------------------\n");
}

int main()
{
	int DeviceCount, UserInput;
	cudaGetDeviceCount(&DeviceCount);

	if (DeviceCount > 1)
	{
		printf("This machine has %d video cards.\n1 - Choose one card by number\n2 - Choose all cards\nYour choice: ");
		scanf("%d", &UserInput);
		if (UserInput == 1)
		{
			printf("Choose the a number between 0 and %d: ", (DeviceCount - 1));
			scanf("%d", &UserInput);
			if (UserInput >= 0 && UserInput <= DeviceCount - 1)
				ShowGPUProps(UserInput);
			else
				printf("A wrong value was typed. Please try again.\n");
			main();
		}
		else
		{
			for (int i = 0; i < DeviceCount - 1; i++)
				ShowGPUProps(i);
		}
	}
	else
	{
		ShowGPUProps(0);
	}
	return 0;
}
