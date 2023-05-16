// Buckwild Benchmarking CLI program 

#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>
#include <math.h>


__global__ void throttle(float *in, float *out, int rows, int columns, int count);
__global__ void red_avg_gpu(float *in, float *out, int rowLength, int columnLength, int countRows, int countCols);
__host__ void help_msg();
__host__ int main(int argc, char **argv);
__host__ void CUDA_errorcheck();

void help_msg() {
	printf("Compares reduction .\n");
	printf("\t-n: Length of the array\n");
	printf("\t-g: Number of threads per GPU blocks\n");
}


__global__ void throttle(float *in, float *out, int rows, int columns, int count) {
	int currIdx = (blockIdx.x * blockDim.x + threadIdx.x) * count;
	int topIdx = currIdx + count;

	if (topIdx > rows) topIdx = rows;

	while (currIdx < topIdx) {
		out[currIdx] = 0;

		for (int i = 0; i < rows; i++) {
			out[currIdx] += in[i + currIdx * columns];
		}

		out[currIdx] /= rows;
		currIdx += 1;
	}
}


__global__ void red_avg_gpu(float *in, float *out, int rowLength, int columnLength, int countRows, int countCols) {
	
	float localReduced = 0;
	__shared__ float rowReduced;

	if (threadIdx.x == 1) {
		rowReduced = 0;
	}
	
	int blockId = blockIdx.x * countCols;
	int iMax = blockId + countCols;	
	int jStart = threadIdx.x;

	if (iMax > columnLength) {
		iMax = columnLength;
	}

	int i, j;

	for (i = blockId; i < iMax; i++) {
		for (j = jStart; j < rowLength; j += blockDim.x) {
			localReduced += in[i * rowLength + j];
		}

		atomicAdd(&rowReduced, localReduced);
		__syncthreads();

		if (threadIdx.x == 0) {
			out[i] = rowReduced / (float) rowLength;
			rowReduced = 0;
		}

		__syncthreads();

		localReduced = 0;
	}
}

__host__ void CUDA_errorcheck() {
	cudaError_t err = cudaGetLastError();
	if (err != cudaSuccess) {
		printf("CUDA Error: %s (%d)", cudaGetErrorString(err), err);
		exit(1);
	}
}


__host__ int main(int argc, char **argv) {

	int threads = 2;
	int arrayDim = 32;
	int benchRuns = 16;
	int printCompute = 0;

	int inputOpt;
	while((inputOpt = getopt(argc, argv, ":cg:n:b:")) != -1) {
		switch(inputOpt) {
			case 'n':
				arrayDim = atoi(optarg);
				}
				break;
			// Handle other inputs
			case 'g':
				threads = atoi(optarg);
				break;

			case 'b':
				benchRuns = atoi(optarg);
				break;

			case 'h':
				help_msg();
				exit(0);

			// Handle edge/error cases
			case '?':
				if ((optopt == 'n') || (optopt == 'g') || (optopt == 'b')) {
					fprintf(stderr, "Option '%c' requires an argument.\n", optopt);
				} else {
					fprintf(stderr, "Option '%c' is unknown or encountered an error.\n", optopt);
				}

				help_msg();
				return 1;

			default:
				break;
		}
	}



	cudaEvent_t initGPUTime1, finalGPUTime1, initGPUTime2, finalGPUTime2;
	float timeInMilli1, timeInMilli2;
	float benchTime1 = 0, benchTime2 = 0;
	cudaEventCreate(&initGPUTime1); cudaEventCreate(&finalGPUTime1); cudaEventCreate(&initGPUTime2); cudaEventCreate(&finalGPUTime2);

	int countRows = 2;
	int countCols = 32;

	while ((arrayDim/threads) > countRows) {
		countRows *= 2;
	}

	dim3 dimBlock(threads); // blocking 

	dim3 dimGrid_assgn1((arrayDim/dimBlock.x) + (!(arrayDim%dimBlock.x)?0:1)); // ceil(arrayDim/dimBlock.x)
	dim3 dimGrid_assgn2((arrayDim / countCols) + ((arrayDim % countCols == 0)?0:1)); // ceil(arrayDim/countCols)

	printf("Block: %d\tGrid1: %d\tGrid2: %d\n", dimBlock.x, dimGrid_assgn1.x, dimGrid_assgn2.x);

	float* originalArray = (float*) malloc(sizeof(float) * arrayDim * arrayDim);
	float* reducedVector1 = (float*) calloc(arrayDim, sizeof(float));
	float* reducedVector2 = (float*) calloc(arrayDim, sizeof(float));

	for (int i = 0; i < arrayDim * arrayDim; i++) {
		originalArray[i] = (float) drand48();
	}

	float* originalArray_gpu;
	float* reducedVector1_gpu;
	float* reducedVector2_gpu;

	cudaMalloc((void **) &originalArray_gpu, (size_t) sizeof(float) * arrayDim * arrayDim); CUDA_errorcheck();
	cudaMalloc((void **) &reducedVector1_gpu,(size_t) sizeof(float) * arrayDim); CUDA_errorcheck();
	cudaMalloc((void **) &reducedVector2_gpu,(size_t) sizeof(float) * arrayDim); CUDA_errorcheck();

	cudaMemcpy(originalArray_gpu,	originalArray, sizeof(float) * arrayDim * arrayDim, cudaMemcpyHostToDevice); CUDA_errorcheck();
	cudaMemcpy(reducedVector1_gpu, 	reducedVector1,sizeof(float) * arrayDim, cudaMemcpyHostToDevice); CUDA_errorcheck();
	cudaMemcpy(reducedVector2_gpu, 	reducedVector2,sizeof(float) * arrayDim, cudaMemcpyHostToDevice); CUDA_errorcheck();


	for (int i = 0; i < benchRuns; i++) {
		cudaEventRecord(initGPUTime1);
		// Taken from Method 1 
		throttle<<<dimGrid_assgn1,dimBlock>>>(originalArray_gpu, reducedVector1_gpu, arrayDim, arrayDim, 1); CUDA_errorcheck();
		cudaEventRecord(finalGPUTime1);

		cudaEventRecord(initGPUTime2);
		// Taken from Method 2
		red_avg_gpu<<<dimGrid_assgn2,dimBlock>>>(originalArray_gpu, reducedVector2_gpu, arrayDim, arrayDim, countRows, countCols); CUDA_errorcheck();
		cudaEventRecord(finalGPUTime2);


		cudaEventSynchronize(finalGPUTime1); cudaEventSynchronize(finalGPUTime2);
		cudaEventElapsedTime(&timeInMilli1, initGPUTime1, finalGPUTime1); cudaEventElapsedTime(&timeInMilli2, initGPUTime2, finalGPUTime2);

		cudaDeviceSynchronize(); CUDA_errorcheck();

		benchTime1 += timeInMilli1 / 1000.0;
		benchTime2 += timeInMilli2 / 1000.0;
	}

	benchTime1 /= (float) benchRuns;
	benchTime2 /= (float) benchRuns;

	cudaMemcpy(reducedVector1, reducedVector1_gpu, sizeof(float) * arrayDim, cudaMemcpyDeviceToHost); CUDA_errorcheck();
	cudaMemcpy(reducedVector2, reducedVector2_gpu, sizeof(float) * arrayDim, cudaMemcpyDeviceToHost); CUDA_errorcheck();

	printf("Method 1 \t\tAssignemnt 2\t\tRatio (A1/A2)\n");
	printf("%f\t\t%f\t\t%f\n", benchTime1, benchTime2, benchTime1 / benchTime2);

}


// __global__ void throttle(float* originalArray, float* reducedVector, int arrayDim, int arrayDim2, int throttle) {
//     int index = blockIdx.x * blockDim.x + threadIdx.x;
//     int stride = blockDim.x * gridDim.x;

//     for (int i = index; i < arrayDim; i += stride) {
//         float sum = 0;
//         for (int j = 0; j < arrayDim2; j++) {
//             sum += originalArray[i * arrayDim2 + j];
//         }
//         reducedVector[i] = sum / (float) arrayDim2;
//     }
// }

