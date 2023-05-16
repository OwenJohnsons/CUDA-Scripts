#include <stdio.h>
#include <cuda_runtime.h>
#include <math.h>

__device__ double expint(double x) {
    // Implementation of the exponential integral function
    // Modify this function if you have a specific implementation
    return exp(x) / x;
}

__global__ void computeExponentialIntegral(float* input, float* output, int size) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;

    if (tid < size) {
        float x = input[tid];
        output[tid] = expint(x);
    }
}

int main() {
    int size = 10;  // Number of elements
    float* input, * output;
    float* d_input, * d_output;

    // Allocate host memory
    input = (float*)malloc(size * sizeof(float));
    output = (float*)malloc(size * sizeof(float));

    // Allocate device memory
    cudaMalloc((void**)&d_input, size * sizeof(float));
    cudaMalloc((void**)&d_output, size * sizeof(float));

    // Initialize input data
    for (int i = 0; i < size; i++) {
        input[i] = i;  // Example input values
    }

    // Copy input data from host to device
    cudaMemcpy(d_input, input, size * sizeof(float), cudaMemcpyHostToDevice);

    // Launch kernel
    int threadsPerBlock = 256;
    int blocksPerGrid = (size + threadsPerBlock - 1) / threadsPerBlock;
    computeExponentialIntegral<<<blocksPerGrid, threadsPerBlock>>>(d_input, d_output, size);

    // Copy output data from device to host
    cudaMemcpy(output, d_output, size * sizeof(float), cudaMemcpyDeviceToHost);

    // Print results
    printf("Exponential Integral:\n");
    for (int i = 0; i < size; i++) {
        printf("Ei(%f) = %f\n", input[i], output[i]);
    }

    // Cleanup
    free(input);
    free(output);
    cudaFree(d_input);
    cudaFree(d_output);

    return 0;
}
