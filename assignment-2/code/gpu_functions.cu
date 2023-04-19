/* 
Code Purpose: 
Author: 
Date: 
*/

#include <stdio.h>
#include <cuda.h>
#include <CUDA_functions.h>

// Prototype Functions 
// __global__ simulation

// Surface Intialization
surface<void, cudaSurfaceType2D> intial_surf;
surface<void, cudaSurfaceType2D> surf;

__host__ void CUDA_errorcheck(){
    cudaError_t error = cudaGetLastError();
    if (error != cudaSuccess){
        printf("CUDA Error: %s \n", cudaGetErrorString(error));
        exit(1);
    }
}

// Kernal Function for Radiator Simulation 
__global__ void radiator_sim(float *currentRadiator_gpu, float *passedRadiator_gpu, int n, int m, int iterations)
{
    float values[5]; // Array to hold the values of the neighbors and performing temperature weighting.
    
    // Get the thread ID
    int blockID = blockIdx.x * m; // Block IDs for kernal to process 
    int maxID = blockID + m; // Max ID for kernal to process

    if (maxID > m) // If the max ID is greater than the total number of threads, set it to the total number of threads
    {
        maxID = m;
    }

    // Loop through the iterations 
    for (int i =0; i < maxID; i++){
        int offset = i * m; // Offset for the threads to process
        // Loop through the threads 
        for (int j = blockID; j < maxID; j++)
        {
            j = j + offset; // Add the offset to the thread ID
            // Get the values of the neighbors
            values[0] = currentRadiator_gpu[j - 2]; // Leftmost 
            values[1] = currentRadiator_gpu[j - 1]; // Left
            values[2] = currentRadiator_gpu[j]; // Center
            values[3] = currentRadiator_gpu[j + 1]; // Right
            values[4] = currentRadiator_gpu[j + 2]; // Rightmost

            // Perform the temperature weighting
            currentRadiator_gpu[i * n + j] = (values[0] * 1.65) + (values[1] * 1.35) + (values[2]) + (values[3] * 0.65) + (values[4] * 0.35);
            // printf("row, col: %d, %d \n Rad Value: %f", i, j, currentRadiator_gpu[i * n + j]); // for debugging

            __syncthreads(); // Sync the threads
        }
    }

}

// GPU Reduction Function
__global__ void reduce_matrix_gpu(float *input_vector, float *output_vector, int n, int m)
{
    // Get the thread ID
    int blockID = blockIdx.x * m; // Block IDs for kernal to process 
    int maxID = blockID + m; // Max ID for kernal to process

    if (maxID > m) // If the max ID is greater than the total number of threads, set it to the total number of threads
    {
        maxID = m;
    }

    // Loop through the threads 
    for (int i =0; i < maxID; i++){
        for (int j = threadIdx.x; j < n; j+=blockDim.x)
        {
            output_vector[i] += input_vector[i * n + j]; // Add the values to the output vector

            __syncthreads(); // Sync the threads
        }
    }
}

// Main Function to simulate on the GPU 

__host__ extern radiator_gpu(){

    if (verbose){
        printf("Beginning GPU Simulation")
    }

    // Initialize the variables
    float *temperatures_gpu, *currentRadiator_gpu, *passedRadiator_gpu; // *output_vector_gpu, *output_vector_cpu;
    int n_counts = 32; int m_counts = 32 

    while ((m/threads) > n_counts){
        n_counts = n_counts * 2; // To balance the workload between threads by increasing the number of threads being used until the workload per thread is sufficiently small.
    }

    // Time Keeping Variables
    cudaEvent_t start, stop, runtime_start, runtime_stop;
    float time, runtime;
    cudaEventCreate(&start); cudaEventCreate(&stop); 
    cudaEventCreate(&runtime_start); cudaEventCreate(&runtime_stop);

    if timing {
        cudaEventRecord(start, 0); // Start the timer
    }

    cudaMalloc((void**)&temperatures_gpu, sizeof(float) * m); // Allocate memory for the temperatures

    if (verbose){
        printf("Allocating memory for GPU arrays!")
    }

    // Allocate memory for the GPU arrays
    cudaMalloc((void**)&currentRadiator_gpu, sizeof(float) * m * n); // Allocate memory for the current radiator
    cudaMalloc((void**)&passedRadiator_gpu, sizeof(float) * m * n); // Allocate memory for the passed radiator
    CUDA_errorcheck(); // Check for errors
}