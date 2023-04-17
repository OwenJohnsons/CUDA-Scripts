/* 
Code Purpose: 
Author: 
Date: 
*/

#include <stdio.h>
#include <cuda.h>

// Prototype Functions 
// __global__ simulation

// Surface Intialization
surface<void, cudaSurfaceType2D> intial_surf;
surface<void, cudaSurfaceType2D> surf;

// Kernal Function for Radiator Simulation 
__global__ void radiator_sim(float *currentRadiator_gpu, float *passedRadiator_gpu, int n, int m, int iterations)
{
    // Get the thread ID
    int blockID = blockIdx.x * m // Block IDs for kernal to process 
    int maxID = blockID + m; // Max ID for kernal to process

    if (maxID > m) // If the max ID is greater than the total number of threads, set it to the total number of threads
    {
        maxID = m;
    }

    // Loop through the iterations 
    for (i =0; i < maxID; i++)
    offset = i * m; // Offset for the threads to process
    {
        // Loop through the threads 
        for (j = blockID + 2; j < maxID; j++)
        {
            // Get the current value of the radiator 
            float current = currentRadiator_gpu[j];

            // Get the values of the neighbors 
            float left = currentRadiator_gpu[j-1];
            float right = currentRadiator_gpu[j+1];
            float top = currentRadiator_gpu[j-m];
            float bottom = currentRadiator_gpu[j+m];

            // Calculate the average of the neighbors 
            float average = (left + right + top + bottom) / 4;

            // Set the new value of the radiator 
            passedRadiator_gpu[j] = average;
        }
    }

}