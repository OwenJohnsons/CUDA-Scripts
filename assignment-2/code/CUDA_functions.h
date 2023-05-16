// Header to import CUDA functions to the main program written in C language
#ifndef CUDA_FUNCTIONS_H
#define CUDA_FUNCTIONS_H

#ifdef __cplusplus
extern "C" {
#endif

// Function for GPU simulation 
extern int radiator_gpu(float *temperatures, float *temperatures_gpu, float **radiator, int n, int m, int iterations, int threads, int timing, int verbose, int M_iterations);

#ifdef __cplusplus
}
#endif

#endif // CUDA_FUNCTIONS_H