/**
Code Purpose: Simulate temperature of a cylindrical radiator using both C and CUDA 
Author: Owen A. Johnson
Date of last major update: 13/04/2023 
*/

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <time.h>
#include <CUDA_functions.h>

// Function prototypes
void cli_flags();
void cli_args(int argc, char *argv[], int *n, int *m, int *verbose, int *cpu, int *print_results, int *gpu_threads, int *iterations, int *M_iterations); 
float* create_radiator(int m, int n);
void cpu_calculation(int m, int n, float* intial_radiator, int iterations);
void radiator_weighting(float** previousMatrix, float** nextMatrix, int m, int n);
void row_average(float** matrix, float **array, int m, int n); 
void print_temps(float **array, int elements, int iterations); 

int n = 32, m = 32, verbose = 0, cpu = 0, iterations = 5, t = 0, print_results = 0, gpu_threads = -1, M_iterations = 0;  // Intial variables for command line arguments

/*
Command line arguments
*/
void cli_flags(){
    printf("Simulate temperature of a cylindrical radiator using both C and CUDA using the following command line arguments:\n");
    printf("\t -h: Display this help message\n");
    printf("\t -n: Radiator Width\n");
    printf("\t -m: Radiator Height\n");
    printf("\t -verbose: Display debug information\n");
    printf("\t -cpu: Complete computation on the CPU\n");
    printf("\t -g: Complete computation on the GPU with specified threads.\n");
    printf("\t -t: Time the simulation\n");
    printf("\t -i: Number of iterations to run the simulation for\n");
    printf("\t -a: Every N iterations, print the average temperature of the radiator\n");
    printf("\t -print: Print the temperatures of the radiator at each iteration (large output).\n");
}

/* 
Main Function
*/
void main(int argc, char *argv[]){
    cli_args(argc, argv, &n, &m, &verbose, &cpu, &print_results, &gpu_threads, &iterations, &M_iterations); // Handle command line arguments
    float* intial_radiator = create_radiator(m, n);
    if (cpu == 1){
        cpu_calculation(m, n, intial_radiator, iterations);
    }
    
}

/* 
Handling of command line arguments
*/
void cli_args(int argc, char *argv[], int *n, int *m, int *verbose, int *cpu, int *print_results, int *gpu_threads, int *iterations, int *M_iterations){
    int i;
    for(i = 1; i < argc; i++){
        if(strcmp(argv[i], "-h") == 0){
            cli_flags();
            exit(0);
        }
        else if(strcmp(argv[i], "-n") == 0){
           *n = atoi(argv[i+1]);
           if (*n < 1){
               printf("Error: Radiator width must be greater than 0.");
               exit(1);
           }
           if ((*n % 32) != 0){
               printf("Warning: Radiator width is expected to be a multiple of 32.\n");
               
           }
        }
        else if(strcmp(argv[i], "-m") == 0){
            *m = atoi(argv[i+1]);
            if (*m < 1){
               printf("Error: Radiator length must be greater than 0.\n");
               exit(1);
           }
            if ((*m % 32) != 0){
               printf("Warning: Radiator width is expected to be a multiple of 32.\n");
            }
        }
        else if(strcmp(argv[i], "-verbose") == 0){
            *verbose = 1;
        }
        else if(strcmp(argv[i], "-print") == 0){
            *print_results = 1;
        }
        else if(strcmp(argv[i], "-cpu") == 0){
            *cpu = 1;
        }
        else if(strcmp(argv[i], "-g") == 0){
            int *gpu_threads = atoi(argv[i+1]);
        }
        else if(strcmp(argv[i], "-t") == 0){
            int *t = 1;
        }
        else if(strcmp(argv[i], "-i") == 0){
            *iterations = atoi(argv[i+1]);
        }
        else if(strcmp(argv[i], "-a") == 0){
            int *M_iterations = atoi(argv[i+1]);
        }
        else{
            printf("Error: Invalid command line argument.\n");
            exit(1);
        }
    }
    return 0;
}

/*
Creating the radiator matrix given m and n values
*/
float* create_radiator(int m, int n){
    printf("------\n");
    printf("Creating radiator matrix of size %d x %d\n", m, n);
    printf("...and simulating for %d iterations.\n", iterations);

    int i, j;
    float* intial_radiator;
    intial_radiator = (float *)malloc(m * n * sizeof(float)); // Allocate memory for the radiator matrix
    for (int i = 0; i < m; i++){// creating array as per the assignment specification 
        intial_radiator[i*n] = 1*(float)(i+1)/(float)(m);
        intial_radiator[i*n + n + 1] = 0.7 * (float) (i + 1) / (float) m;
    }
    if (verbose == 1){
        printf("Radiator matrix created.\n");
    }

    float* tempertures; // Create pointer for the temperatures matrix
    tempertures = calloc(m * n, sizeof(float)); // Contigous memory allocation for the temperatures matrix. Intialises memory blocks and sets them to 0
    if( verbose == 1){
        printf("Temperatures matrix created.\n");
    }

    // Error checking
    if (intial_radiator == NULL || tempertures == NULL){
        printf("Error: Memory allocation failed.\n");
        exit(1);
    }
    else{
        printf("Matrix memory allocation successful.\n------\n");
    }
    return intial_radiator;
}

void cpu_calculation(int m, int n, float* intial_radiator, int iterations){
    clock_t begin = clock(); // Start clock
    if (verbose == 1){
        printf("CPU calculation started.\n");
    }
    // Array initialisation
    float *previous_radiator, *current_radiator, *tempertures; // pointers for the previous and current radiator matrices
    previous_radiator = calloc(m * n, sizeof(float)); // intialising previous time step of the radiator matrix
    current_radiator = calloc(m * n, sizeof(float)); // intialising current time step of the radiator matrix
    tempertures = calloc(n, sizeof(float)); // intialising the temperatures array
       
    if (previous_radiator == NULL || current_radiator == NULL || tempertures == NULL){
        printf("Error: Memory allocation failed.\n");
        exit(1);
    }
    else{
        if (verbose == 1){
            printf("CPU matrix allocation complete.\n");
        }
    }
    // copying the initial parameter radiator matrix to the previous radiator matrix
    if (verbose == 1){
        printf("Copying initial radiator matrix to previous radiator matrix.\n");
        }
    memcpy(previous_radiator, intial_radiator, m * n * sizeof(float));
    memcpy(current_radiator, intial_radiator, m * n * sizeof(float));

    // CPU calculation and simulation 
   
    printf("\n------\nStarting CPU calculation and simulation....");

    // Looping through the number of iterations
    for (int i = 0; i < iterations; i++){
        radiator_weighting(&previous_radiator, &current_radiator, m, n);

        row_average(&current_radiator, &tempertures, m, n);
        if (print_results == 1){
            print_temps(&tempertures, m, i + 1); // skips intial radiator and starts with first iteration. 
        }
    
    }
    printf("\nDone!\n");
    clock_t end = clock(); // End clock
    double time_spent = (double)(end - begin) / CLOCKS_PER_SEC; // Calculate time taken
    if (time == 1);
        printf("CPU calculation and simulation completed in %f seconds.\n------\n", time_spent);
}

void radiator_weighting(float** previousMatrix, float** nextMatrix, int m, int n){
    for (int i = 0; i < m; i++){
        for (int j = 0; j < n; j++){
            (*nextMatrix)[i * n + j] = ((1.65*(*previousMatrix)[i * m + (j - 2)])+(1.35*(*previousMatrix)[i * m + (j - 1)])+ (*previousMatrix)[i * m + j]+ (0.65*(*previousMatrix)[i * m + ((j + 1) % m)])+(0.35*(*previousMatrix)[i * m + ((j + 2) % m)])) / (float)(5.0);
        }
    }
}

void row_average(float** matrix, float **array, int m, int n){
    for (int i = 0; i < m; i++){
        for (int j = 0; j < n; j++){
            (*array)[i] += (*matrix)[i * n + j];
        }
        (*array)[i] = (*array)[i] / (float)(n);
    }
}

void print_temps(float **array, int elements, int iterations){
    if (iterations){
        printf("\n------\nAverage row temperatures after %d iterations\n------\n", iterations);
        for (int i = 0; i < elements; i++){
            printf("%f\n", (*array)[i]);
        }
    }
}

void gpu_execution(tempertures, intial_radiator,  n,  m,  iterations,  gpu_threads, t,  verbose,  M_iterations){
    printf("GPU execution started.\n");

    float *results; // Create pointer for the resul
    results = calloc(m*n, sizeof(float)); // Allocate memory for the radiator matrix
    if (results == NULL){
        printf("Error: Memory allocation failed.\n");
        exit(1);
    }
    else{
        if (verbose == 1){
        printf("GPU matrix allocation complete.\n");
        }
    }

    // GPU execution
    if (radiator_gpu(tempertures,  results,  intial_radiator,  n,  m,  iterations,  gpu_threads, t,  verbose,  M_iterations) > 0){
        printf("Error: GPU execution failed.\n");
        exit(1);
    }
    else{
        printf("GPU execution complete.\n");
    }

}


