/**
Code Purpose: Simulate temperature of a cylindrical radiator using both C and CUDA 
Author: Owen A. Johnson
Date of last major update: 13/04/2023 
*/

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>


// Function prototypes


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
    printf("\t -gpu: Complete computation on the GPU\n");
    printg("\t -t: Time the simulation\n");
}

/* 
Handling of command line arguments
*/
void cli_args(int argc, char *argv[], int *n, int *m, int *verbose, int *cpu){
    int i;
    for(i = 1; i < argc; i++){
        if(strcmp(argv[i], "-h") == 0){
            cli_flags();
            exit(0);
        }
        else if(strcmp(argv[i], "-n") == 0){
            *n = atoi(argv[i+1]);
        }
        else if(strcmp(argv[i], "-m") == 0){
            *m = atoi(argv[i+1]);
        }
        else if(strcmp(argv[i], "-verbose") == 0){
            *verbose = 1;
        }
        else if(strcmp(argv[i], "-cpu") == 0){
            *cpu = 1;
        }
        else if(strcmp(argv[i], "-gpu") == 0){
            *gpu = 1;
        }
        else if(strcmp(argv[i], "-t") == 0){
            *t = 1;
        }
    }
}

/*
Creating the radiator matrix given m and n values
*/
void create_radiator(int m, int n){
    if verbose == 1:
        printf("Creating radiator matrix of size %d x %d", m, n);

    int i, j;
    float* intial_radiator;
    intial_radiator = (float *)malloc(m * n * sizeof(float)); // Allocate memory for the radiator matrix
    for (int i = 0; i < m; i++){// creating array as per the assignment specification 
        intial_radiator[i*n] = 1*(float)(i+1)/(float)(m);
        intial_radiator[i*n + n + 1] = 0.7*float(i+1)/float(m);
    }
    if verbose == 1:
        printf("Radiator matrix created.");

    float* tempertures; // Create pointer for the temperatures matrix
    tempertures = calloc(m * n, sizeof(float)); // Contigous memory allocation for the temperatures matrix. Intialises memory blocks and sets them to 0
    if verbose == 1:
        printf("Temperatures matrix created.");

    // Error checking
    if (intial_radiator == NULL || tempertures == NULL){
        printf("Error: Memory allocation failed.");
        exit(1);
    }
    else{
        printf("Memory allocation successful.");
    }
}

void cpu_calculation(){
    // TODO
    if verbose == 1:
        printf("CPU calculation started.");
    // Array initialisation
    previous_radiator = calloc(m * n, sizeof(float)); // intialising previous time step of the radiator matrix
    current_radiator = calloc(m * n, sizeof(float)); // intialising current time step of the radiator matrix
       
    if (previous_radiator == NULL || current_radiator == NULL || tempertures == NULL){
        printf("Error: Memory allocation failed.");
        exit(1);
    }
    else{
        if (verbose)  printf("CPU matrix allocation complete.");
    }
    // copying the initial parameter radiator matrix to the previous radiator matrix
    if verbose == 1:
        printf("Copying initial radiator matrix to previous radiator matrix.");
    memcpy(previous_radiator, intial_radiator, m * n * sizeof(float));
    memcpy(current_radiator, intial_radiator, m * n * sizeof(float));

    // CPU calculation and simulation 
    if verbose == 1:
        printf("Starting CPU calculation and simulation.");
}

void radiator_weighting(float** previousMatrix, float** nextMatrix, int m, int n){
    // TODO
    for (int i = 0; i < m; i++){
        for (int j = 0; j < n; j++){
            (*nextMatrix)[i * n + j] = ( (1.65*(*previousMatrix)[i * rowLength + (j - 2)])+(1.35*(*previousMatrix)[i * rowLength + (j - 1)])+ (*previousMatrix)[i * rowLength + j]+ (0.65*(*previousMatrix)[i * rowLength + ((j + 1) % rowLength)])+(0.35*(*previousMatrix)[i * rowLength + ((j + 2) % rowLength)])) / (float)(5.0);
        }
    }

}

void main(int argc, char *argv[]){
    int n = 0, m = 0, verbose = 0, cpu = 0;
    cli_args(argc, argv, &n, &m, &verbose, &cpu);
    create_radiator(m, n);
}
