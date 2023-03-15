/*
Code Purpose: Create an single precision floating point n x m matrix with values between -5 and 5. N and M are command line arguments for size of the matrix. 
Author: Owen A. Johnson
Date: 03/03/2023
*/

#include <stdio.h>
#include <stdlib.h>
#include <time.h>

int main(int argc, char *argv[])
{
    // int n = atoi(argv[1]); // sizes of the matrix specified by command line arguments
    // int m = atoi(argv[2]);

    // if (argc < 2)
    // {
    //     printf("Error: Please specify the size of the matrix as two command line arguments.\n");
    //     return 1;
    // }

    // if (strcmp(argv[1], "-n") == 0)
    // {
    //     printf("n size specified as x \n");
    //     return 0;
    // }
    // if (strcmp(argv[1], "-m") == 0)
    // {
    //     printf("m size specified as x \n");
    //     return 0;
    // }

    int n = 5;
    int m = 5;

    long int seed = time(NULL); // Use the current time as the seed
    srand48(seed); // Initialize the random number generator with the seed
    
    // float *matrix[n][m] = (float *)malloc(n * m * sizeof(float));
    float matrix[n][m];

    srand(time(0));
    for (int i = 0; i < n; i++)
    {
        for (int j = 0; j < m; j++)
        {
            matrix[i][j] = ((float)(drand48())*10.0)-5.0; // Random value between -5 and 5
        }
    }

    FILE *fp = fopen("matrix.txt", "w"); // open file for writing
    if (fp == NULL) {
        printf("Error opening file.\n");
        return 1;
    }
    
    // write matrix to file
    for (int i = 0; i < n; i++) {
        for (int j = 0; j < m; j++) {
            fprintf(fp, "%f ", matrix[i][j]); // write element to file
        }
        fprintf(fp, "\n"); // add newline character after each row
    }
    
    fclose(fp); // close file

}
