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

    int n = 10;
    int m = 10;
    
    float *matrix = (float *)malloc(n * m * sizeof(float));
    srand(time(0));
    for (int i = 0; i < n; i++)
    {
        for (int j = 0; j < m; j++)
        {
            matrix[i * m + j] = ((float)(drand48())*10.0)-5.0; // Random value between -5 and 5
        }
    }
    FILE *fp;
    fp = fopen("matrix.txt", "w");
    for (int i = 0; i < n; i++)
    {
        for (int j = 0; j < m; j++)
        {
            fprintf(fp, "%f ", matrix[i * m + j]);
        }
    }

    fclose(fp);
}
