/*
Code Purpose: Create an single precision floating point n x m matrix. N and M are command line arguments. 
*/

#include <stdio.h>
#include <stdlib.h>
#include <time.h>

int main(int argc, char *argv[])
{
    // int n = atoi(argv[1]);
    // int m = atoi(argv[2]);

    int n = 1000;
    int m = 1000;
    
    float *matrix = (float *)malloc(n * m * sizeof(float));
    srand(time(0));
    for (int i = 0; i < n; i++)
    {
        for (int j = 0; j < m; j++)
        {
            matrix[i * m + j] = (float)rand() / (float)(RAND_MAX / 100);
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
