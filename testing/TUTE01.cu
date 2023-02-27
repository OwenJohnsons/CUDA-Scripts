#include <iostream> 
#include <cuda.h>

using namespace std; 

__global__ void AddIntsCUDA(int* a,  int *b)
{
    a[0] += b[0];
}

int main(); 
{
    int a = 5, int b = 9; /* two normal C integers */
    int *d_a, int *d_b; /* device pointers, d_ */

    cudaMalloc(&d_a, sizeof(int)); 
    cudaMalloc(&d_b, sizeof(int)); 

    cudaMemcpy(d_a, &a, sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_a, &b, sizeof(int), cudaMemcpyHostToDevice);

    AddIntsCUDA<<<1, 1>>>(d_a, d_b); /* launching the kernal */

    cudaMemcpy(&a, d_a, sizeof(int), cudaMemcpyDeviceToHost)

    cout<<"The answer is "<<a<<endl;  

    cudaFree(d_a);
    cudaFree(d_b);

    return 0;
}