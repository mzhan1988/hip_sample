
// Utilities and system includes
#include <assert.h>
#include <stdio.h>
#include <stdlib.h>
#include <math.h>

// hip runtime
#include <hip/hip_runtime.h>
#include "rocblas.h"

#ifndef min
#define min(a,b) ((a < b) ? a : b)
#endif
#ifndef max
#define max(a,b) ((a > b) ? a : b)
#endif

#define MATRIX_SIZE 5760
#define ITER 10

typedef struct
{
    unsigned int wA; 
    unsigned int hA; 
    unsigned int wB; 
    unsigned int hB; 
    unsigned int wC; 
    unsigned int hC; 
} MatrixSize;

void randomInit(double *data, int size)
{
    for (int i = 0; i < size; ++i)
    {   
        data[i] = rand() / (double)RAND_MAX;
    }   
}

void printMatrix(double *data, int size)
{
    for (int i = 0; i < size; ++i)
    {   
        printf("%lf\n", data[i]);
    }   
}

int main(int argc, char **argv)
{
    printf("CUBLAS MatrixMul test - Starting...\n");

    //init matrix size
    MatrixSize matrix_size;
    matrix_size.wA = MATRIX_SIZE;
    matrix_size.hA = MATRIX_SIZE;
    matrix_size.wB = MATRIX_SIZE;
    matrix_size.hB = MATRIX_SIZE;
    matrix_size.wC = MATRIX_SIZE;
    matrix_size.hC = MATRIX_SIZE;
    printf("MatrixA(%u,%u), MatrixB(%u,%u), MatrixC(%u,%u)\n",
           matrix_size.wA, matrix_size.hA,
           matrix_size.wB, matrix_size.hB,
           matrix_size.wC, matrix_size.hC);

    //init matrix
    unsigned int size_A = matrix_size.wA * matrix_size.hA;
    unsigned int mem_size_A = sizeof(double) * size_A;
    double *h_A = (double *)malloc(mem_size_A);
    unsigned int size_B = matrix_size.wB * matrix_size.hB;
    unsigned int mem_size_B = sizeof(double) * size_B;
    double *h_B = (double *)malloc(mem_size_B);
    unsigned int size_C = matrix_size.wC * matrix_size.hC;
    unsigned int mem_size_C = sizeof(double) * size_C;
    double *h_C = (double *)malloc(mem_size_C);

    randomInit(h_A, size_A);
    randomInit(h_B, size_B);
    memset(h_C, 0, mem_size_C);

    double *d_A, *d_B, *d_C;
    hipMalloc((void **) &d_A, mem_size_A);
    hipMalloc((void **) &d_B, mem_size_A);
    hipMalloc((void **) &d_C, mem_size_A);

    hipMemcpy(d_A, h_A, mem_size_A, hipMemcpyHostToDevice);
    hipMemcpy(d_B, h_B, mem_size_B, hipMemcpyHostToDevice);

    //get cuda device
    int devID = 0;
    hipError_t error;
    hipSetDevice(devID);
    hipDeviceProp_t deviceProp;
    hipGetDevice(&devID);
    hipGetDeviceProperties(&deviceProp, devID);
    printf("GPU Device %d: \"%s\" with compute capability %d.%d\n\n", devID, deviceProp.name, deviceProp.major, deviceProp.minor);

    const double alpha = 1.0f;
    const double beta  = 0.0f;
    rocblas_handle handle;
    rocblas_create_handle(&handle);

    rocblas_operation transa = rocblas_operation_none, transb = rocblas_operation_transpose;

    //Perform warmup operation with cublas
    //rocblas_dgemm(handle, transa, transb, matrix_size.wB, matrix_size.hA, matrix_size.wA, &alpha, d_B, matrix_size.wB, d_A, matrix_size.wA, &beta, d_C, matrix_size.wC);
    //hipDeviceSynchronize();

    hipEvent_t start;
    hipEventCreate(&start);
    hipEvent_t stop;
    hipEventCreate(&stop);
    int nIter = ITER;
    hipEventRecord(start, NULL);
    for (int j = 0; j < nIter; j++)
    {   
        rocblas_dgemm(handle, transa, transb, matrix_size.wB, matrix_size.hA, matrix_size.wA, &alpha, d_B, matrix_size.wB, d_A, matrix_size.wA, &beta, d_C, matrix_size.wC);
    }   
    error = hipEventRecord(stop, NULL);
    error = hipEventSynchronize(stop);

    float msecTotal = 0.0f;
    error = hipEventElapsedTime(&msecTotal, start, stop);

    // Compute and print the performance
    float msecPerMatrixMul = msecTotal / nIter;
    double flopsPerMatrixMul = 2.0 * (double)matrix_size.wA * (double)matrix_size.hA * (double)matrix_size.wB;
    double gigaFlops = (flopsPerMatrixMul * 1.0e-9f) / (msecPerMatrixMul / 1000.0f);
    printf(
        "Performance= %.2f GFlop/s, Time= %.3f msec, Size= %.0f Ops.\n",
        gigaFlops,
        msecPerMatrixMul,
        flopsPerMatrixMul);

    hipDeviceSynchronize();
    //copy bak
    hipMemcpy(h_C, d_C, mem_size_C, hipMemcpyDeviceToHost);
    //printMatrix(h_C, 100);

    rocblas_destroy_handle(handle);
    hipFree(d_A);
    hipFree(d_B);
    hipFree(d_C);
    free(h_A);
    free(h_B);
    free(h_C);
    //check result
    return 0;
}
