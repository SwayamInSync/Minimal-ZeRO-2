#ifndef UTILS_H
#define UTILS_H

#include <cuda_runtime.h>
#include <cstdio>
#include <stdexcept>
#include <string> 


#include <mpi.h>
#include <nccl.h>



#define CUDA_CHECK(call)                                                      \
do {                                                                          \
    cudaError_t err = call;                                                   \
    if (err != cudaSuccess) {                                                 \
        fprintf(stderr, "CUDA Error at %s:%d - %s\n", __FILE__, __LINE__,     \
                cudaGetErrorString(err));                                     \
                      \
                                           \
        throw std::runtime_error(cudaGetErrorString(err));                    \
    }                                                                         \
} while (0)


#define MPI_CHECK(call)                                                       \
do {                                                                          \
    int mpi_err = call;                                                       \
    if (mpi_err != MPI_SUCCESS) {                                             \
        char err_string[MPI_MAX_ERROR_STRING];                                \
        int len_of_err_string;                                                \
        MPI_Error_string(mpi_err, err_string, &len_of_err_string);            \
        fprintf(stderr, "MPI Error at %s:%d - %s\n", __FILE__, __LINE__,      \
                err_string);                                                  \
        MPI_Abort(MPI_COMM_WORLD, mpi_err);                                   \
        throw std::runtime_error(err_string);         \
    }                                                                         \
} while (0)



#define NCCL_CHECK(call)                                                      \
do {                                                                          \
    ncclResult_t nccl_err = call;                                             \
    if (nccl_err != ncclSuccess) {                                            \
        fprintf(stderr, "NCCL Error at %s:%d - %s\n", __FILE__, __LINE__,     \
                ncclGetErrorString(nccl_err));                                \
                                           \
                                           \
        throw std::runtime_error(ncclGetErrorString(nccl_err));               \
    }                                                                         \
} while (0)



__global__ void add_scaled_kernel(float* C, const float* A, const float* B, float alpha, size_t n);
__global__ void mul_kernel(float* C, const float* A, const float* B, size_t n);
__global__ void scale_kernel(float* A, float alpha, size_t n);
__global__ void set_kernel(float* A, float value, size_t n);
__global__ void sgd_momentum_update_kernel(float* params, float* momentum, const float* grads,
                                           float lr, float beta1, size_t n);


void print_device_vector(const float* d_vec, size_t n, const char* name, int rank);


#endif 