
#include "utils.h"
#include <vector>
#include <iostream>
#include <iomanip> 
#include <algorithm> 


__global__ void add_scaled_kernel(float* C, const float* A, const float* B, float alpha, size_t n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        C[idx] = A[idx] + alpha * B[idx];
    }
}

__global__ void mul_kernel(float* C, const float* A, const float* B, size_t n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        C[idx] = A[idx] * B[idx];
    }
}

__global__ void scale_kernel(float* A, float alpha, size_t n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        A[idx] *= alpha;
    }
}

__global__ void set_kernel(float* A, float value, size_t n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        A[idx] = value;
    }
}

__global__ void sgd_momentum_update_kernel(float* params, float* momentum, const float* grads,
                                           float lr, float beta1, size_t n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        float current_grad = grads[idx];
        float m_new = momentum[idx] * beta1 + current_grad;
        momentum[idx] = m_new;
        params[idx] -= lr * m_new;
    }
}



void print_device_vector(const float* d_vec, size_t n, const char* name, int rank) {
    
    if (rank != 0) {
        return;
    }
    std::vector<float> h_vec(n);
    CUDA_CHECK(cudaMemcpy(h_vec.data(), d_vec, n * sizeof(float), cudaMemcpyDeviceToHost));
    std::cout << "[Rank " << rank << "] " << name << " (" << n << " elements): [";
    size_t print_limit = std::min((size_t)10, n); 
    for (size_t i = 0; i < print_limit; ++i) {
        std::cout << std::fixed << std::setprecision(4) << h_vec[i] << (i == print_limit - 1 ? "" : ", ");
    }
    if (n > print_limit) {
        std::cout << ", ...";
    }
    std::cout << "]" << std::endl;
}