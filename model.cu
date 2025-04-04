
#include "model.h"
#include "utils.h"
#include <cuda_runtime.h>
#include <random> 
#include <iostream>
#include <vector> 

LinearModel::LinearModel(size_t in_dim, size_t out_dim) : input_dim(in_dim), output_dim(out_dim) {
    W_size = output_dim * input_dim;
    b_size = output_dim;
    total_params = W_size + b_size;

    
    int rank = 0;
    #ifdef MPI_COMM_WORLD 
      MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    #endif

    if (rank == 0) {
        std::cout << "Initializing Linear Model (" << input_dim << " -> " << output_dim << ")" << std::endl;
        std::cout << "  Weight size (W): " << W_size << std::endl;
        std::cout << "  Bias size (b): " << b_size << std::endl;
        std::cout << "  Total parameters: " << total_params << std::endl;
    }


    
    CUDA_CHECK(cudaMalloc(&d_W, W_size * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_b, b_size * sizeof(float)));

    
    std::vector<float> h_W(W_size);
    std::vector<float> h_b(b_size);

    std::mt19937 gen(1337); 
    std::uniform_real_distribution<> dis(-0.1f, 0.1f);
    for (size_t i = 0; i < W_size; ++i) h_W[i] = dis(gen);
    for (size_t i = 0; i < b_size; ++i) h_b[i] = 0.0f; 

    CUDA_CHECK(cudaMemcpy(d_W, h_W.data(), W_size * sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_b, h_b.data(), b_size * sizeof(float), cudaMemcpyHostToDevice));

    
}






LinearModel::~LinearModel() noexcept {
    
    
    int rank = 0;
    #ifdef MPI_COMM_WORLD
      MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    #endif

    if (d_W) {
        cudaError_t err = cudaFree(d_W);
        if (err != cudaSuccess && rank == 0) fprintf(stderr, "CUDA Error freeing d_W: %s\n", cudaGetErrorString(err));
    }
    if (d_b) {
         cudaError_t err = cudaFree(d_b);
         if (err != cudaSuccess && rank == 0) fprintf(stderr, "CUDA Error freeing d_b: %s\n", cudaGetErrorString(err));
    }
    if (d_W_grad) {
         cudaError_t err = cudaFree(d_W_grad);
         if (err != cudaSuccess && rank == 0) fprintf(stderr, "CUDA Error freeing d_W_grad: %s\n", cudaGetErrorString(err));
    }
    if (d_b_grad) {
         cudaError_t err = cudaFree(d_b_grad);
         if (err != cudaSuccess && rank == 0) fprintf(stderr, "CUDA Error freeing d_b_grad: %s\n", cudaGetErrorString(err));
    }
     
     
}

void LinearModel::allocate_gradients() {
    int rank = 0;
    #ifdef MPI_COMM_WORLD
      MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    #endif

    if (!d_W_grad) {
        CUDA_CHECK(cudaMalloc(&d_W_grad, W_size * sizeof(float)));
        
    }
     if (!d_b_grad) {
        CUDA_CHECK(cudaMalloc(&d_b_grad, b_size * sizeof(float)));
        
    }
}

void LinearModel::zero_gradients(cudaStream_t stream) {
    
    
    allocate_gradients();
    

    
    if (d_W_grad) { 
        CUDA_CHECK(cudaMemsetAsync(d_W_grad, 0, W_size * sizeof(float), stream));
    }
     if (d_b_grad) {
        CUDA_CHECK(cudaMemsetAsync(d_b_grad, 0, b_size * sizeof(float), stream));
    }
}

size_t LinearModel::get_total_params() const {
    return total_params;
}

void LinearModel::get_param_and_grad_pointers(float** params, float** grads) {
    if (params) { 
        params[0] = d_W;
        params[1] = d_b;
    }
    if (grads) { 
        
        allocate_gradients();
        grads[0] = d_W_grad;
        grads[1] = d_b_grad;
    }
}




float* LinearModel::forward(const float* d_input, size_t batch_size, cudaStream_t stream) {
    float* d_output = nullptr;
    size_t output_size = batch_size * output_dim;
    CUDA_CHECK(cudaMalloc(&d_output, output_size * sizeof(float)));

    const int threads_per_block = 256;
    const int blocks_per_grid = (output_size + threads_per_block - 1) / threads_per_block;
    set_kernel<<<blocks_per_grid, threads_per_block, 0, stream>>>(d_output, 1.0f, output_size);
    CUDA_CHECK(cudaGetLastError());

    return d_output;
}

void LinearModel::backward(const float* d_input, const float* d_output_grad, size_t batch_size, cudaStream_t stream) {
    
    allocate_gradients();

    const int threads_per_block = 256;

    
    int blocks_W = (W_size + threads_per_block - 1) / threads_per_block;
    set_kernel<<<blocks_W, threads_per_block, 0, stream>>>(d_W_grad, 0.1f, W_size);
    CUDA_CHECK(cudaGetLastError());

    
    int blocks_b = (b_size + threads_per_block - 1) / threads_per_block;
    set_kernel<<<blocks_b, threads_per_block, 0, stream>>>(d_b_grad, 0.01f, b_size);
    CUDA_CHECK(cudaGetLastError());
}