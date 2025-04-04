#ifndef MODEL_H
#define MODEL_H

#include <cstddef> 
#include <cuda_runtime.h> 


struct LinearModel {
    float* d_W = nullptr; 
    float* d_b = nullptr; 
    float* d_W_grad = nullptr; 
    float* d_b_grad = nullptr; 

    size_t input_dim;
    size_t output_dim;
    size_t W_size; 
    size_t b_size; 
    size_t total_params; 

    
    LinearModel(size_t in_dim, size_t out_dim);

    
    ~LinearModel() noexcept; 

    
    void allocate_gradients();

    
    void zero_gradients(cudaStream_t stream = 0); 

    
    size_t get_total_params() const;

    
    void get_param_and_grad_pointers(float** params, float** grads);

    
    float* forward(const float* d_input, size_t batch_size, cudaStream_t stream = 0);
    void backward(const float* d_input, const float* d_output_grad, size_t batch_size, cudaStream_t stream = 0);
};

#endif 