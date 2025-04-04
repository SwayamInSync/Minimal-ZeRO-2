
#ifndef ZERO_OPTIMIZER_H
#define ZERO_OPTIMIZER_H

#include "model.h"
#include "utils.h"
#include <vector>
#include <cstddef>
#include <cuda_runtime.h>

class Zero2Optimizer {
public:
    Zero2Optimizer(LinearModel& model, float learning_rate, float beta1,
                   int rank, int world_size, ncclComm_t comm);
    ~Zero2Optimizer();
    void step(cudaStream_t stream = 0);

private:
    LinearModel& model_;
    float lr_;
    float beta1_;
    int rank_;
    int world_size_;
    ncclComm_t comm_;

    
    float* d_local_momentum_shard_ = nullptr;
    size_t local_shard_size_ = 0;
    size_t shard_offset_ = 0;
    float* d_reduced_grad_shard_ = nullptr;

    
    float* d_flat_gradients_ = nullptr;
    float* d_flat_parameters_ = nullptr; 
    size_t total_params_ = 0;

    
    void partition_optimizer_states();
    void flatten_gradients(cudaStream_t stream);
    void reduce_scatter_gradients(cudaStream_t stream);
    void update_local_params_and_optimizer_state(cudaStream_t stream);
    void flatten_updated_params(cudaStream_t stream); 
    void allgather_flat_parameters(cudaStream_t stream); 
    void unflatten_parameters(cudaStream_t stream); 
    void get_pointers_and_sizes(std::vector<float*>& param_ptrs,
                                std::vector<float*>& grad_ptrs,
                                std::vector<size_t>& sizes);
};

#endif 