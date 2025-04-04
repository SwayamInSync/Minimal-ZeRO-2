
#include "zero_optimizer.h"
#include "utils.h" 
#include <iostream>
#include <numeric>
#include <cmath>
#include <algorithm>
#include <vector>
#include <cstdio> 
Zero2Optimizer::Zero2Optimizer(LinearModel& model, float learning_rate, float beta1,
                               int rank, int world_size, ncclComm_t comm)
    : model_(model), lr_(learning_rate), beta1_(beta1),
      rank_(rank), world_size_(world_size), comm_(comm)
{
    total_params_ = model_.get_total_params(); 

    
    if (rank_ == 0) {
        std::cout << "Initializing Zero2Optimizer..." << std::endl;
        std::cout << "  World Size: " << world_size_ << std::endl;
        std::cout << "  Learning Rate: " << lr_ << std::endl;
        std::cout << "  Momentum Beta1: " << beta1_ << std::endl;
    }

    
    if (world_size <= 0) throw std::invalid_argument("World size must be positive.");
    if (rank < 0 || rank >= world_size) throw std::invalid_argument("Rank out of bounds.");
    if (comm_ == nullptr) throw std::invalid_argument("NCCL communicator cannot be null.");

    model_.allocate_gradients(); 
    partition_optimizer_states(); 

    
    if (local_shard_size_ > 0) {
        CUDA_CHECK(cudaMalloc(&d_reduced_grad_shard_, local_shard_size_ * sizeof(float)));
    }

    
    if (total_params_ > 0) {
        CUDA_CHECK(cudaMalloc(&d_flat_gradients_, total_params_ * sizeof(float)));
        CUDA_CHECK(cudaMalloc(&d_flat_parameters_, total_params_ * sizeof(float)));
         if (rank_ == 0) {
             std::cout << "  Allocated flat gradient buffer (" << total_params_ << " elements)." << std::endl;
             std::cout << "  Allocated flat parameter buffer (" << total_params_ << " elements)." << std::endl;
         }
    } else {
        if (rank_ == 0) {
            std::cerr << "Warning: Model reported 0 total parameters." << std::endl;
        }
    }
}
Zero2Optimizer::~Zero2Optimizer() {
    
    auto logCudaError = [&](cudaError_t err, const char* msg) {
        if (err != cudaSuccess) {
            int rank = 0;
            #ifdef MPI_COMM_WORLD 
            
            int initialized = 0;
            if (MPI_Initialized(&initialized) == MPI_SUCCESS && initialized) {
               if (MPI_Comm_rank(MPI_COMM_WORLD, &rank) != MPI_SUCCESS) rank = -1; 
            } else {
                rank = -1; 
            }
            #endif
            
            if (rank <= 0) fprintf(stderr, "[Rank %d] CUDA Error %s: %s\n", rank, msg, cudaGetErrorString(err));
        }
    };
    if (d_local_momentum_shard_) logCudaError(cudaFree(d_local_momentum_shard_), "freeing d_local_momentum_shard_");
    if (d_reduced_grad_shard_) logCudaError(cudaFree(d_reduced_grad_shard_), "freeing d_reduced_grad_shard_");
    if (d_flat_gradients_) logCudaError(cudaFree(d_flat_gradients_), "freeing d_flat_gradients_");
    if (d_flat_parameters_) logCudaError(cudaFree(d_flat_parameters_), "freeing d_flat_parameters_");

    
}

void Zero2Optimizer::partition_optimizer_states() {
    
    if (total_params_ == 0 && rank_ == 0) {
         std::cerr << "Warning: Partitioning optimizer states with 0 total parameters." << std::endl;
         local_shard_size_ = 0;
         shard_offset_ = 0;
         return;
    }

    if (rank_ == 0) {
        std::cout << "  Total model parameters for partitioning: " << total_params_ << std::endl;
    }

    size_t base_shard_size = total_params_ / world_size_;
    size_t remainder = total_params_ % world_size_;
    local_shard_size_ = base_shard_size + (rank_ < remainder ? 1 : 0);
    shard_offset_ = rank_ * base_shard_size + std::min((size_t)rank_, remainder);

     if (rank_ == 0) { 
        for(int r=0; r<world_size_; ++r) {
            size_t r_shard_size = base_shard_size + (r < remainder ? 1 : 0);
            size_t r_shard_offset = r * base_shard_size + std::min((size_t)r, remainder);
            std::cout << "    Rank " << r << " -> Offset: " << r_shard_offset
                      << ", Size: " << r_shard_size << std::endl;
        }
    }

    
    if (local_shard_size_ > 0) {
        CUDA_CHECK(cudaMalloc(&d_local_momentum_shard_, local_shard_size_ * sizeof(float)));
        CUDA_CHECK(cudaMemset(d_local_momentum_shard_, 0, local_shard_size_ * sizeof(float)));
    } else {
        d_local_momentum_shard_ = nullptr; 
    }
}
void Zero2Optimizer::get_pointers_and_sizes(std::vector<float*>& param_ptrs,
                                            std::vector<float*>& grad_ptrs,
                                            std::vector<size_t>& sizes) {
    param_ptrs.clear();
    grad_ptrs.clear();
    sizes.clear();
    float* params[2];
    float* grads[2];
    
    model_.get_param_and_grad_pointers(params, grads);
    param_ptrs.push_back(params[0]); 
    grad_ptrs.push_back(grads[0]);   
    sizes.push_back(model_.W_size);  
    param_ptrs.push_back(params[1]); 
    grad_ptrs.push_back(grads[1]);   
    sizes.push_back(model_.b_size);  
}
void Zero2Optimizer::flatten_gradients(cudaStream_t stream) {
    if (!d_flat_gradients_) {
        
        printf("[Rank %d] ERROR: d_flat_gradients_ is NULL in flatten_gradients\n", rank_); fflush(stdout);
        MPI_Abort(MPI_COMM_WORLD, 1); 
        return;
    }

    std::vector<float*> grad_ptrs;
    std::vector<float*> param_ptrs; 
    std::vector<size_t> sizes;
    
    get_pointers_and_sizes(param_ptrs, grad_ptrs, sizes);

    float* d_W_grad = grad_ptrs[0];
    float* d_b_grad = grad_ptrs[1];
    size_t W_size = sizes[0];
    size_t b_size = sizes[1];

    
    MPI_Barrier(MPI_COMM_WORLD); 
    printf("[Rank %d] flatten_gradients: d_W_grad=%p, d_b_grad=%p, d_flat_gradients_=%p\n",
           rank_, (void*)d_W_grad, (void*)d_b_grad, (void*)d_flat_gradients_);
    fflush(stdout);
    MPI_Barrier(MPI_COMM_WORLD); 

    
    if (!d_W_grad && W_size > 0) {
         printf("[Rank %d] ERROR: d_W_grad is NULL in flatten_gradients before memcpy!\n", rank_); fflush(stdout);
         MPI_Abort(MPI_COMM_WORLD, 1);
    }
     if (!d_b_grad && b_size > 0) {
         printf("[Rank %d] ERROR: d_b_grad is NULL in flatten_gradients before memcpy!\n", rank_); fflush(stdout);
         MPI_Abort(MPI_COMM_WORLD, 1);
    }
    

    
    if (W_size > 0) {
        CUDA_CHECK(cudaMemcpyAsync(d_flat_gradients_, d_W_grad, W_size * sizeof(float), cudaMemcpyDeviceToDevice, stream));
    }
    
    if (b_size > 0) {
        CUDA_CHECK(cudaMemcpyAsync(d_flat_gradients_ + W_size, d_b_grad, b_size * sizeof(float), cudaMemcpyDeviceToDevice, stream));
    }
}
void Zero2Optimizer::reduce_scatter_gradients(cudaStream_t stream) {
    if (local_shard_size_ == 0 || !d_flat_gradients_) return; 

    
    flatten_gradients(stream); 

    
    
    

    
    NCCL_CHECK(ncclReduceScatter(d_flat_gradients_,       
                                 d_reduced_grad_shard_, 
                                 local_shard_size_,     
                                 ncclFloat,
                                 ncclSum,
                                 comm_,
                                 stream));
    
}
void Zero2Optimizer::update_local_params_and_optimizer_state(cudaStream_t stream) {
     if (local_shard_size_ == 0) return; 

    
    if (!d_local_momentum_shard_ || !d_reduced_grad_shard_) {
        printf("[Rank %d] ERROR: Optimizer state shards are NULL in update function!\n", rank_); fflush(stdout);
        MPI_Abort(MPI_COMM_WORLD, 1);
        return;
    }

    float* d_params[2];
    model_.get_param_and_grad_pointers(d_params, nullptr); 
    float* d_W = d_params[0];
    float* d_b = d_params[1];
    size_t W_size = model_.W_size;

    
    size_t current_offset_in_shard = 0;
    size_t remaining_in_shard = local_shard_size_;
    const int threads_per_block = 256;

    
    size_t global_start = shard_offset_;
    size_t global_end = shard_offset_ + local_shard_size_;

    
    if (global_start < W_size) { 
        size_t W_update_start_in_global_W = global_start;
        size_t W_update_count = std::min(global_end, W_size) - W_update_start_in_global_W;

        if (W_update_count > 0) {
            int blocks_W = (W_update_count + threads_per_block - 1) / threads_per_block;
            sgd_momentum_update_kernel<<<blocks_W, threads_per_block, 0, stream>>>(
                d_W + W_update_start_in_global_W,                   
                d_local_momentum_shard_ + current_offset_in_shard,  
                d_reduced_grad_shard_ + current_offset_in_shard,    
                lr_, beta1_,
                W_update_count);
            CUDA_CHECK(cudaGetLastError()); 
            current_offset_in_shard += W_update_count;
            remaining_in_shard -= W_update_count;
        }
    }

     
    if (remaining_in_shard > 0 && global_end > W_size) { 
        size_t b_update_start_in_global_b = std::max(0L, (long)shard_offset_ - (long)W_size);
        size_t b_update_count = remaining_in_shard;

         if (b_update_count > 0) {
            int blocks_b = (b_update_count + threads_per_block - 1) / threads_per_block;
             sgd_momentum_update_kernel<<<blocks_b, threads_per_block, 0, stream>>>(
                d_b + b_update_start_in_global_b,                   
                d_local_momentum_shard_ + current_offset_in_shard,  
                d_reduced_grad_shard_ + current_offset_in_shard,    
                lr_, beta1_,
                b_update_count);
            CUDA_CHECK(cudaGetLastError()); 
         }
    }
    
}
void Zero2Optimizer::flatten_updated_params(cudaStream_t stream) {
    if (!d_flat_parameters_ || local_shard_size_ == 0) return;

    std::vector<float*> param_ptrs;
    std::vector<float*> grad_ptrs; 
    std::vector<size_t> sizes;
    get_pointers_and_sizes(param_ptrs, grad_ptrs, sizes);
    float* d_W = param_ptrs[0];
    float* d_b = param_ptrs[1];
    size_t W_size = sizes[0];
    

    
    size_t global_start = shard_offset_;
    size_t global_end = shard_offset_ + local_shard_size_;

    
    if (global_start < W_size) {
        size_t W_update_start_in_global_W = global_start;
        size_t W_update_count = std::min(global_end, W_size) - W_update_start_in_global_W;
        if (W_update_count > 0) {
            
            if (!d_W) { printf("[Rank %d] ERROR: d_W is NULL in flatten_updated_params!\n", rank_); fflush(stdout); MPI_Abort(MPI_COMM_WORLD, 1); }
            CUDA_CHECK(cudaMemcpyAsync(d_flat_parameters_ + W_update_start_in_global_W, 
                                       d_W + W_update_start_in_global_W,                
                                       W_update_count * sizeof(float),
                                       cudaMemcpyDeviceToDevice, stream));
        }
    }

    
    size_t W_elements_in_shard = (global_start < W_size) ? std::min(global_end, W_size) - global_start : 0;
    if (local_shard_size_ > W_elements_in_shard) {
        size_t b_update_start_in_global_b = std::max(0L, (long)shard_offset_ - (long)W_size);
        size_t b_update_count = local_shard_size_ - W_elements_in_shard;
        if (b_update_count > 0) {
             
             if (!d_b) { printf("[Rank %d] ERROR: d_b is NULL in flatten_updated_params!\n", rank_); fflush(stdout); MPI_Abort(MPI_COMM_WORLD, 1); }
             CUDA_CHECK(cudaMemcpyAsync(d_flat_parameters_ + W_size + b_update_start_in_global_b, 
                                        d_b + b_update_start_in_global_b,                       
                                        b_update_count * sizeof(float),
                                        cudaMemcpyDeviceToDevice, stream));
        }
    }
    
}
void Zero2Optimizer::allgather_flat_parameters(cudaStream_t stream) {
    if (!d_flat_parameters_ || total_params_ == 0) return;

    
    flatten_updated_params(stream);
    
    

    
    MPI_Barrier(MPI_COMM_WORLD);
    printf("[Rank %d] Before AllGather(flat): local_shard_size = %zu, shard_offset = %zu\n", rank_, local_shard_size_, shard_offset_);
    fflush(stdout);
    MPI_Barrier(MPI_COMM_WORLD);

    
    NCCL_CHECK(ncclAllGather(d_flat_parameters_ + shard_offset_, 
                             d_flat_parameters_,                
                             local_shard_size_,                 
                             ncclFloat,
                             comm_,
                             stream));
    
}
void Zero2Optimizer::unflatten_parameters(cudaStream_t stream) {
     if (!d_flat_parameters_ || total_params_ == 0) return;

    std::vector<float*> param_ptrs;
    std::vector<float*> grad_ptrs; 
    std::vector<size_t> sizes;
    get_pointers_and_sizes(param_ptrs, grad_ptrs, sizes);
    float* d_W = param_ptrs[0];
    float* d_b = param_ptrs[1];
    size_t W_size = sizes[0];
    size_t b_size = sizes[1];

    
    if (W_size > 0 && d_W) {
        CUDA_CHECK(cudaMemcpyAsync(d_W, d_flat_parameters_, W_size * sizeof(float), cudaMemcpyDeviceToDevice, stream));
    } else if (W_size > 0 && !d_W) {
         printf("[Rank %d] ERROR: d_W is NULL in unflatten_parameters!\n", rank_); fflush(stdout); MPI_Abort(MPI_COMM_WORLD, 1);
    }

    
    if (b_size > 0 && d_b) {
        CUDA_CHECK(cudaMemcpyAsync(d_b, d_flat_parameters_ + W_size, b_size * sizeof(float), cudaMemcpyDeviceToDevice, stream));
    } else if (b_size > 0 && !d_b) {
         printf("[Rank %d] ERROR: d_b is NULL in unflatten_parameters!\n", rank_); fflush(stdout); MPI_Abort(MPI_COMM_WORLD, 1);
    }

    
    
    
}

void Zero2Optimizer::step(cudaStream_t stream) {
    reduce_scatter_gradients(stream); 
    update_local_params_and_optimizer_state(stream); 
    allgather_flat_parameters(stream); 
    unflatten_parameters(stream); 
    CUDA_CHECK(cudaStreamSynchronize(stream));
}
