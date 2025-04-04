#include "utils.h"
#include "model.h"
#include "zero_optimizer.h"
#include <iostream>
#include <vector>
#include <memory> 
#include <stdexcept> 


const size_t INPUT_DIM = 1024;
const size_t OUTPUT_DIM = 1024;
const size_t BATCH_SIZE = 64; // Per GPU batch size
const float LEARNING_RATE = 0.01f;
const float MOMENTUM_BETA1 = 0.9f;
const int NUM_STEPS = 10000; 


int main(int argc, char* argv[]) {
    MPI_CHECK(MPI_Init(&argc, &argv));
    int rank, world_size;
    MPI_CHECK(MPI_Comm_rank(MPI_COMM_WORLD, &rank));
    MPI_CHECK(MPI_Comm_size(MPI_COMM_WORLD, &world_size));

    int num_devices;
    CUDA_CHECK(cudaGetDeviceCount(&num_devices));
    if (num_devices < world_size) {
        if (rank == 0) {
            fprintf(stderr, "Error: Not enough CUDA devices available (%d) for the number of MPI ranks (%d).\n",
                    num_devices, world_size);
        }
        MPI_Abort(MPI_COMM_WORLD, 1);
        return 1;
    }
    // Assign device based on rank (simple mapping)
    int device_id = rank % num_devices;
    CUDA_CHECK(cudaSetDevice(device_id));
    if (rank == 0) {
         std::cout << "===== ZeRO-2 Optimizer Prototype (Multi-GPU) =====" << std::endl;
         std::cout << "MPI World Size: " << world_size << std::endl;
         std::cout << "CUDA Devices Found: " << num_devices << std::endl;
    }
     printf("[Rank %d] Using CUDA device %d\n", rank, device_id);


    // --- NCCL Initialization ---
    ncclComm_t comm;
    ncclUniqueId nccl_id;
    // Rank 0 creates a unique ID and broadcasts it
    if (rank == 0) {
        NCCL_CHECK(ncclGetUniqueId(&nccl_id));
    }
    MPI_CHECK(MPI_Bcast((void*)&nccl_id, sizeof(nccl_id), MPI_BYTE, 0, MPI_COMM_WORLD));
    NCCL_CHECK(ncclCommInitRank(&comm, world_size, nccl_id, rank));
    if (rank == 0) {
        std::cout << "NCCL Communicator initialized." << std::endl;
    }

    cudaStream_t stream = 0;

    // --- Model and Optimizer Setup (One per Rank) ---
    // Each rank creates its own model instance (parameters are replicated)
    // and its own optimizer instance.
    LinearModel model(INPUT_DIM, OUTPUT_DIM);
    Zero2Optimizer optimizer(model, LEARNING_RATE, MOMENTUM_BETA1, rank, world_size, comm);

    if (rank == 0) {
        std::cout << "\nCreated Model and Zero2Optimizer on each rank." << std::endl;
    }

    // --- Dummy Data (Each rank creates identical data for simplicity) ---
    size_t input_size = BATCH_SIZE * INPUT_DIM;
    float* d_input;
    CUDA_CHECK(cudaMalloc(&d_input, input_size * sizeof(float)));
    const int threads = 256;
    int blocks = (input_size + threads - 1) / threads;
    set_kernel<<<blocks, threads, 0, stream>>>(d_input, 0.5f + rank * 0.01f, input_size); // Slightly different input per rank
    CUDA_CHECK(cudaStreamSynchronize(stream)); // Ensure data is ready
    if (rank == 0) {
        std::cout << "Created dummy input data on each GPU." << std::endl;
    }

    // --- Training Loop ---
    if (rank == 0) {
        std::cout << "\n--- Starting Training Loop (" << NUM_STEPS << " steps) ---" << std::endl;
    }
    for (int step = 0; step < NUM_STEPS; ++step) {
         if (rank == 0) {
            std::cout << "\n===== Step " << step + 1 << " =====" << std::endl;
         }
         // Quick barrier before starting step for cleaner logs (optional)
         MPI_CHECK(MPI_Barrier(MPI_COMM_WORLD));



        float* d_output = model.forward(d_input, BATCH_SIZE, stream);
        size_t output_data_size = BATCH_SIZE * OUTPUT_DIM;
        float* d_output_grad;
        CUDA_CHECK(cudaMalloc(&d_output_grad, output_data_size * sizeof(float)));
        blocks = (output_data_size + threads - 1) / threads;
        set_kernel<<<blocks, threads, 0, stream>>>(d_output_grad, 0.01f * (step + 1) * (rank + 1), output_data_size); // Vary gradient
        CUDA_CHECK(cudaStreamSynchronize(stream));


        // --- Backward Pass ---
        model.zero_gradients(stream);
        CUDA_CHECK(cudaStreamSynchronize(stream));
        model.backward(d_input, d_output_grad, BATCH_SIZE, stream);
        CUDA_CHECK(cudaStreamSynchronize(stream));

        optimizer.step(stream);


        CUDA_CHECK(cudaFree(d_output));
        CUDA_CHECK(cudaFree(d_output_grad));

        if (step == 0 || step == NUM_STEPS - 1) {
             MPI_CHECK(MPI_Barrier(MPI_COMM_WORLD)); // Sync before printing from rank 0
             if (rank == 0) {
                 std::cout << "\n--- Parameters after Step " << step + 1 << " (Rank 0 view) ---" << std::endl;
                 print_device_vector(model.d_W, std::min((size_t)10, model.W_size), "W", rank);
                 print_device_vector(model.d_b, std::min((size_t)10, model.b_size), "b", rank);
                 std::cout << "------------------------------------" << std::endl;
             }
        }

    }

    MPI_CHECK(MPI_Barrier(MPI_COMM_WORLD)); // Final barrier before cleanup
    if (rank == 0) {
        std::cout << "\n--- Training Loop Finished ---" << std::endl;
    }

    CUDA_CHECK(cudaFree(d_input));

    NCCL_CHECK(ncclCommDestroy(comm));

    MPI_CHECK(MPI_Finalize());
    return 0;
}
