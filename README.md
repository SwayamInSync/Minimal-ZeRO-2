# Minimal ZeRO-2 CUDA/MPI/NCCL Prototype

This project provides a minimal, "from-scratch" implementation of the core concepts behind the **ZeRO (Zero Redundancy Optimizer) Stage 2** strategy using C++, CUDA, MPI, and NCCL. It demonstrates how to partition optimizer states and gradients across multiple GPUs to reduce memory redundancy during distributed training, allowing larger models to be handled.

This is intended as an educational prototype to understand the underlying mechanics, not as a production-ready, fully-featured library.

## Features

* Demonstrates ZeRO Stage 2 concepts:
    * Parameter replication.
    * Optimizer state partitioning (using SGD+Momentum as an example).
    * Gradient partitioning via `ncclReduceScatter`.
    * Parameter synchronization via `ncclAllGather`.
* Multi-GPU execution using MPI for initialization and NCCL for collective communication.
* Uses flattened buffers for gradients (`ReduceScatter`) and parameters (`AllGather`) for efficient communication.
* Includes a simple linear model and a basic training loop for demonstration.

## Dependencies

To build and run this project, you will need:

* **CMake:** Version 3.18 or higher recommended (older versions might have issues with CUDA/NCCL detection).
* **C++ Compiler:** Supporting C++17 (e.g., GCC 9+).
* **CUDA Toolkit:** Tested with 12.4. Ensure compatibility with your GPU driver and NCCL version.
* **MPI Implementation:** An MPI library compatible with your system and compiler (e.g., Open MPI, MPICH). `mpicxx` should be available.
* **NVIDIA NCCL Library:** The NCCL library compatible with your CUDA version and MPI implementation. **Crucially, the development package (`libnccl-dev` or `nccl-devel`) must be installed** to provide headers and potentially CMake configuration files.
* Configured for Nvidia-A100s but can be modified top differnet architectures inside the `CMakeLists.txt`

## Build Instructions

1.  **Clone the repository (if applicable).**
2.  **Ensure all dependencies are installed.**
3.  **Modify the build script`run.sh` as following**

    ```bash
    #!/bin/bash

    # --- Set Paths (IMPORTANT!) ---
    # Adjust CUDA_HOME to your CUDA Toolkit installation path
    export CUDA_HOME=/usr/local/cuda-12.4
    export PATH=$CUDA_HOME/bin:$PATH
    # Optional: Set LD_LIBRARY_PATH if libraries are not in standard locations
    # export LD_LIBRARY_PATH=$CUDA_HOME/lib64:/path/to/nccl/lib:$LD_LIBRARY_PATH

    # Optional: Set NCCL Path if not found automatically by CMake
    # Replace with the actual path where NCCL is installed (e.g., /usr or /usr/local/nccl-xyz)
    # export NCCL_INSTALL_PATH="/usr"

    echo "Using CUDA from: $CUDA_HOME"
    # echo "Using NCCL from: $NCCL_INSTALL_PATH" # Uncomment if setting NCCL path

    # --- Build Steps ---
    echo "Cleaning build directory..."
    rm -rf build
    mkdir build
    cd build

    echo "Running CMake..."

    # Add necessary -D flags if CMake has trouble finding components automatically
    # -D CMAKE_CUDA_COMPILER:PATH=$CUDA_HOME/bin/nvcc  # Usually needed if multiple CUDA versions exist
    # -D CUDA_TOOLKIT_ROOT_DIR:PATH=$CUDA_HOME        # Good practice hint
    # -D CMAKE_PREFIX_PATH:PATH=$NCCL_INSTALL_PATH    # Use if NCCL isn't found automatically

    cmake .. \
        -D CMAKE_CUDA_COMPILER:PATH=$CUDA_HOME/bin/nvcc \
        -D CUDA_TOOLKIT_ROOT_DIR:PATH=$CUDA_HOME
        # Add -D CMAKE_PREFIX_PATH=$NCCL_INSTALL_PATH if needed

    # Check CMake exit code
    if [ $? -ne 0 ]; then
      echo "CMake configuration failed!"
      exit 1
    fi

    echo "Running Make..."
    make -j

    # Check Make exit code
    if [ $? -ne 0 ]; then
      echo "Make build failed!"
      exit 1
    fi

    echo "Build successful!"
    cd ..
    ```
4.  **Make the script executable:** `chmod +x run.sh`
5.  **Run the script:** `bash run.sh`

## Running Instructions

Execute the compiled program using `mpirun` or `mpiexec`. Specify the number of processes (`-np`) equal to the number of GPUs you want to use.

```bash
# Example for 4 GPUs
mpirun -np 4 ./build/zero2_demo
```