#!/bin/bash
export NCCL_DEBUG=INFO
export CUDA_HOME=/usr/local/cuda-12.4
export PATH=$CUDA_HOME/bin:$PATH

echo "Using CUDA from: $CUDA_HOME"
echo "Updated PATH: $PATH"


echo "Cleaning build directory..."
rm -rf build
mkdir build
cd build

echo "Running CMake (forcing CUDA compiler)..."

cmake .. \
    -D CMAKE_CUDA_COMPILER:PATH=$CUDA_HOME/bin/nvcc \
    -D CUDA_TOOLKIT_ROOT_DIR:PATH=$CUDA_HOME

if [ $? -ne 0 ]; then
  echo "CMake configuration failed!"
  exit 1
fi

echo "Running Make..."
make -j

if [ $? -ne 0 ]; then
  echo "Make build failed!"
  exit 1
fi

echo "Build successful!"
cd ..