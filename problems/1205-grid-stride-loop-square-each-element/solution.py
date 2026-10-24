#include <cuda_runtime.h>
#include <vector>

__global__ void square_kernel(const float* x, float* out, int n) {
    // Use a grid-stride loop:
    //   for (int i = start; i < n; i += stride) out[i] = x[i] * x[i];
    const int start  = blockIdx.x * blockDim.x + threadIdx.x;

    if (start >= n) return ;

    const int stride = blockDim.x * gridDim.x;

    for (int i=start; i < n; i += stride) {
        out[i] = x[i] * x[i];
    }
}

std::vector<float> square(const std::vector<float>& x) {
    if (x.empty()) return {};

    // Allocate Memory
    const int no_elements = x.size();
    const size_t bytes = no_elements * sizeof(float);

    float* squared_ptr = nullptr;
    float* x_ptr = nullptr;

    cudaMalloc(&squared_ptr, bytes);
    cudaMalloc(&x_ptr, bytes);

    // Transfer the information to the GPU
    cudaMemcpy(x_ptr, x.data(), bytes, cudaMemcpyHostToDevice);

    // Launch the kernel
    const int blocks = 2;
    const int threads_per_block = 4;

    square_kernel<<<blocks, threads_per_block>>>(x_ptr, squared_ptr, no_elements);

    // Copy back the results
    std::vector<float> squared(no_elements);

    cudaMemcpy(squared.data(), squared_ptr, bytes, cudaMemcpyDeviceToHost);

    cudaFree(squared_ptr);
    cudaFree(x_ptr);

    return squared;
}
