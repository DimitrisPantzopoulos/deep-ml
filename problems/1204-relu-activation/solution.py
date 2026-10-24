#include <cuda_runtime.h>
#include <vector>

__global__ void relu_kernel(const float* x, float* out, int n) {
    // out[i] = max(0, x[i])
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx >= n) return;

    out[idx] = fmaxf(0.0f, x[idx]);
}

std::vector<float> relu(const std::vector<float>& x) {
    const int no_elements = x.size();

    if (no_elements == 0) return {};

    // Allocate memory in the GPU for ReLU
    const size_t bytes = no_elements * sizeof(float);

    float* relu_ptr = nullptr;
    float* x_ptr = nullptr;

    cudaMalloc(&relu_ptr, bytes);
    cudaMalloc(&x_ptr, bytes);

    // Send the information to the GPU
    cudaMemcpy(x_ptr, x.data(), bytes, cudaMemcpyHostToDevice);

    const int block = 256;
    const int grid  = (no_elements + block - 1) / block;

    relu_kernel<<<grid, block>>>(x_ptr, relu_ptr, no_elements);

    // Copy out the results of the ReLU
    std::vector<float> relu_out(no_elements);

    cudaMemcpy(relu_out.data(), relu_ptr, bytes, cudaMemcpyDeviceToHost);

    // Free the memory
    cudaFree(relu_ptr);
    cudaFree(x_ptr);

    return relu_out;
}
