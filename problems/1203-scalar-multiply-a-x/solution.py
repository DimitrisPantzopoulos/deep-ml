#include <cuda_runtime.h>
#include <vector>

__global__ void scale_kernel(
        const float* x, 
        float a, 
        float* y, 
        int n
    ) {
    // y[i] = a * x[i]

    const int thread_idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (thread_idx >= n) return;

    y[thread_idx] = a * x[thread_idx];
}

std::vector<float> scalar_multiply(const std::vector<float>& x, float a) {
    const int n = x.size();

    if (n == 0) return {};
    if (a == 0) return std::vector<float>(n, 0.0f);

    // Allocate Memory on the GPU to scale multiply.
    const int bytes = n * sizeof(float);

    float* y_ptr = nullptr;

    cudaMalloc(&y_ptr, bytes);

    // Move the data to the GPU
    float* x_ptr = nullptr;

    cudaMalloc(&x_ptr, bytes);

    cudaMemcpy(x_ptr, x.data(), bytes, cudaMemcpyHostToDevice);

    // Launch the Kernel
    const int block = 256;
    const int grid  = (n + block - 1) / block;

    scale_kernel<<<grid, block>>>(x_ptr, a, y_ptr, n);

    // Copy back the data and Free
    std::vector<float> scaled_out(n);

    cudaMemcpy(scaled_out.data(), y_ptr, bytes, cudaMemcpyDeviceToHost);

    cudaFree(y_ptr);
    cudaFree(x_ptr);

    return scaled_out;
}
