#include <cuda_runtime.h>
#include <vector>

__global__ void add_kernel(
    const float* a, 
    const float* b, 
    float* c, 
    int n
) {
    // c[i] = a[i] + b[i], guarded by i < n
    const int thread_idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (thread_idx >= n) return;

    c[thread_idx] = a[thread_idx] + b[thread_idx];
}

std::vector<float> vector_add(
    const std::vector<float>& a, 
    const std::vector<float>& b
) { 
    // Safety Checks
    if (a.size() != b.size()) return {};
    if (a.size() == 0 || b.size() == 0) return {};

    // allocate a, b, c on the device; copy a and b over (Host -> Device);
    const int n = a.size();
    const size_t bytes = n * sizeof(float);

    float* a_ptr = nullptr;
    float* b_ptr = nullptr;
    float* c_ptr = nullptr;

    cudaMalloc(&a_ptr, bytes);
    cudaMalloc(&b_ptr, bytes);
    cudaMalloc(&c_ptr, bytes);
    
    cudaMemcpy(a_ptr, a.data(), bytes, cudaMemcpyHostToDevice);
    cudaMemcpy(b_ptr, b.data(), bytes, cudaMemcpyHostToDevice);
    
    // launch the kernel; copy c back; free memory
    const int block = 256;
    const int grid  = (n + block - 1) / block;

    add_kernel<<<grid, block>>>(a_ptr, b_ptr, c_ptr, n);

    std::vector<float> c_out(n);

    cudaMemcpy(c_out.data(), c_ptr, bytes, cudaMemcpyDeviceToHost);

    cudaFree(a_ptr);
    cudaFree(b_ptr);
    cudaFree(c_ptr);

    return c_out;
}
