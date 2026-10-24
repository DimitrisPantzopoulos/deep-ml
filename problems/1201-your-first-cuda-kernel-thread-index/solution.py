#include <cuda_runtime.h>
#include <vector>

__global__ void index_kernel(int* out, int n) {
    // Compute this thread's global index and, if it is < n, write it to out.
    const int thread_global_index = blockIdx.x * blockDim.x + threadIdx.x;

    if (thread_global_index < n) {
        out[thread_global_index] = thread_global_index;
    }
}

std::vector<int> global_thread_indices(int n) {
    if (n <= 0) return {};

    const size_t bytes = n * sizeof(int);

    // 1. allocate device memory for n ints
    int* d_out = nullptr;

    cudaMalloc(&d_out, bytes);

    // 2. launch the kernel with enough threads to cover n
    const int block = 256;
    const int grid  = (n - 1) / block + 1;

    index_kernel<<<grid, block>>>(d_out, n);

    // 3. copy the result back to the host and return it
    std::vector<int> h_out(n);
    
    cudaMemcpy(h_out.data(), d_out, bytes, cudaMemcpyDeviceToHost);
    cudaFree(d_out);

    return h_out;
}
