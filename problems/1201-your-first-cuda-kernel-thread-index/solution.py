#include <cuda_runtime.h>
#include <vector>

__global__ void index_kernel(int* out, int n) {
    // Compute this thread's global index and, if it is < n, write it to out.
    int kernel_idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (kernel_idx < n) {
        out[kernel_idx] = kernel_idx;
    } 
}

std::vector<int> global_thread_indices(int n) {
    std::vector<int> h_out(n);

    if (n <= 0) { return h_out; }

    // 1. allocate device memory for n ints
    int* d_out;
    cudaMalloc(&d_out, n * sizeof(int));

    // 2. launch the kernel with enough threads to cover n
    int block = 256;
    int grid  = (n + block - 1) / block;

    index_kernel<<<grid, block>>>(d_out, n);

    // 3. copy the result back to the host and return it
    cudaMemcpy(h_out.data(), d_out, n * sizeof(int), cudaMemcpyDeviceToHost);
    cudaFree(d_out);
    
    return h_out;
}
