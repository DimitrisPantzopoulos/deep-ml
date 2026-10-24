#include <cuda_runtime.h>
#include <vector>

__global__ void matadd_kernel(
        const float* A, 
        const float* B, 
        float* C, 
        int rows, 
        int cols
    ) {
    // Compute (row, col) from the 2D thread index; guard row<rows && col<cols;
    // C[row*cols + col] = A[row*cols + col] + B[row*cols + col];

    const int col = blockIdx.x * blockDim.x + threadIdx.x;
    const int row = blockIdx.y * blockDim.y + threadIdx.y;

    if (row >= rows || col >= cols) return;

    const int idx = row * cols + col;

    C[idx] = A[idx] + B[idx];
}

std::vector<float> flatten_matrix(
    const std::vector<std::vector<float>>& x,
    const int rows, 
    const int cols
) {
    std::vector<float> flat; flat.reserve(rows * cols);

    for (const auto& row : x) {
        flat.insert(flat.end(), row.begin(), row.end());
    }

    return flat;
}

std::vector<float> matrix_add(
        const std::vector<std::vector<float>>& A,
        const std::vector<std::vector<float>>& B
) {
    if (A.empty() || A.front().empty()) return {};

    // flatten A and B to row-major 1D, run the kernel, return C flattened
    const int rows = A.size();
    const int cols = A.front().size();

    const auto flat_A = flatten_matrix(A, rows, cols);
    const auto flat_B = flatten_matrix(B, rows, cols);

    // Allocate Memory
    const int no_elements = flat_A.size();
    const size_t bytes = no_elements * sizeof(float);

    float* A_ptr = nullptr;
    float* B_ptr = nullptr;
    float* C_ptr = nullptr;

    cudaMalloc(&A_ptr, bytes);
    cudaMalloc(&B_ptr, bytes);
    cudaMalloc(&C_ptr, bytes);

    // Transfer the Memory to the GPU
    cudaMemcpy(A_ptr, flat_A.data(), bytes, cudaMemcpyHostToDevice);
    cudaMemcpy(B_ptr, flat_B.data(), bytes, cudaMemcpyHostToDevice);

    // Launch the Kernel
    const dim3 block(16, 16);

    const dim3 grid(
        (cols + block.x - 1) / block.x,
        (rows + block.y - 1) / block.y
    );

    matadd_kernel<<<grid, block>>>(A_ptr, B_ptr, C_ptr, rows, cols);

    // Copy the data and free memory
    std::vector<float> c_out(no_elements);

    cudaMemcpy(c_out.data(), C_ptr, bytes, cudaMemcpyDeviceToHost);

    cudaFree(A_ptr);
    cudaFree(B_ptr);
    cudaFree(C_ptr);

    return c_out;
}
