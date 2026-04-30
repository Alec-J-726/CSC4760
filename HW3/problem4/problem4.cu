#include <cuda_runtime.h>

#include <cmath>
#include <iostream>

__global__ void add_vector_to_rows_kernel(const double* A, const double* B, double* C, int rows,
                                          int cols) {
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    const int total = rows * cols;
    if (idx < total) {
        const int col = idx % cols;
        C[idx] = A[idx] + B[col];
    }
}

static bool check_cuda(cudaError_t err, const char* msg) {
    if (err != cudaSuccess) {
        std::cerr << "CUDA error (" << msg << "): " << cudaGetErrorString(err) << "\n";
        return false;
    }
    return true;
}

int main() {
    constexpr int rows = 3;
    constexpr int cols = 3;
    constexpr int nA = rows * cols;
    constexpr int nB = cols;

    const double h_A[nA] = {
        130.0, 147.0, 115.0,  //
        224.0, 158.0, 187.0,  //
        54.0,  158.0, 120.0};
    const double h_B[nB] = {221.0, 12.0, 157.0};
    double h_C[nA] = {0.0};

    const double h_expected[nA] = {
        351.0, 159.0, 272.0,  //
        445.0, 170.0, 344.0,  //
        275.0, 170.0, 277.0};

    double *d_A = nullptr, *d_B = nullptr, *d_C = nullptr;
    if (!check_cuda(cudaMalloc(&d_A, nA * sizeof(double)), "cudaMalloc d_A") ||
        !check_cuda(cudaMalloc(&d_B, nB * sizeof(double)), "cudaMalloc d_B") ||
        !check_cuda(cudaMalloc(&d_C, nA * sizeof(double)), "cudaMalloc d_C")) {
        cudaFree(d_A);
        cudaFree(d_B);
        cudaFree(d_C);
        return 1;
    }

    if (!check_cuda(cudaMemcpy(d_A, h_A, nA * sizeof(double), cudaMemcpyHostToDevice),
                    "cudaMemcpy H2D A") ||
        !check_cuda(cudaMemcpy(d_B, h_B, nB * sizeof(double), cudaMemcpyHostToDevice),
                    "cudaMemcpy H2D B")) {
        cudaFree(d_A);
        cudaFree(d_B);
        cudaFree(d_C);
        return 1;
    }

    constexpr int threads = 128;
    const int blocks = (nA + threads - 1) / threads;
    add_vector_to_rows_kernel<<<blocks, threads>>>(d_A, d_B, d_C, rows, cols);

    if (!check_cuda(cudaGetLastError(), "kernel launch") ||
        !check_cuda(cudaDeviceSynchronize(), "kernel sync")) {
        cudaFree(d_A);
        cudaFree(d_B);
        cudaFree(d_C);
        return 1;
    }

    if (!check_cuda(cudaMemcpy(h_C, d_C, nA * sizeof(double), cudaMemcpyDeviceToHost),
                    "cudaMemcpy D2H C")) {
        cudaFree(d_A);
        cudaFree(d_B);
        cudaFree(d_C);
        return 1;
    }

    std::cout << "C (A + row-vector B):\n";
    for (int r = 0; r < rows; ++r) {
        for (int c = 0; c < cols; ++c) {
            std::cout << h_C[r * cols + c] << (c + 1 < cols ? '\t' : '\n');
        }
    }

    double max_abs_err = 0.0;
    for (int i = 0; i < nA; ++i) {
        max_abs_err = fmax(max_abs_err, fabs(h_C[i] - h_expected[i]));
    }
    std::cout << "Max abs error vs expected Soln: " << max_abs_err << "\n";

    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);
    return 0;
}
