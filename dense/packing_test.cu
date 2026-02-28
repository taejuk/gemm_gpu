#include <iostream>
#include <vector>
#include <cuda_runtime.h>
#include <cmath>

#define TILE_SIZE 32

// -------------------------------------------------------------------------
// 1. Naive GEMM (최적화 없음)
// -------------------------------------------------------------------------
__global__ void naive_gemm_kernel(float* A, float* B, float* C, int n) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (row < n && col < n) {
        float val = 0.0f;
        for (int i = 0; i < n; i++) {
            val += A[row * n + i] * B[i * n + col];
        }
        C[row * n + col] = val;
    }
}

// -------------------------------------------------------------------------
// 2. Tiled GEMM (기존 사용자 코드)
// -------------------------------------------------------------------------
__global__ void tiled_gemm_kernel(float* A, float* B, float* C, int n) {
    __shared__ float tile_A[TILE_SIZE][TILE_SIZE + 1];
    __shared__ float tile_B[TILE_SIZE][TILE_SIZE + 1];

    int bx = blockIdx.x; int by = blockIdx.y;
    int tx = threadIdx.x; int ty = threadIdx.y;

    int row = by * TILE_SIZE + ty;
    int col = bx * TILE_SIZE + tx;
    float val = 0.0f;

    for (int k = 0; k < n; k += TILE_SIZE) {
        if (row < n && (k + tx) < n) tile_A[ty][tx] = A[row * n + (k + tx)];
        else tile_A[ty][tx] = 0.0f;

        if (col < n && (k + ty) < n) tile_B[ty][tx] = B[(k + ty) * n + col];
        else tile_B[ty][tx] = 0.0f;

        __syncthreads();

        for (int i = 0; i < TILE_SIZE; i++) {
            val += tile_A[ty][i] * tile_B[i][tx];
        }
        __syncthreads();
    }
    if (row < n && col < n) C[row * n + col] = val;
}

// -------------------------------------------------------------------------
// 3-1. Packing Kernel (데이터를 타일 크기로 연속되게 재배치)
// -------------------------------------------------------------------------
__global__ void pack_matrix_kernel(float* src, float* packed_dst, int n) {
    int row = blockIdx.y * TILE_SIZE + threadIdx.y;
    int col = blockIdx.x * TILE_SIZE + threadIdx.x;

    if (row < n && col < n) {
        int src_idx = row * n + col;
        int grid_width = n / TILE_SIZE;
        int tile_idx = blockIdx.y * grid_width + blockIdx.x;
        int in_tile_idx = threadIdx.y * TILE_SIZE + threadIdx.x;
        int dst_idx = tile_idx * (TILE_SIZE * TILE_SIZE) + in_tile_idx;

        packed_dst[dst_idx] = src[src_idx];
    }
}

// -------------------------------------------------------------------------
// 3-2. Packed Tiled GEMM (패킹된 데이터를 사용)
// -------------------------------------------------------------------------
__global__ void packed_tiled_gemm_kernel(float* packed_A, float* packed_B, float* C, int n) {
    __shared__ float tile_A[TILE_SIZE][TILE_SIZE];
    __shared__ float tile_B[TILE_SIZE][TILE_SIZE];

    int bx = blockIdx.x; int by = blockIdx.y;
    int tx = threadIdx.x; int ty = threadIdx.y;

    int row = by * TILE_SIZE + ty;
    int col = bx * TILE_SIZE + tx;
    float val = 0.0f;
    int grid_width = n / TILE_SIZE;
    int tile_elements = TILE_SIZE * TILE_SIZE;

    for (int k_idx = 0; k_idx < grid_width; k_idx++) {
        // 완벽하게 연속된 메모리 주소에서 초고속 로딩
        int a_tile_offset = (by * grid_width + k_idx) * tile_elements;
        tile_A[ty][tx] = packed_A[a_tile_offset + (ty * TILE_SIZE + tx)];

        int b_tile_offset = (k_idx * grid_width + bx) * tile_elements;
        tile_B[ty][tx] = packed_B[b_tile_offset + (ty * TILE_SIZE + tx)];

        __syncthreads();

        for (int i = 0; i < TILE_SIZE; i++) {
            val += tile_A[ty][i] * tile_B[i][tx];
        }
        __syncthreads();
    }
    if (row < n && col < n) C[row * n + col] = val;
}

// -------------------------------------------------------------------------
// 메인 함수
// -------------------------------------------------------------------------
int main() {
    int n = 4096; // 행렬 크기 (메모리 대역폭 차이를 보기 위해 크게 설정)
    size_t bytes = n * n * sizeof(float);

    std::vector<float> h_A(n * n, 1.0f);
    std::vector<float> h_B(n * n, 2.0f);
    std::vector<float> h_C(n * n, 0.0f);

    float *d_A, *d_B, *d_C;
    float *d_packed_A, *d_packed_B; // 패킹용 추가 메모리

    cudaMalloc(&d_A, bytes); cudaMalloc(&d_B, bytes); cudaMalloc(&d_C, bytes);
    cudaMalloc(&d_packed_A, bytes); cudaMalloc(&d_packed_B, bytes);

    cudaMemcpy(d_A, h_A.data(), bytes, cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, h_B.data(), bytes, cudaMemcpyHostToDevice);

    dim3 threads(TILE_SIZE, TILE_SIZE);
    dim3 blocks((n + TILE_SIZE - 1) / TILE_SIZE, (n + TILE_SIZE - 1) / TILE_SIZE);

    cudaEvent_t start, stop;
    cudaEventCreate(&start); cudaEventCreate(&stop);

    double flops = 2.0 * pow(n, 3); // 총 부동소수점 연산량

    // =====================================================================
    // 1. Naive GEMM
    // =====================================================================
    cudaEventRecord(start);
    naive_gemm_kernel<<<blocks, threads>>>(d_A, d_B, d_C, n);
    cudaEventRecord(stop); cudaEventSynchronize(stop);
    float naive_ms = 0; cudaEventElapsedTime(&naive_ms, start, stop);

    // =====================================================================
    // 2. Tiled GEMM
    // =====================================================================
    cudaEventRecord(start);
    tiled_gemm_kernel<<<blocks, threads>>>(d_A, d_B, d_C, n);
    cudaEventRecord(stop); cudaEventSynchronize(stop);
    float tiled_ms = 0; cudaEventElapsedTime(&tiled_ms, start, stop);

    // =====================================================================
    // 3. Packed Tiled GEMM
    // =====================================================================
    // 3-1. Packing Time 측정
    cudaEventRecord(start);
    pack_matrix_kernel<<<blocks, threads>>>(d_A, d_packed_A, n);
    pack_matrix_kernel<<<blocks, threads>>>(d_B, d_packed_B, n);
    cudaEventRecord(stop); cudaEventSynchronize(stop);
    float pack_ms = 0; cudaEventElapsedTime(&pack_ms, start, stop);

    // 3-2. Packed GEMM 연산 시간 측정
    cudaEventRecord(start);
    packed_tiled_gemm_kernel<<<blocks, threads>>>(d_packed_A, d_packed_B, d_C, n);
    cudaEventRecord(stop); cudaEventSynchronize(stop);
    float packed_ms = 0; cudaEventElapsedTime(&packed_ms, start, stop);

    float total_packed_ms = pack_ms + packed_ms;

    // =====================================================================
    // 결과 출력
    // =====================================================================
    std::cout << "=== GEMM Performance Comparison ===" << std::endl;
    std::cout << "Matrix Size: " << n << " x " << n << std::endl;
    std::cout << "-----------------------------------" << std::endl;
    std::cout << "1. Naive GEMM    : " << naive_ms << " ms (" << (flops * 1e-9) / (naive_ms * 1e-3) << " GFLOPS)\n";
    std::cout << "2. Tiled GEMM    : " << tiled_ms << " ms (" << (flops * 1e-9) / (tiled_ms * 1e-3) << " GFLOPS)\n";
    std::cout << "-----------------------------------" << std::endl;
    std::cout << "[ Packing Time   : " << pack_ms << " ms ]\n";
    std::cout << "[ Packed Compute : " << packed_ms << " ms ]\n";
    std::cout << "3. Total Packed  : " << total_packed_ms << " ms (" << (flops * 1e-9) / (total_packed_ms * 1e-3) << " GFLOPS)\n";
    std::cout << "-----------------------------------" << std::endl;
    std::cout << "Tiled vs Naive Speedup  : " << naive_ms / tiled_ms << " x\n";
    std::cout << "Packed vs Tiled Speedup : " << tiled_ms / total_packed_ms << " x (Including Packing Overhead!)\n";

    cudaFree(d_A); cudaFree(d_B); cudaFree(d_C);
    cudaFree(d_packed_A); cudaFree(d_packed_B);
    return 0;
}
