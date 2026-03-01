#include <iostream>
#include <vector>
#include <cuda_runtime.h>
#include <cmath>
#include <cuda_fp16.h>
#include <mma.h>
using namespace nvcuda;

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

#define TILE_SIZE 32
__global__ void tiled_gemm_kernel(float* A, float* B, float* C, int n) {
    __shared__ float tile_A[TILE_SIZE][TILE_SIZE];
    __shared__ float tile_B[TILE_SIZE][TILE_SIZE];
    int bx = blockIdx.x; int by = blockIdx.y;
    int tx = threadIdx.x; int ty = threadIdx.y;

    int row = by * TILE_SIZE + ty;
    int col = bx * TILE_SIZE + tx;
    float val = 0.0f;

    for (int k = 0; k < n; k += TILE_SIZE) {
        // 크기가 넘어갈 때에 대한 예외처리
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

__global__ void pack_float_to_half_kernel(float* src, half* packed_dst, int n) {
    int row = blockIdx.y * TILE_SIZE + threadIdx.y;
    int col = blockIdx.x * TILE_SIZE + threadIdx.x;

    if (row < n && col < n) {
        int src_idx = row * n + col;
	int grid_width = n / TILE_SIZE;
	int tile_idx = blockIdx.y * grid_width + blockIdx.x;
	int in_tile_idx = threadIdx.y * TILE_SIZE + threadIdx.x;
        int dst_idx = tile_idx * (TILE_SIZE * TILE_SIZE) + in_tile_idx;

	packed_dst[dst_idx] = __float2half(src[src_idx]);
    }

}

__global__ void packed_tiled_wmma_gemm_kernel(half* packed_A, half* packed_B, float* C, int n) {
    __shared__ half tile_A[TILE_SIZE][TILE_SIZE];
    __shared__ half tile_B[TILE_SIZE][TILE_SIZE];
    int bx = blockIdx.x; 
    int by = blockIdx.y;
    
    int tid = threadIdx.y * blockDim.x + threadIdx.x;
    int warp_id = tid / 32;

    int warps_per_row = TILE_SIZE / 16; 
    
    int warp_row = (warp_id / warps_per_row) * 16;
    int warp_col = (warp_id % warps_per_row) * 16;
    // global 행렬에서 위치가 어디에 있는지
    int global_row = by * TILE_SIZE + warp_row;
    int global_col = bx * TILE_SIZE + warp_col;
    
    wmma::fragment<wmma::matrix_a, 16, 16, 16, half, wmma::row_major> a_frag;
    wmma::fragment<wmma::matrix_b, 16, 16, 16, half, wmma::row_major> b_frag;
    wmma::fragment<wmma::accumulator, 16, 16, 16, float> c_frag;
    
    wmma::fill_fragment(c_frag, 0.0f);
    
    int grid_width = n / TILE_SIZE;
    int tile_elements = TILE_SIZE * TILE_SIZE;

    for (int k_idx = 0; k_idx < grid_width; k_idx++) {
	// shared memory로 가져온다.
        for (int i = tid; i < tile_elements; i += blockDim.x * blockDim.y) {
            int ty = i / TILE_SIZE;
            int tx = i % TILE_SIZE;

            int a_tile_offset = (by * grid_width + k_idx) * tile_elements;
            tile_A[ty][tx] = packed_A[a_tile_offset + i];

            int b_tile_offset = (k_idx * grid_width + bx) * tile_elements;
            tile_B[ty][tx] = packed_B[b_tile_offset + i];
        }
        __syncthreads();

        if (warp_row < TILE_SIZE && warp_col < TILE_SIZE) {
            for (int i = 0; i < TILE_SIZE; i += 16) {
                wmma::load_matrix_sync(a_frag, &tile_A[warp_row][i], TILE_SIZE);
                wmma::load_matrix_sync(b_frag, &tile_B[i][warp_col], TILE_SIZE);

                wmma::mma_sync(c_frag, a_frag, b_frag, c_frag);
            }
        }
        __syncthreads();
    }

    if (warp_row < TILE_SIZE && warp_col < TILE_SIZE) {
        if (global_row < n && global_col < n) {
            wmma::store_matrix_sync(C + global_row * n + global_col, c_frag, n, wmma::mem_row_major);
        }
    }
}



int main() {
    int n = 4096;
    size_t bytes = n * n * sizeof(float);

    std::vector<float> h_A(n * n, 1.0f);
    std::vector<float> h_B(n * n, 2.0f);
    std::vector<float> h_C(n * n, 0.0f);

    float *d_A, *d_B, *d_C;
    float *d_packed_A, *d_packed_B; // 패킹용 추가 메모리
    size_t half_bytes = n * n * sizeof(half);
    half *d_half_packed_A, *d_half_packed_B;
    float *d_C_wmma;
    
    cudaMalloc(&d_half_packed_A, half_bytes);
    cudaMalloc(&d_half_packed_B, half_bytes);
    cudaMalloc(&d_C_wmma, bytes);
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


    // 4. packed + tensor core

    cudaEventRecord(start);
    pack_float_to_half_kernel<<<blocks, threads>>>(d_A, d_half_packed_A, n);
    pack_float_to_half_kernel<<<blocks, threads>>>(d_B, d_half_packed_B, n);
    cudaEventRecord(stop); cudaEventSynchronize(stop);
    float pack_half_ms = 0; cudaEventElapsedTime(&pack_half_ms, start, stop);

    size_t threads_num = TILE_SIZE * TILE_SIZE * 32 / (16*16);
    dim3 threads_wmma(threads_num);
    cudaEventRecord(start);
    packed_tiled_wmma_gemm_kernel<<<blocks, threads_wmma>>>(d_half_packed_A, d_half_packed_B, d_C_wmma, n);
    cudaEventRecord(stop); cudaEventSynchronize(stop);
    float wmma_ms = 0; cudaEventElapsedTime(&wmma_ms, start, stop);
    float total_wmma_ms = pack_half_ms + wmma_ms;
    
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
    std::cout << "4. Tensor Core   : " << total_wmma_ms << " ms (" << (flops * 1e-9) / (total_wmma_ms * 1e-3) << " GFLOPS)\n";
    std::cout << "-----------------------------------" << std::endl;
    std::cout << "Tiled vs Naive Speedup  : " << naive_ms / tiled_ms << " x\n";
    std::cout << "Packed vs Tiled Speedup : " << tiled_ms / total_packed_ms << " x (Including Packing Overhead!)\n";
    std::cout << "WMMA vs Packed Speedup   : " << total_packed_ms / total_wmma_ms << " x\n";
    std::cout << "WMMA vs Naive Speedup    : " << naive_ms / total_wmma_ms << " x\n";
    cudaFree(d_A); cudaFree(d_B); cudaFree(d_C);
    cudaFree(d_packed_A); cudaFree(d_packed_B);
    cudaFree(d_half_packed_A); cudaFree(d_half_packed_B); cudaFree(d_C_wmma);
    return 0;
}
