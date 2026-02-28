#include <iostream>
#include <vector>
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <mma.h>
#include <cmath>

using namespace nvcuda;

#define TILE_SIZE 16 // 텐서 코어는 16x16 단위를 사용합니다.

// -------------------------------------------------------------------------
// [유틸리티] float 초기화 및 half(FP16) 변환
// -------------------------------------------------------------------------
__global__ void init_half_matrix(half* mat, float val, int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) mat[idx] = __float2half(val);
}

// -------------------------------------------------------------------------
// 1. Naive WMMA (일반 Row-Major Global Memory에서 텐서 코어 바로 사용)
// -------------------------------------------------------------------------
__global__ void wmma_naive_kernel(half* A, half* B, float* C, int n) {
    int bx = blockIdx.x; 
    int by = blockIdx.y;
    int row = by * TILE_SIZE; 
    int col = bx * TILE_SIZE;

    wmma::fragment<wmma::matrix_a, TILE_SIZE, TILE_SIZE, TILE_SIZE, half, wmma::row_major> a_frag;
    wmma::fragment<wmma::matrix_b, TILE_SIZE, TILE_SIZE, TILE_SIZE, half, wmma::row_major> b_frag;
    wmma::fragment<wmma::accumulator, TILE_SIZE, TILE_SIZE, TILE_SIZE, float> c_frag;

    wmma::fill_fragment(c_frag, 0.0f);

    for (int k = 0; k < n; k += TILE_SIZE) {
        // Global Memory에서 직접 로드 (메모리 주소가 띄엄띄엄 떨어져 있음)
        wmma::load_matrix_sync(a_frag, A + row * n + k, n);
        wmma::load_matrix_sync(b_frag, B + k * n + col, n);
        
        wmma::mma_sync(c_frag, a_frag, b_frag, c_frag);
    }
    wmma::store_matrix_sync(C + row * n + col, c_frag, n, wmma::mem_row_major);
}

// -------------------------------------------------------------------------
// 2-1. Packing Kernel (Row-Major -> Block-Major로 완벽하게 연속된 메모리 배치)
// -------------------------------------------------------------------------
__global__ void pack_half_kernel(half* src, half* dst, int n) {
    int row = blockIdx.y * TILE_SIZE + threadIdx.y;
    int col = blockIdx.x * TILE_SIZE + threadIdx.x;

    if (row < n && col < n) {
        int src_idx = row * n + col;
        int grid_width = n / TILE_SIZE;
        
        // 16x16 타일 단위로 메모리 주소를 재정렬!
        int tile_idx = blockIdx.y * grid_width + blockIdx.x;
        int in_tile_idx = threadIdx.y * TILE_SIZE + threadIdx.x;
        
        int dst_idx = tile_idx * (TILE_SIZE * TILE_SIZE) + in_tile_idx;
        dst[dst_idx] = src[src_idx];
    }
}

// -------------------------------------------------------------------------
// 2-2. Ultimate GEMM (Packing + Tiling + Tensor Core)
// -------------------------------------------------------------------------
__global__ void ultimate_gemm_kernel(half* packed_A, half* packed_B, float* C, int n) {
    // [Tiling] Shared Memory 선언
    __shared__ half s_A[TILE_SIZE][TILE_SIZE];
    __shared__ half s_B[TILE_SIZE][TILE_SIZE];

    int bx = blockIdx.x; 
    int by = blockIdx.y;
    int tx = threadIdx.x; // 0 ~ 31 (1 Warp)

    int row = by * TILE_SIZE;
    int col = bx * TILE_SIZE;

    // [Tensor Core] 16x16x16 Fragment 선언
    wmma::fragment<wmma::matrix_a, TILE_SIZE, TILE_SIZE, TILE_SIZE, half, wmma::row_major> a_frag;
    wmma::fragment<wmma::matrix_b, TILE_SIZE, TILE_SIZE, TILE_SIZE, half, wmma::row_major> b_frag;
    wmma::fragment<wmma::accumulator, TILE_SIZE, TILE_SIZE, TILE_SIZE, float> c_frag;

    wmma::fill_fragment(c_frag, 0.0f);

    int grid_width = n / TILE_SIZE;
    int tile_elements = TILE_SIZE * TILE_SIZE; // 256 elements per tile

    for (int k_idx = 0; k_idx < grid_width; k_idx++) {
        
        // [Packing 시너지] 완벽하게 연속된 메모리 오프셋 계산
        int a_offset = (by * grid_width + k_idx) * tile_elements;
        int b_offset = (k_idx * grid_width + bx) * tile_elements;

        // Warp의 32개 스레드가 각각 8개씩 담당하여 256개의 요소를 Shared Memory로 초고속 로딩
        #pragma unroll
        for (int i = 0; i < 8; i++) {
            // Coalesced Access를 위해 tx + i * 32 패턴 사용
            int local_idx = tx + i * 32; 
            int r = local_idx / TILE_SIZE;
            int c = local_idx % TILE_SIZE;
            
            s_A[r][c] = packed_A[a_offset + local_idx];
            s_B[r][c] = packed_B[b_offset + local_idx];
        }
        __syncthreads(); // 모든 데이터가 Shared Memory에 올라올 때까지 대기

        // Shared Memory에서 Tensor Core 레지스터로 로딩 (초고속!)
        wmma::load_matrix_sync(a_frag, &s_A[0][0], TILE_SIZE);
        wmma::load_matrix_sync(b_frag, &s_B[0][0], TILE_SIZE);

        // 텐서 코어 연산 발사!
        wmma::mma_sync(c_frag, a_frag, b_frag, c_frag);
        __syncthreads();
    }

    // 최종 결과는 Row-Major 형태로 Global Memory에 바로 저장
    if (row < n && col < n) {
        wmma::store_matrix_sync(C + row * n + col, c_frag, n, wmma::mem_row_major);
    }
}

// -------------------------------------------------------------------------
// 메인 함수
// -------------------------------------------------------------------------
int main() {
    int n = 4096; // 4096 x 4096 크기
    size_t size_half = n * n * sizeof(half);
    size_t size_float = n * n * sizeof(float);

    half *d_A, *d_B, *d_packed_A, *d_packed_B;
    float *d_C_naive, *d_C_ult;

    cudaMalloc(&d_A, size_half); cudaMalloc(&d_B, size_half);
    cudaMalloc(&d_packed_A, size_half); cudaMalloc(&d_packed_B, size_half);
    cudaMalloc(&d_C_naive, size_float); cudaMalloc(&d_C_ult, size_float);

    // 디바이스에서 초기화 (CPU -> GPU 메모리 복사 시간 배제)
    int numElements = n * n;
    init_half_matrix<<<(numElements + 255) / 256, 256>>>(d_A, 1.0f, numElements);
    init_half_matrix<<<(numElements + 255) / 256, 256>>>(d_B, 2.0f, numElements);
    cudaDeviceSynchronize();

    cudaEvent_t start, stop;
    cudaEventCreate(&start); cudaEventCreate(&stop);
    double flops = 2.0 * pow(n, 3);

    // [실행 환경 설정] 텐서 코어의 WMMA는 32스레드(1워프) 단위로 동작
    dim3 threads_pack(TILE_SIZE, TILE_SIZE);
    dim3 blocks_pack(n / TILE_SIZE, n / TILE_SIZE);
    
    dim3 threads_wmma(32); // 1 Warp per Block
    dim3 blocks_wmma(n / TILE_SIZE, n / TILE_SIZE);

    // =====================================================================
    // 1. Naive WMMA 테스트
    // =====================================================================
    cudaEventRecord(start);
    wmma_naive_kernel<<<blocks_wmma, threads_wmma>>>(d_A, d_B, d_C_naive, n);
    cudaEventRecord(stop); cudaEventSynchronize(stop);
    float naive_ms = 0; cudaEventElapsedTime(&naive_ms, start, stop);

    // =====================================================================
    // 2. Ultimate GEMM (Pack + Tile + WMMA) 테스트
    // =====================================================================
    // 2-1. Packing Time
    cudaEventRecord(start);
    pack_half_kernel<<<blocks_pack, threads_pack>>>(d_A, d_packed_A, n);
    pack_half_kernel<<<blocks_pack, threads_pack>>>(d_B, d_packed_B, n);
    cudaEventRecord(stop); cudaEventSynchronize(stop);
    float pack_ms = 0; cudaEventElapsedTime(&pack_ms, start, stop);

    // 2-2. Ultimate Compute Time
    cudaEventRecord(start);
    ultimate_gemm_kernel<<<blocks_wmma, threads_wmma>>>(d_packed_A, d_packed_B, d_C_ult, n);
    cudaEventRecord(stop); cudaEventSynchronize(stop);
    float ult_ms = 0; cudaEventElapsedTime(&ult_ms, start, stop);

    float total_ult_ms = pack_ms + ult_ms;

    // =====================================================================
    // 결과 출력
    // =====================================================================
    std::cout << "=== The ULTIMATE GEMM Performance ===" << std::endl;
    std::cout << "Matrix Size: " << n << " x " << n << std::endl;
    std::cout << "-------------------------------------" << std::endl;
    std::cout << "1. Naive WMMA    : " << naive_ms << " ms (" << (flops * 1e-9) / (naive_ms * 1e-3) << " GFLOPS)\n";
    std::cout << "-------------------------------------" << std::endl;
    std::cout << "[ Packing Time   : " << pack_ms << " ms ]\n";
    std::cout << "[ Ultimate Calc  : " << ult_ms << " ms ]\n";
    std::cout << "2. Total Ultimate: " << total_ult_ms << " ms (" << (flops * 1e-9) / (total_ult_ms * 1e-3) << " GFLOPS)\n";
    std::cout << "-------------------------------------" << std::endl;
    std::cout << "Speedup (Ultimate vs Naive): " << naive_ms / total_ult_ms << " x\n";

    cudaFree(d_A); cudaFree(d_B); cudaFree(d_packed_A); cudaFree(d_packed_B);
    cudaFree(d_C_naive); cudaFree(d_C_ult);
    return 0;
}
