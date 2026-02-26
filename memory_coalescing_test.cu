#include <iostream>
#include <vector>
#include <random>
#include <cuda_runtime.h>
#include "util/goldilocks.cuh"
#include "dense/dense.cuh"

// 검증 함수
bool verify_results(int N, const std::vector<uint64_t>& h_y1, const std::vector<uint64_t>& h_y2) {
    for (int i = 0; i < N; i++) {
        if (h_y1[i] != h_y2[i]) {
            printf("Mismatch at index %d: %lu != %lu\n", i, h_y1[i], h_y2[i]);
            return false;
        }
    }
    return true;
}

int main() {
    // 1. 실험 설정
    const int N = 8192; // 8192 x 8192 행렬 (Coalescing 효과를 크게 보기 위해 사이즈를 키움)
    
    printf("=== Dense Matrix Memory Coalescing Benchmark ===\n");
    printf("Matrix Size: %d x %d\n", N, N);
    
    // 2. 데이터 생성 (CPU)
    std::vector<uint64_t> h_A(N * N);
    std::vector<uint64_t> h_x(N);
    std::vector<uint64_t> h_y_naive(N);
    std::vector<uint64_t> h_y_vector(N);

    // 랜덤 데이터 채우기 (Goldilocks 범위 내)
    std::mt19937 gen(1234);
    std::uniform_int_distribution<uint64_t> dist(1, GOLDILOCKS_P - 1);
    
    // Dense 행렬이므로 모든 값을 채웁니다.
    for (int i = 0; i < N * N; i++) h_A[i] = dist(gen);
    for (int i = 0; i < N; i++) h_x[i] = dist(gen);

    // 3. GPU 메모리 할당 및 복사
    uint64_t *d_A, *d_x, *d_y_naive, *d_y_vector;
    cudaMalloc(&d_A, N * N * sizeof(uint64_t));
    cudaMalloc(&d_x, N * sizeof(uint64_t));
    cudaMalloc(&d_y_naive, N * sizeof(uint64_t));
    cudaMalloc(&d_y_vector, N * sizeof(uint64_t));

    cudaMemcpy(d_A, h_A.data(), N * N * sizeof(uint64_t), cudaMemcpyHostToDevice);
    cudaMemcpy(d_x, h_x.data(), N * sizeof(uint64_t), cudaMemcpyHostToDevice);

    cudaEvent_t start, stop;
    cudaEventCreate(&start); cudaEventCreate(&stop);

    // ---------------------------------------------------------
    // Case 1: Naive Kernel (Uncoalesced Access)
    // ---------------------------------------------------------
    int blockSize = 256;
    int numBlocks = (N + blockSize - 1) / blockSize;

    // Warm-up
    dense_mv_kernel<<<numBlocks, blockSize>>>(N, d_A, d_x, d_y_naive);
    
    cudaEventRecord(start);
    // 반복 실행하여 평균 측정 (선택 사항)
    dense_mv_kernel<<<numBlocks, blockSize>>>(N, d_A, d_x, d_y_naive);
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    
    float time_naive = 0;
    cudaEventElapsedTime(&time_naive, start, stop);
    
    cudaMemcpy(h_y_naive.data(), d_y_naive, N * sizeof(uint64_t), cudaMemcpyDeviceToHost);

    // ---------------------------------------------------------
    // Case 2: Vector Kernel (Coalesced Access)
    // ---------------------------------------------------------
    // Grid 계산 중요: 워프 하나가 행 하나를 맡음
    int warp_size = 32;
    int num_warps = N; // 총 N개의 워프 필요
    int warps_per_block = blockSize / warp_size; // 256 / 32 = 8
    int vector_grid_size = (num_warps + warps_per_block - 1) / warps_per_block;

    // Warm-up
    dense_mv_vector_kernel<<<vector_grid_size, blockSize>>>(N, d_A, d_x, d_y_vector);

    cudaEventRecord(start);
    dense_mv_vector_kernel<<<vector_grid_size, blockSize>>>(N, d_A, d_x, d_y_vector);
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);

    float time_vector = 0;
    cudaEventElapsedTime(&time_vector, start, stop);

    cudaMemcpy(h_y_vector.data(), d_y_vector, N * sizeof(uint64_t), cudaMemcpyDeviceToHost);

    // ---------------------------------------------------------
    // 결과 출력
    // ---------------------------------------------------------
    printf("\n[Performance Result]\n");
    printf("1. Naive Kernel (Uncoalesced): %.3f ms\n", time_naive);
    printf("2. Vector Kernel (Coalesced) : %.3f ms\n", time_vector);
    printf(">> Speedup: %.2fx\n", time_naive / time_vector);

    // 정합성 검증
    if (verify_results(N, h_y_naive, h_y_vector)) {
        printf("\n[Success] Results match!\n");
    } else {
        printf("\n[Failure] Results mismatch!\n");
    }

    // 메모리 해제
    cudaFree(d_A); cudaFree(d_x); cudaFree(d_y_naive); cudaFree(d_y_vector);
    cudaEventDestroy(start); cudaEventDestroy(stop);
    
    return 0;
}
