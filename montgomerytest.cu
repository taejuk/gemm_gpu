#include <iostream>
#include <cstdint>
#include <cuda_runtime.h>

#define GOLDILOCKS_P 0xFFFFFFFF00000001ULL

// ------------------------------------------------------------------
// 1. [Bad] Naive Modulo Kernel
// ------------------------------------------------------------------
__global__ void kernel_standard_modulo(uint64_t a, uint64_t b, uint64_t* out, int iter) {
    uint64_t res = a;
    for (int i = 0; i < iter; i++) {
        // CUDA에서 128비트 연산 후 % 연산은 매우 무거운 소프트웨어 루틴을 호출함
        unsigned __int128 mul = (unsigned __int128)res * b;
        res = (uint64_t)(mul % GOLDILOCKS_P); 
    }
    *out = res;
}

// ------------------------------------------------------------------
// 2. [Good] Optimized Goldilocks Kernel (비트 연산)
// ------------------------------------------------------------------
__global__ void kernel_optimized_goldilocks(uint64_t a, uint64_t b, uint64_t* out, int iter) {
    uint64_t res = a;
    
    for (int i = 0; i < iter; i++) {
        unsigned __int128 n = (unsigned __int128)res * b;
        uint64_t lo = (uint64_t)n;
        uint64_t hi = (uint64_t)(n >> 64);

        // Reduction: res = lo - hi + hi * 2^32
        // = lo + (hi << 32) - hi
        // (중간 계산 overflow 방지를 위해 __int128 사용)
        unsigned __int128 t = (unsigned __int128)lo + ((unsigned __int128)hi << 32) - hi;
        
        // Final reduction logic (Branchless style preferred on GPU)
        uint64_t t_lo = (uint64_t)t;
        uint64_t t_hi = (uint64_t)(t >> 64);
        
        // t = t_lo + t_hi * 2^64 
        //   = t_lo + t_hi * (2^32 - 1)
        uint64_t final_val = t_lo + t_hi * 0xFFFFFFFFULL;

        // GPU에서는 Warp Divergence를 피하기 위해 if문 대신 
        // min() 같은 함수나 비트 연산을 쓰는 게 좋지만, 
        // 여기선 비교를 위해 일단 if문 유지 (그래도 %보단 빠름)
        if (final_val >= GOLDILOCKS_P) final_val -= GOLDILOCKS_P;
        
        res = final_val;
    }
    *out = res;
}

int main() {
    uint64_t h_a = 123456789;
    uint64_t h_b = 987654321;
    uint64_t *d_out;
    cudaMalloc(&d_out, sizeof(uint64_t));

    int iterations = 1000000; // 100만 번 반복

    // === 1. Standard Modulo 측정 ===
    cudaEvent_t start, stop;
    cudaEventCreate(&start); cudaEventCreate(&stop);

    cudaEventRecord(start);
    kernel_standard_modulo<<<1, 1>>>(h_a, h_b, d_out, iterations);
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);

    float time_standard = 0;
    cudaEventElapsedTime(&time_standard, start, stop);
    printf("GPU Standard Modulo (%%): %.3f ms\n", time_standard);

    // === 2. Optimized Goldilocks 측정 ===
    cudaEventRecord(start);
    kernel_optimized_goldilocks<<<1, 1>>>(h_a, h_b, d_out, iterations);
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);

    float time_optimized = 0;
    cudaEventElapsedTime(&time_optimized, start, stop);
    printf("GPU Optimized Goldilocks: %.3f ms\n", time_optimized);
    
    printf("Speedup: %.2fx\n", time_standard / time_optimized);

    cudaFree(d_out);
    return 0;
}