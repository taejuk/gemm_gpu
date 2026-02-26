
#pragma once
#include <cuda_runtime.h>

#define GOLDILOCKS_P 0xFFFFFFFF00000001ULL

__device__ __forceinline__ uint64_t goldilocks_mul(uint64_t a, uint64_t b) {
    unsigned __int128 n = (unsigned __int128)a * b;
    uint64_t lo = (uint64_t)n;
    uint64_t hi = (uint64_t)(n >> 64);

    // Reduction: res = lo - hi + hi * 2^32
    unsigned __int128 t = (unsigned __int128)lo + ((unsigned __int128)hi << 32) - hi;

    // Final reduction
    uint64_t r0 = (uint64_t)t;
    uint64_t r1 = (uint64_t)(t >> 64);
    uint64_t res = r0 + r1 * 0xFFFFFFFFULL;

    if (res >= GOLDILOCKS_P) res -= GOLDILOCKS_P;
    if (res >= GOLDILOCKS_P) res -= GOLDILOCKS_P; // Safety check
    return res;
}

__device__ __forceinline__ uint64_t goldilocks_add(uint64_t a, uint64_t b) {
    unsigned __int128 res = (unsigned __int128)a + b;
    if (res >= GOLDILOCKS_P) res -= GOLDILOCKS_P;
    return (uint64_t)res;
}

__global__ void dense_mv_kernel(int N, const uint64_t* A, const uint64_t* x, uint64_t* y) {
    int row = blockDim.x * blockIdx.x + threadIdx.x;
    if (row < N) {
        uint64_t sum = 0;
        for (int col = 0; col < N; col++) {
            uint64_t val = A[row * N + col];
            if (val != 0) {
                sum = goldilocks_add(sum, goldilocks_mul(val, x[col]));
            }
        }
        y[row] = sum;
    }
}

__global__ void changeMatrix(int N, uint64_t* A, const int row, const int col, uint64_t change) {
    A[row * N + col] = change;
}


