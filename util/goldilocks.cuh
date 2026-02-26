
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

__device__ __forceinline__ void atomicAddGoldilocks(uint64_t* address, uint64_t val) {
    unsigned long long* address_as_ull = (unsigned long long*)address;
    unsigned long long old = *address_as_ull, assumed;
    do {
        assumed = old;
        uint64_t sum = goldilocks_add((uint64_t)assumed, val);
        old = atomicCAS(address_as_ull, assumed, (unsigned long long)sum);
    } while (assumed != old);
}

__global__ void changeMatrix(int N, uint64_t* A, const int row, const int col, uint64_t change) {
    A[row * N + col] = change;
}


