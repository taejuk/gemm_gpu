#pragma once
#include <cuda_runtime.h>
#include "../util/goldilocks.cuh"

__global__ void dense_mv_kernel(int N, const uint64_t* A, const uint64_t* x, uint64_t* y) {
    int row = blockDim.x * blockIdx.x + threadIdx.x;
    if (row < N) {
        uint64_t sum = 0;
        for (int col = 0; col < N; col++) {
            uint64_t val = A[row * N + col];
            if (val != 0) { // 0이어도 읽긴 읽어야 함 (Dense니까)
                sum = goldilocks_add(sum, goldilocks_mul(val, x[col]));
            }
        }
        y[row] = sum;
    }
}
