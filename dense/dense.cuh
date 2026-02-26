#pragma once
#include <cuda_runtime.h>
#include "../util/goldilocks.cuh"

#ifndef WARP_SIZE
#define WARP_SIZE 32
#endif

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

__global__ void dense_mv_vector_kernel(int N, const uint64_t* A, const uint64_t* x, uint64_t* y) {
    int thread_id = blockDim.x * blockIdx.x + threadIdx.x;
    int warp_id   = thread_id / WARP_SIZE;
    int lane      = thread_id % WARP_SIZE;

    int row = warp_id; // 워프 하나가 행 하나 담당

    if (row < N) {
        uint64_t sum = 0;

        for (int col = lane; col < N; col += WARP_SIZE) {
            uint64_t val = A[row * N + col]; // 연속 메모리 접근

            if (val != 0) {
                sum = goldilocks_add(sum, goldilocks_mul(val, x[col]));
            }
        }

        // 3. Warp-level Reduction (부분합 합치기)
        for (int offset = WARP_SIZE / 2; offset > 0; offset /= 2) {
            uint64_t friend_val = __shfl_down_sync(0xffffffff, sum, offset);
            sum = goldilocks_add(sum, friend_val);
        }

        // 4. 결과 저장 (Lane 0만 수행)
        if (lane == 0) {
            y[row] = sum;
        }
    }
}
