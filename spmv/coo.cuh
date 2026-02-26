#pragma once
#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include "../util/goldilocks.cuh"

__global__ void spmv_coo_atomic_kernel(int num_updates, const int* rows, const int* cols, 
                                       const uint64_t* vals, const uint64_t* x, uint64_t* y) {
    int idx = blockDim.x * blockIdx.x + threadIdx.x;
    if (idx < num_updates) {
        int r = rows[idx];
        int c = cols[idx];
        uint64_t v = vals[idx];

        uint64_t product = goldilocks_mul(v, x[c]);
        atomicAddGoldilocks(&y[r], product);
    }
}
