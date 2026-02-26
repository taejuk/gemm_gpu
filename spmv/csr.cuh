#pragma once
#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include "../util/goldilocks.cuh"

#ifndef WARP_SIZE
#define WARP_SIZE 32
#endif

__global__ void spmv_csr_vector_kernel(int num_rows, 
                                       const int* row_ptr, 
                                       const int* col_ind, 
                                       const uint64_t* vals, 
                                       const uint64_t* x, 
                                       uint64_t* y) {
    int thread_id = blockDim.x * blockIdx.x + threadIdx.x;
    int warp_id   = thread_id / WARP_SIZE;
    int lane      = thread_id % WARP_SIZE;

    int row = warp_id;

    if (row < num_rows) {
        int row_start = row_ptr[row];
        int row_end   = row_ptr[row + 1];

        uint64_t sum = 0;

        for (int i = row_start + lane; i < row_end; i += WARP_SIZE) {
            uint64_t val = vals[i];
            int col      = col_ind[i];
            uint64_t term = goldilocks_mul(val, x[col]);
            
            sum = goldilocks_add(sum, term);
        }

        for (int offset = WARP_SIZE / 2; offset > 0; offset /= 2) {
            uint64_t friend_val = __shfl_down_sync(0xffffffff, sum, offset);
            sum = goldilocks_add(sum, friend_val);
        }
        if (lane == 0) {
            y[row] = sum;
        }
    }
}
