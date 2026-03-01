#include <iostream>
#include <vector>
#include <random>
#include <algorithm>
#include <map>
#include <chrono> 
#include <cuda_runtime.h>

#include "util/goldilocks.cuh"
#include "spmv/csr.cuh"
#include "spmv/coo.cuh" 

struct UpdateInfo {
    int r, c;
    uint64_t new_val;
};

void rebuild_csr_on_host(int N, const std::map<std::pair<int, int>, uint64_t>& sparse_map,
                         std::vector<int>& row_ptr,
                         std::vector<int>& col_ind,
                         std::vector<uint64_t>& vals) {
    std::fill(row_ptr.begin(), row_ptr.end(), 0);
    col_ind.clear();
    vals.clear();
    
    for (const auto& item : sparse_map) {
        row_ptr[item.first.first + 1]++;
        col_ind.push_back(item.first.second);
        vals.push_back(item.second);
    }

    for (int i = 0; i < N; i++) {
        row_ptr[i + 1] += row_ptr[i];
    }
}

__global__ void update_csr_gpu_kernel(int upd_idx, const int* upd_rows, const int* upd_cols, const uint64_t* upd_vals,
                                      int* row_ptr, int* col_ind, uint64_t* vals, 
                                      int N, int* d_nnz) {
    int r = upd_rows[upd_idx];
    int c = upd_cols[upd_idx];
    uint64_t v = upd_vals[upd_idx];

    int start = row_ptr[r];
    int end = row_ptr[r+1];
    bool found = false;
    int insert_pos = end;

    for (int i = start; i < end; i++) {
        if (col_ind[i] == c) {
            vals[i] = v;
            found = true;
            break;
        } else if (col_ind[i] > c) {
            insert_pos = i;
            break;
        }
    }

    if (!found) {
        int nnz = *d_nnz;
        
        for (int i = nnz; i > insert_pos; i--) {
            col_ind[i] = col_ind[i-1];
            vals[i] = vals[i-1];
        }
        
        col_ind[insert_pos] = c;
        vals[insert_pos] = v;

        for (int i = r + 1; i <= N; i++) {
            row_ptr[i]++;
        }
        *d_nnz = nnz + 1;
    }
}

__global__ void append_coo_buffer_kernel(int upd_idx, const int* upd_rows, const int* upd_cols, const uint64_t* upd_vals,
                                         int* buf_rows, int* buf_cols, uint64_t* buf_vals, int current_buffer_size) {
    int r = upd_rows[upd_idx];
    int c = upd_cols[upd_idx];
    uint64_t v = upd_vals[upd_idx];

    buf_rows[current_buffer_size] = r;
    buf_cols[current_buffer_size] = c;
    buf_vals[current_buffer_size] = v;
}

int main(int argc, char* argv[]) {
    if(argc != 2) {
        fprintf(stderr, "Enter the density! (e.g., 0.01 for 1%%)\n");
        return 1;
    }
    
    const int N = 4096;
    const float density = std::stof(argv[1]); 
    const int num_test_updates = 100;

    printf("=== Dynamic Sparse Update Benchmark (GPU-Side) ===\n");
    printf("Matrix: %d x %d, Density: %.2f%%\n", N, N, density * 100);
    printf("Testing %d updates sequentially on GPU...\n\n", num_test_updates);
    printf("Strategies:\n");
    printf("  1. CSR No-Buffer  : GPU Naive Insert & Shift (O(NNZ) bottleneck)\n");
    printf("  2. CSR Hybrid     : GPU COO Append (O(1) fast) -> Dual Kernels\n\n");

    std::map<std::pair<int, int>, uint64_t> sparse_map;
    std::vector<uint64_t> h_x(N, 1);

    std::mt19937 gen(1234);
    std::uniform_int_distribution<> dist_idx(0, N - 1);
    std::uniform_int_distribution<uint64_t> dist_val(1, 1000);

    int target_nnz = (int)(N * N * density);
    while (sparse_map.size() < target_nnz) {
        int r = dist_idx(gen);
        int c = dist_idx(gen);
        uint64_t v = dist_val(gen);
        sparse_map[{r, c}] = v;
    }

    std::vector<int> h_upd_rows(num_test_updates), h_upd_cols(num_test_updates);
    std::vector<uint64_t> h_upd_vals(num_test_updates);
    for (int i = 0; i < num_test_updates; i++) {
        h_upd_rows[i] = dist_idx(gen);
        h_upd_cols[i] = dist_idx(gen);
        h_upd_vals[i] = dist_val(gen);
    }

    int *d_upd_rows, *d_upd_cols;
    uint64_t *d_upd_vals;
    cudaMalloc(&d_upd_rows, num_test_updates * sizeof(int));
    cudaMalloc(&d_upd_cols, num_test_updates * sizeof(int));
    cudaMalloc(&d_upd_vals, num_test_updates * sizeof(uint64_t));
    cudaMemcpy(d_upd_rows, h_upd_rows.data(), num_test_updates * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_upd_cols, h_upd_cols.data(), num_test_updates * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_upd_vals, h_upd_vals.data(), num_test_updates * sizeof(uint64_t), cudaMemcpyHostToDevice);

    uint64_t *d_x, *d_y;
    cudaMalloc(&d_x, N * sizeof(uint64_t));
    cudaMalloc(&d_y, N * sizeof(uint64_t));
    cudaMemcpy(d_x, h_x.data(), N * sizeof(uint64_t), cudaMemcpyHostToDevice);

    std::vector<int> h_row_ptr(N + 1);
    std::vector<int> h_col_ind;
    std::vector<uint64_t> h_vals;
    rebuild_csr_on_host(N, sparse_map, h_row_ptr, h_col_ind, h_vals);

    int max_nnz = target_nnz + num_test_updates + 1000;
    int *d_row_ptr, *d_col_ind;
    uint64_t *d_vals;
    cudaMalloc(&d_row_ptr, (N + 1) * sizeof(int));
    cudaMalloc(&d_col_ind, max_nnz * sizeof(int));
    cudaMalloc(&d_vals, max_nnz * sizeof(uint64_t));
    
    cudaMemcpy(d_row_ptr, h_row_ptr.data(), (N + 1) * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_col_ind, h_col_ind.data(), h_vals.size() * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_vals, h_vals.data(), h_vals.size() * sizeof(uint64_t), cudaMemcpyHostToDevice);

    int* d_current_nnz;
    cudaMalloc(&d_current_nnz, sizeof(int));
    cudaMemcpy(d_current_nnz, &target_nnz, sizeof(int), cudaMemcpyHostToDevice);

    double total_time_csr_nobuf = 0.0;

    for (int i = 0; i < num_test_updates; i++) {
        auto start_cpu = std::chrono::high_resolution_clock::now();

        update_csr_gpu_kernel<<<1, 1>>>(i, d_upd_rows, d_upd_cols, d_upd_vals, 
                                        d_row_ptr, d_col_ind, d_vals, N, d_current_nnz);

        int warps_per_block = 256 / 32;
        int grid_size = (N + warps_per_block - 1) / warps_per_block;
        spmv_csr_vector_kernel<<<grid_size, 256>>>(N, d_row_ptr, d_col_ind, d_vals, d_x, d_y);

        cudaDeviceSynchronize();
        auto end_cpu = std::chrono::high_resolution_clock::now();
        total_time_csr_nobuf += std::chrono::duration<double, std::milli>(end_cpu - start_cpu).count();
    }

    cudaMemcpy(d_row_ptr, h_row_ptr.data(), (N + 1) * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_col_ind, h_col_ind.data(), h_vals.size() * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_vals, h_vals.data(), h_vals.size() * sizeof(uint64_t), cudaMemcpyHostToDevice);

    int *d_buf_rows, *d_buf_cols;
    uint64_t *d_buf_vals;
    cudaMalloc(&d_buf_rows, num_test_updates * sizeof(int));
    cudaMalloc(&d_buf_cols, num_test_updates * sizeof(int));
    cudaMalloc(&d_buf_vals, num_test_updates * sizeof(uint64_t));

    double total_time_hybrid = 0.0;

    for (int i = 0; i < num_test_updates; i++) {
        auto start_cpu = std::chrono::high_resolution_clock::now();

        int current_buffer_size = i;

        append_coo_buffer_kernel<<<1, 1>>>(i, d_upd_rows, d_upd_cols, d_upd_vals, 
                                           d_buf_rows, d_buf_cols, d_buf_vals, current_buffer_size);

        int warps_per_block = 256 / 32;
        int grid_size_csr = (N + warps_per_block - 1) / warps_per_block;
        spmv_csr_vector_kernel<<<grid_size_csr, 256>>>(N, d_row_ptr, d_col_ind, d_vals, d_x, d_y);

        int new_buffer_size = current_buffer_size + 1;
        if (new_buffer_size > 0) {
            int coo_blocks = (new_buffer_size + 255) / 256;
            spmv_coo_atomic_kernel<<<coo_blocks, 256>>>(new_buffer_size, d_buf_rows, d_buf_cols, d_buf_vals, d_x, d_y);
        }

        cudaDeviceSynchronize();
        auto end_cpu = std::chrono::high_resolution_clock::now();
        total_time_hybrid += std::chrono::duration<double, std::milli>(end_cpu - start_cpu).count();
    }

    printf("--------------------------------------------------\n");
    printf("[1] CSR No-Buffer (Shift) : %.3f ms / update\n", total_time_csr_nobuf / num_test_updates);
    printf("[2] CSR Hybrid (COO Buf)  : %.3f ms / update\n", total_time_hybrid / num_test_updates);
    printf("--------------------------------------------------\n");
    printf("Speedup (Hybrid vs Naive) : %.2f x\n", total_time_csr_nobuf / total_time_hybrid);

    cudaFree(d_x); cudaFree(d_y);
    cudaFree(d_upd_rows); cudaFree(d_upd_cols); cudaFree(d_upd_vals);
    cudaFree(d_row_ptr); cudaFree(d_col_ind); cudaFree(d_vals); cudaFree(d_current_nnz);
    cudaFree(d_buf_rows); cudaFree(d_buf_cols); cudaFree(d_buf_vals);

    return 0;
}
