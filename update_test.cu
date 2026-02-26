#include <iostream>
#include <vector>
#include <random>
#include <algorithm>
#include <map>
#include <chrono> // CPU 시간 측정을 위해 사용
#include <cuda_runtime.h>

// 기존 헤더 파일 포함
#include "util/goldilocks.cuh"
#include "dense/dense.cuh"
#include "spmv/csr.cuh"
// COO 커널 헤더가 없다면 아래 커널 정의를 사용, 있다면 include 하세요.
#include "spmv/coo.cuh" 

// -----------------------------------------------------------------------------
// Helper: Update Info Structure
// -----------------------------------------------------------------------------
struct UpdateInfo {
    int r, c;
    uint64_t new_val;
};

// -----------------------------------------------------------------------------
// Kernel: Dense Partial Update (GPU에서 바로 수정)
// -----------------------------------------------------------------------------
__global__ void update_dense_matrix_kernel(int num_updates, const int* rows, const int* cols, 
                                           const uint64_t* vals, uint64_t* A, int N) {
    int idx = blockDim.x * blockIdx.x + threadIdx.x;
    if (idx < num_updates) {
        // 행 우선(Row-major) 인덱싱으로 해당 위치 값만 갱신
        A[rows[idx] * N + cols[idx]] = vals[idx];
    }
}

// -----------------------------------------------------------------------------
// Helper: Rebuild CSR from Map (CPU 작업)
// -----------------------------------------------------------------------------
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

// -----------------------------------------------------------------------------
// Main Benchmark
// -----------------------------------------------------------------------------
int main(int argc, char* argv[]) {
    if(argc != 2) {
        fprintf(stderr, "Enter the density!\n");
	return 1;
    }
    // 1. 설정
    const int N = 4096;
    const float density = std::stof(argv[1]); // 1% Sparsity
    const int num_test_updates = 100; // 100번 업데이트 반복 측정

    printf("=== Update Cost Benchmark (Dense vs CSR vs Hybrid) ===\n");
    printf("Matrix: %d x %d, Density: %.2f%%\n", N, N, density * 100);
    printf("Testing %d updates sequentially...\n\n", num_test_updates);
    printf("Strategies:\n");
    printf("  1. Dense Optimized: Partial Update (GPU In-place)\n");
    printf("  2. CSR No-Buffer  : Host Rebuild -> Full Copy\n");
    printf("  3. CSR Hybrid     : Append to GPU Buffer -> Dual Kernels\n\n");

    // 2. 데이터 초기화
    std::map<std::pair<int, int>, uint64_t> sparse_map;
    std::vector<uint64_t> h_A_dense(N * N, 0);
    std::vector<uint64_t> h_x(N, 1);

    std::mt19937 gen(1234);
    std::uniform_int_distribution<> dist_idx(0, N - 1);
    std::uniform_int_distribution<uint64_t> dist_val(1, 1000);

    // 초기 데이터 생성
    int target_nnz = (int)(N * N * density);
    while (sparse_map.size() < target_nnz) {
        int r = dist_idx(gen);
        int c = dist_idx(gen);
        uint64_t v = dist_val(gen);
        sparse_map[{r, c}] = v;
        h_A_dense[r * N + c] = v;
    }

    // 업데이트 목록 미리 생성
    std::vector<UpdateInfo> updates;
    for (int i = 0; i < num_test_updates; i++) {
        updates.push_back({dist_idx(gen), dist_idx(gen), dist_val(gen)});
    }

    // GPU 기본 메모리 할당
    uint64_t *d_x, *d_y;
    cudaMalloc(&d_x, N * sizeof(uint64_t));
    cudaMalloc(&d_y, N * sizeof(uint64_t));
    cudaMemcpy(d_x, h_x.data(), N * sizeof(uint64_t), cudaMemcpyHostToDevice);

    // =========================================================================
    // Experiment 1: Dense Optimized Update (Partial Update)
    // =========================================================================
    uint64_t* d_A_dense;
    cudaMalloc(&d_A_dense, N * N * sizeof(uint64_t));
    cudaMemcpy(d_A_dense, h_A_dense.data(), N * N * sizeof(uint64_t), cudaMemcpyHostToDevice);

    int *d_upd_row, *d_upd_col;
    uint64_t *d_upd_val;
    cudaMalloc(&d_upd_row, sizeof(int));
    cudaMalloc(&d_upd_col, sizeof(int));
    cudaMalloc(&d_upd_val, sizeof(uint64_t));

    double total_time_dense = 0.0;

    for (const auto& u : updates) {
        auto start_cpu = std::chrono::high_resolution_clock::now();

        // 1. 전송 (1 element)
        cudaMemcpy(d_upd_row, &u.r, sizeof(int), cudaMemcpyHostToDevice);
        cudaMemcpy(d_upd_col, &u.c, sizeof(int), cudaMemcpyHostToDevice);
        cudaMemcpy(d_upd_val, &u.new_val, sizeof(uint64_t), cudaMemcpyHostToDevice);

        // 2. 수정 커널
        update_dense_matrix_kernel<<<1, 1>>>(1, d_upd_row, d_upd_col, d_upd_val, d_A_dense, N);

        // 3. SpMV
        int blockSize = 256;
        int numBlocks = (N + blockSize - 1) / blockSize;
        dense_mv_kernel<<<numBlocks, blockSize>>>(N, d_A_dense, d_x, d_y);
        
        cudaDeviceSynchronize();
        auto end_cpu = std::chrono::high_resolution_clock::now();
        total_time_dense += std::chrono::duration<double, std::milli>(end_cpu - start_cpu).count();
    }

    // =========================================================================
    // Experiment 2: CSR No-Buffer (Rebuild on Host + Full Copy)
    // =========================================================================
    // Temp copy of map for Exp 2 so we don't affect Exp 3
    auto map_exp2 = sparse_map; 
    
    std::vector<int> h_row_ptr(N + 1);
    std::vector<int> h_col_ind;
    std::vector<uint64_t> h_vals;
    h_col_ind.reserve(target_nnz + num_test_updates);
    h_vals.reserve(target_nnz + num_test_updates);

    int *d_row_ptr, *d_col_ind;
    uint64_t *d_vals;
    int max_nnz = target_nnz + num_test_updates + 1000;
    cudaMalloc(&d_row_ptr, (N + 1) * sizeof(int));
    cudaMalloc(&d_col_ind, max_nnz * sizeof(int));
    cudaMalloc(&d_vals, max_nnz * sizeof(uint64_t));

    double total_time_csr_nobuf = 0.0;

    for (const auto& u : updates) {
        auto start_cpu = std::chrono::high_resolution_clock::now();

        // 1. CPU Map Update & Rebuild
        map_exp2[{u.r, u.c}] = u.new_val;
        rebuild_csr_on_host(N, map_exp2, h_row_ptr, h_col_ind, h_vals);
        int current_nnz = h_vals.size();

        // 2. Full Copy
        cudaMemcpy(d_row_ptr, h_row_ptr.data(), (N + 1) * sizeof(int), cudaMemcpyHostToDevice);
        cudaMemcpy(d_col_ind, h_col_ind.data(), current_nnz * sizeof(int), cudaMemcpyHostToDevice);
        cudaMemcpy(d_vals, h_vals.data(), current_nnz * sizeof(uint64_t), cudaMemcpyHostToDevice);

        // 3. SpMV
        int warps_per_block = 256 / 32;
        int grid_size = (N + warps_per_block - 1) / warps_per_block;
        spmv_csr_vector_kernel<<<grid_size, 256>>>(N, d_row_ptr, d_col_ind, d_vals, d_x, d_y);

        cudaDeviceSynchronize();
        auto end_cpu = std::chrono::high_resolution_clock::now();
        total_time_csr_nobuf += std::chrono::duration<double, std::milli>(end_cpu - start_cpu).count();
    }

    // =========================================================================
    // Experiment 3: CSR Hybrid (Static CSR + Dynamic COO Buffer)
    // =========================================================================
    // 초기 Static CSR 설정 (Exp 2에서 쓴 Map 원본 사용)
    rebuild_csr_on_host(N, sparse_map, h_row_ptr, h_col_ind, h_vals);
    int static_nnz = h_vals.size();
    
    // Static CSR을 GPU로 전송 (측정 시간 포함 X - 초기화 과정이므로)
    cudaMemcpy(d_row_ptr, h_row_ptr.data(), (N + 1) * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_col_ind, h_col_ind.data(), static_nnz * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_vals, h_vals.data(), static_nnz * sizeof(uint64_t), cudaMemcpyHostToDevice);

    // 버퍼 할당
    int *d_buf_rows, *d_buf_cols;
    uint64_t *d_buf_vals;
    cudaMalloc(&d_buf_rows, num_test_updates * sizeof(int));
    cudaMalloc(&d_buf_cols, num_test_updates * sizeof(int));
    cudaMalloc(&d_buf_vals, num_test_updates * sizeof(uint64_t));

    double total_time_hybrid = 0.0;
    int current_buffer_size = 0;

    for (const auto& u : updates) {
        auto start_cpu = std::chrono::high_resolution_clock::now();

        // 1. 버퍼에 업데이트 추가 (Append only 1 element)
        // 버퍼의 현재 위치(current_buffer_size)에 값을 복사
        cudaMemcpy(d_buf_rows + current_buffer_size, &u.r, sizeof(int), cudaMemcpyHostToDevice);
        cudaMemcpy(d_buf_cols + current_buffer_size, &u.c, sizeof(int), cudaMemcpyHostToDevice);
        cudaMemcpy(d_buf_vals + current_buffer_size, &u.new_val, sizeof(uint64_t), cudaMemcpyHostToDevice);
        
        current_buffer_size++;

        // 2. SpMV 실행 (Static CSR + Buffer COO)
        // 2-1. Static CSR
        int warps_per_block = 256 / 32;
        int grid_size_csr = (N + warps_per_block - 1) / warps_per_block;
        spmv_csr_vector_kernel<<<grid_size_csr, 256>>>(N, d_row_ptr, d_col_ind, d_vals, d_x, d_y);

        // 2-2. Buffer COO
        if (current_buffer_size > 0) {
            int coo_blocks = (current_buffer_size + 255) / 256;
            spmv_coo_atomic_kernel<<<coo_blocks, 256>>>(current_buffer_size, d_buf_rows, d_buf_cols, d_buf_vals, d_x, d_y);
        }

        cudaDeviceSynchronize();
        auto end_cpu = std::chrono::high_resolution_clock::now();
        total_time_hybrid += std::chrono::duration<double, std::milli>(end_cpu - start_cpu).count();
    }

    // =========================================================================
    // Report
    // =========================================================================
    printf("--------------------------------------------------\n");
    printf("[1] Dense Optimized: %.3f ms / update\n", total_time_dense / num_test_updates);
    printf("[2] CSR No-Buffer  : %.3f ms / update\n", total_time_csr_nobuf / num_test_updates);
    printf("[3] CSR Hybrid     : %.3f ms / update\n", total_time_hybrid / num_test_updates);
    printf("--------------------------------------------------\n");

    // Cleanup
    cudaFree(d_x); cudaFree(d_y);
    cudaFree(d_A_dense); cudaFree(d_upd_row); cudaFree(d_upd_col); cudaFree(d_upd_val);
    cudaFree(d_row_ptr); cudaFree(d_col_ind); cudaFree(d_vals);
    cudaFree(d_buf_rows); cudaFree(d_buf_cols); cudaFree(d_buf_vals);

    return 0;
}
