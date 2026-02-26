#include <iostream>
#include <vector>
#include <random>
#include <algorithm>
#include <map>
#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include "dense/dense.cuh"
#include "spmv/csr.cuh"
#include "spmv/coo.cuh"

struct UpdateInfo {
    int r, c;
    uint64_t diff_val;
};

void generate_updates(int N, int num_updates, std::vector<UpdateInfo>& updates) {
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_int_distribution<> dist_idx(0, N - 1);
    std::uniform_int_distribution<uint64_t> dist_val(1, 1000); // 임의의 값

    for(int i=0; i<num_updates; i++) {
        updates.push_back({dist_idx(gen), dist_idx(gen), dist_val(gen)});
    }
}

int main(int argc, char* argv[]) {
    if(argc != 2) {
    	fprintf(stderr, "enter the density!\n");
	return 1;
    }
    const int N = 4096;
    const float density = std::stof(argv[1]);
    const int num_updates = 10000;

    printf("=== Dynamic SpMV Experiment (Goldilocks Field) ===\n");
    printf("Matrix Size: %d x %d\n", N, N);
    printf("Initial Density: %.2f%%\n", density * 100);
    printf("Dynamic Updates: %d elements change\n\n", num_updates);

    std::map<std::pair<int,int>, uint64_t> sparse_map;
    std::vector<uint64_t> h_A_dense(N*N, 0);
    // nnz: number of non zero
    int nnz_target = (int)(N * N * density);
    std::mt19937 gen(1234);
    std::uniform_int_distribution<> dist(0, N - 1);
    // 중복되면 map size는 증가하지 않는다.
    while (sparse_map.size() < nnz_target) {
        int r = dist(gen);
        int c = dist(gen);
        sparse_map[{r, c}] = 1; // Dummy value 1
        h_A_dense[r * N + c] = 1;
    }

    std::vector<int> h_csr_row_ptr(N+1, 0);
    std::vector<int> h_csr_col_ind;
    std::vector<uint64_t> h_csr_vals;
    h_csr_col_ind.reserve(nnz_target);
    h_csr_vals.reserve(nnz_target);

    for (auto const& [key, value]: sparse_map) {
        h_csr_row_ptr[key.first + 1]++;
        h_csr_col_ind.push_back(key.second);
        h_csr_vals.push_back(value);
    }
    for (int i = 0; i < N; i++) h_csr_row_ptr[i+1] += h_csr_row_ptr[i];

    std::vector<uint64_t> h_x(N, 1);

    uint64_t *d_x, *d_y_dense, *d_y_sparse;
    cudaMalloc(&d_x, N * sizeof(uint64_t));
    cudaMalloc(&d_y_dense, N * sizeof(uint64_t));
    cudaMalloc(&d_y_sparse, N * sizeof(uint64_t));
    cudaMemcpy(d_x, h_x.data(), N * sizeof(uint64_t), cudaMemcpyHostToDevice);
    
    uint64_t* d_A_dense;
    cudaMalloc(&d_A_dense, N * N * sizeof(uint64_t));
    cudaMemcpy(d_A_dense, h_A_dense.data(), N * N * sizeof(uint64_t), cudaMemcpyHostToDevice);

    int *d_csr_row_ptr, *d_csr_col_ind;
    uint64_t *d_csr_vals;
    cudaMalloc(&d_csr_row_ptr, (N + 1) * sizeof(int));
    cudaMalloc(&d_csr_col_ind, nnz_target * sizeof(int));
    cudaMalloc(&d_csr_vals, nnz_target * sizeof(uint64_t));

    /*
    cudaMemcpy(d_csr_row_ptr, h_csr_row_ptr.data(), (N + 1) * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_csr_col_ind, h_csr_col_ind.data(), nnz_target * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_csr_vals, h_csr_vals.data(), nnz_target * sizeof(uint64_t), cudaMemcpyHostToDevice);
    */
    std::vector<UpdateInfo> updates;
    //generate_updates(N, num_updates, updates);

    // update하는 로직 추가 -> 하나씩 바뀔때로 구해보자

    cudaEvent_t start, stop;
    cudaEventCreate(&start); cudaEventCreate(&stop);

    cudaEventRecord(start);
    cudaMemcpy(d_A_dense, h_A_dense.data(), N * N * sizeof(uint64_t), cudaMemcpyHostToDevice);
    int blockSize = 256;
    int numBlocks = (N + blockSize - 1) / blockSize;
    dense_mv_kernel<<<numBlocks, blockSize>>>(N, d_A_dense, d_x, d_y_dense);

    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    float time_dense = 0;
    cudaEventElapsedTime(&time_dense, start, stop);

    // update할 때 값 추가
    cudaEventRecord(start);
    cudaMemcpy(d_csr_row_ptr, h_csr_row_ptr.data(), (N + 1) * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_csr_col_ind, h_csr_col_ind.data(), nnz_target * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_csr_vals, h_csr_vals.data(), nnz_target * sizeof(uint64_t), cudaMemcpyHostToDevice);

    int num_warps = N;
    int warps_per_block = blockSize / WARP_SIZE;
    int grid_size = (num_warps + warps_per_block - 1) / warps_per_block;

    spmv_csr_vector_kernel<<<grid_size, blockSize>>>(N, d_csr_row_ptr, d_csr_col_ind, d_csr_vals, d_x, d_y_sparse);
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    float time_sparse = 0;
    cudaEventElapsedTime(&time_sparse, start, stop);

    printf("--------------------------------------------------\n");
    printf("[Dense Strategy] Time: %.3f ms\n", time_dense);
    printf("  - Overhead: Full N^2 copy (%lu bytes)\n", N * N * sizeof(uint64_t));
    printf("--------------------------------------------------\n");
    printf("[Sparse Hybrid]  Time: %.3f ms\n", time_sparse);
    printf("  - Overhead: Only Updates copy (%lu bytes)\n", num_updates * (2 * sizeof(int) + sizeof(uint64_t)));
    printf("--------------------------------------------------\n");

    if (time_sparse < time_dense) {
        printf("Winner: Sparse Hybrid (%.2fx faster)\n", time_dense / time_sparse);
    } else {
        printf("Winner: Dense (Sparse overhead was too high)\n");
    }

    // Cleanup
    cudaFree(d_x); cudaFree(d_y_dense); cudaFree(d_y_sparse);
    cudaFree(d_A_dense);
    cudaFree(d_csr_row_ptr); cudaFree(d_csr_col_ind); cudaFree(d_csr_vals);
    //cudaFree(d_upd_rows); cudaFree(d_upd_cols); cudaFree(d_upd_vals);
    cudaEventDestroy(start); cudaEventDestroy(stop);

    return 0;
}
