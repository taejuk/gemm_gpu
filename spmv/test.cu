#include <iostream>
#include <vector>
#include <random>
#include <algorithm>
#include <map>
#include <cuda_runtime.h>
#include <device_launch_parameters.h>

// =================================================================================
// 1. Constants & Goldilocks Field Arithmetic (Device)
// =================================================================================

#define GOLDILOCKS_P 0xFFFFFFFF00000001ULL

// [Optimized Multiplication] 비트 연산 최적화 (나눗셈 사용 X)
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

// [Optimized Addition] a + b mod P
__device__ __forceinline__ uint64_t goldilocks_add(uint64_t a, uint64_t b) {
    unsigned __int128 res = (unsigned __int128)a + b;
    if (res >= GOLDILOCKS_P) res -= GOLDILOCKS_P;
    return (uint64_t)res;
}

// [Atomic Add for Modular Arithmetic]
// 여러 스레드가 동시에 같은 y[row]에 값을 더할 때 Race Condition 방지
__device__ void atomicAddGoldilocks(uint64_t* address, uint64_t val) {
    unsigned long long* address_as_ull = (unsigned long long*)address;
    unsigned long long old = *address_as_ull, assumed;
    do {
        assumed = old;
        uint64_t sum = goldilocks_add((uint64_t)assumed, val);
        old = atomicCAS(address_as_ull, assumed, (unsigned long long)sum);
    } while (assumed != old);
}

// =================================================================================
// 2. Kernels (Dense, CSR, COO)
// =================================================================================

// [Dense Kernel] Naive Row-major
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

// [CSR Kernel] Static Base Matrix 처리
__global__ void spmv_csr_kernel(int num_rows, const int* row_ptr, const int* col_ind, 
                                const uint64_t* vals, const uint64_t* x, uint64_t* y) {
    int row = blockDim.x * blockIdx.x + threadIdx.x;
    if (row < num_rows) {
        int start = row_ptr[row];
        int end = row_ptr[row + 1];
        uint64_t sum = 0;
        
        for (int i = start; i < end; i++) {
            sum = goldilocks_add(sum, goldilocks_mul(vals[i], x[col_ind[i]]));
        }
        y[row] = sum; // 초기화 (덮어쓰기)
    }
}

// [COO Kernel] Dynamic Updates (Buffer) 처리
// AtomicAdd를 사용하여 기존 CSR 결과 위에 "덧셈" 수행
__global__ void spmv_coo_atomic_kernel(int num_updates, const int* rows, const int* cols, 
                                       const uint64_t* vals, const uint64_t* x, uint64_t* y) {
    int idx = blockDim.x * blockIdx.x + threadIdx.x;
    if (idx < num_updates) {
        int r = rows[idx];
        int c = cols[idx];
        uint64_t v = vals[idx];

        uint64_t product = goldilocks_mul(v, x[c]);
        
        // y[r] += product (Mod P)
        atomicAddGoldilocks(&y[r], product);
    }
}

// =================================================================================
// 3. Helper Functions (Data Generation)
// =================================================================================

struct UpdateInfo {
    int r, c;
    uint64_t diff_val; // 기존 값과 새로운 값의 차이 (혹은 새로운 값)
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

// =================================================================================
// 4. Main Experiment
// =================================================================================

int main() {
    // --- 1. Experimental Setup ---
    const int N = 4096;
    const float density = 0.80f; // 1% Sparsity
    const int num_updates = 1; // 10,000개의 값이 동적으로 변경됨
    
    printf("=== Dynamic SpMV Experiment (Goldilocks Field) ===\n");
    printf("Matrix Size: %d x %d\n", N, N);
    printf("Initial Density: %.2f%%\n", density * 100);
    printf("Dynamic Updates: %d elements change\n\n", num_updates);

    // --- 2. Initial Matrix Generation (CPU) ---
    // Use Map for easy sorted CSR construction
    std::map<std::pair<int, int>, uint64_t> sparse_map;
    std::vector<uint64_t> h_A_dense(N * N, 0);

    int nnz_target = (int)(N * N * density);
    std::mt19937 gen(1234);
    std::uniform_int_distribution<> dist(0, N - 1);

    while (sparse_map.size() < nnz_target) {
        int r = dist(gen);
        int c = dist(gen);
        sparse_map[{r, c}] = 1; // Dummy value 1
        h_A_dense[r * N + c] = 1;
    }

    // Convert Map to CSR (CPU) - Initial Static Build
    std::vector<int> h_csr_row_ptr(N + 1, 0);
    std::vector<int> h_csr_col_ind;
    std::vector<uint64_t> h_csr_vals;
    h_csr_col_ind.reserve(nnz_target);
    h_csr_vals.reserve(nnz_target);

    for (auto const& [key, val] : sparse_map) {
        h_csr_row_ptr[key.first + 1]++;
        h_csr_col_ind.push_back(key.second);
        h_csr_vals.push_back(val);
    }
    for (int i = 0; i < N; i++) h_csr_row_ptr[i+1] += h_csr_row_ptr[i];

    // Input/Output Vectors
    std::vector<uint64_t> h_x(N, 1);
    
    // --- 3. GPU Memory Allocation (Static Parts) ---
    uint64_t *d_x, *d_y_dense, *d_y_sparse;
    cudaMalloc(&d_x, N * sizeof(uint64_t));
    cudaMalloc(&d_y_dense, N * sizeof(uint64_t));
    cudaMalloc(&d_y_sparse, N * sizeof(uint64_t));
    cudaMemcpy(d_x, h_x.data(), N * sizeof(uint64_t), cudaMemcpyHostToDevice);

    // Dense Matrix (Initial)
    uint64_t* d_A_dense;
    cudaMalloc(&d_A_dense, N * N * sizeof(uint64_t));
    cudaMemcpy(d_A_dense, h_A_dense.data(), N * N * sizeof(uint64_t), cudaMemcpyHostToDevice);

    // Sparse CSR (Initial)
    int *d_csr_row_ptr, *d_csr_col_ind;
    uint64_t *d_csr_vals;
    cudaMalloc(&d_csr_row_ptr, (N + 1) * sizeof(int));
    cudaMalloc(&d_csr_col_ind, nnz_target * sizeof(int));
    cudaMalloc(&d_csr_vals, nnz_target * sizeof(uint64_t));
    
    cudaMemcpy(d_csr_row_ptr, h_csr_row_ptr.data(), (N + 1) * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_csr_col_ind, h_csr_col_ind.data(), nnz_target * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_csr_vals, h_csr_vals.data(), nnz_target * sizeof(uint64_t), cudaMemcpyHostToDevice);

    // --- 4. Dynamic Update Scenario ---
    // Generate updates (Diffs)
    std::vector<UpdateInfo> updates;
    generate_updates(N, num_updates, updates);
    
    // Preparation for COO Buffer (Sparse Strategy)
    std::vector<int> h_update_rows, h_update_cols;
    std::vector<uint64_t> h_update_vals;
    for(const auto& u : updates) {
        h_update_rows.push_back(u.r);
        h_update_cols.push_back(u.c);
        h_update_vals.push_back(u.diff_val);
        
        // Also apply to CPU Dense for correctness (In real scenario, this happens in logic)
        // Note: For simplicity, we assume these are 'additive' updates or we just set them.
        // Let's assume we overwrite for Dense:
        h_A_dense[u.r * N + u.c] = u.diff_val; 
    }

    cudaEvent_t start, stop;
    cudaEventCreate(&start); cudaEventCreate(&stop);

    // =========================================================================
    // Strategy A: Dense Matrix (Full Re-upload + Kernel)
    // =========================================================================
    //cudaEventRecord(start);

    // 1. Data Transfer (Full N*N Matrix copy due to updates) - The Bottleneck!
    cudaMemcpy(d_A_dense, h_A_dense.data(), N * N * sizeof(uint64_t), cudaMemcpyHostToDevice);
    cudaEventRecord(start);
    // 2. Dense Kernel Execution
    int blockSize = 256;
    int numBlocks = (N + blockSize - 1) / blockSize;
    dense_mv_kernel<<<numBlocks, blockSize>>>(N, d_A_dense, d_x, d_y_dense);

    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    float time_dense = 0;
    cudaEventElapsedTime(&time_dense, start, stop);


    // =========================================================================
    // Strategy B: Hybrid Sparse (Update Buffer Upload + Dual Kernels)
    // =========================================================================
    // Buffer Memory Allocation (Done "On demand" or pre-allocated)
    int *d_upd_rows, *d_upd_cols;
    uint64_t *d_upd_vals;
    cudaMalloc(&d_upd_rows, num_updates * sizeof(int));
    cudaMalloc(&d_upd_cols, num_updates * sizeof(int));
    cudaMalloc(&d_upd_vals, num_updates * sizeof(uint64_t));

    cudaEventRecord(start);

    // 1. Data Transfer (Only Updates! Small COO copy)
    cudaMemcpy(d_upd_rows, h_update_rows.data(), num_updates * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_upd_cols, h_update_cols.data(), num_updates * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_upd_vals, h_update_vals.data(), num_updates * sizeof(uint64_t), cudaMemcpyHostToDevice);

    // 2. Execution Part 1: Static CSR Kernel
    spmv_csr_kernel<<<numBlocks, blockSize>>>(N, d_csr_row_ptr, d_csr_col_ind, d_csr_vals, d_x, d_y_sparse);

    // 3. Execution Part 2: Dynamic COO Buffer Kernel (Atomic Add)
    int coo_blocks = (num_updates + blockSize - 1) / blockSize;
    spmv_coo_atomic_kernel<<<coo_blocks, blockSize>>>(num_updates, d_upd_rows, d_upd_cols, d_upd_vals, d_x, d_y_sparse);

    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    float time_sparse = 0;
    cudaEventElapsedTime(&time_sparse, start, stop);

    // --- 5. Report Results ---
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
    cudaFree(d_upd_rows); cudaFree(d_upd_cols); cudaFree(d_upd_vals);
    cudaEventDestroy(start); cudaEventDestroy(stop);

    return 0;
}
