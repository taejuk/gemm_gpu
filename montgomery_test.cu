#include <iostream>
#include <cuda_runtime.h>

#define P 0xFFFFFFFF00000001ULL

__device__ __forceinline__ uint64_t naive_mul(uint64_t a, uint64_t b) {
    unsigned __int128 n = (unsigned __int128)a * b;
    return (uint64_t)(n % P); // 하드웨어에서 가장 기피하는 나눗셈(%) 연산 발생
}

__device__ __forceinline__ uint64_t optimized_mul(uint64_t a, uint64_t b) {
    unsigned __int128 n = (unsigned __int128)a * b;
    uint64_t lo = (uint64_t)n;
    uint64_t hi = (uint64_t)(n >> 64);

    // % 연산 없이 오직 덧셈, 뺄셈, 비트 시프트(<<)만으로 모듈러 축소(Reduction)
    unsigned __int128 t = (unsigned __int128)lo + ((unsigned __int128)hi << 32) - hi;

    uint64_t r0 = (uint64_t)t;
    uint64_t r1 = (uint64_t)(t >> 64);
    uint64_t res = r0 + r1 * 0xFFFFFFFFULL;

    if (res >= P) res -= P;
    if (res >= P) res -= P; // Safety check
    return res;
}

__global__ void naive_kernel(uint64_t* data, int num_elements, int iters) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < num_elements) {
        uint64_t val = data[idx];
        // 1000번의 곱셈 루프
        for (int i = 0; i < iters; i++) {
            val = naive_mul(val, 0x123456789ABCDEFULL); 
        }
        data[idx] = val;
    }
}

__global__ void optimized_kernel(uint64_t* data, int num_elements, int iters) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < num_elements) {
        uint64_t val = data[idx];
        // 1000번의 곱셈 루프
        for (int i = 0; i < iters; i++) {
            val = optimized_mul(val, 0x123456789ABCDEFULL);
        }
        data[idx] = val;
    }
}

int main() {
    int num_elements = 1024 * 1024 * 10; // 천만 개의 데이터
    int iters = 1000; // 스레드 1개당 1000번 연속 곱셈

    size_t bytes = num_elements * sizeof(uint64_t);
    uint64_t *d_data_naive, *d_data_opt;

    cudaMalloc(&d_data_naive, bytes);
    cudaMalloc(&d_data_opt, bytes);

    int threads = 256;
    int blocks = (num_elements + threads - 1) / threads;

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    cudaEventRecord(start);
    naive_kernel<<<blocks, threads>>>(d_data_naive, num_elements, iters);
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    float naive_ms = 0;
    cudaEventElapsedTime(&naive_ms, start, stop);

    cudaEventRecord(start);
    optimized_kernel<<<blocks, threads>>>(d_data_opt, num_elements, iters);
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    float opt_ms = 0;
    cudaEventElapsedTime(&opt_ms, start, stop);

    std::cout << "=== ZKML Prime Field Multiplication Benchmark ===" << std::endl;
    std::cout << "Total Elements : " << num_elements << std::endl;
    std::cout << "Iters / Thread : " << iters << std::endl;
    std::cout << "-----------------------------------------------" << std::endl;
    std::cout << "1. Naive (using % op)   : " << naive_ms << " ms\n";
    std::cout << "2. Optimized (Bitwise)  : " << opt_ms << " ms\n";
    std::cout << "-----------------------------------------------" << std::endl;
    std::cout << "Speedup                 : " << naive_ms / opt_ms << " x\n";

    cudaFree(d_data_naive);
    cudaFree(d_data_opt);
    return 0;
}
