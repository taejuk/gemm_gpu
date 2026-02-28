#include <stdio.h>
#include <stdint.h>
#include <cuda_runtime.h>

// 설정: L1(128KB)을 초과하고 L2(V100/Ampere 기준 6MB)에 들어갈 수 있는 크기
// 4MB 배열 (Double은 8바이트이므로 524,288개)
#define L2_SIZE_BYTES (4 * 1024 * 1024) 
#define L2_ELEMS (L2_SIZE_BYTES / sizeof(double))
#define THREADS_PER_BLOCK 1024
#define REPEAT_COUNT 50 // 배열이 커졌으므로 반복 횟수를 약간 줄임

// 논문의 Listing 3.2를 기반으로 한 L2 Bandwidth 측정 커널
__global__ void l2_bw_kernel(uint32_t *startClk, uint32_t *stopClk, double *dsink, double *posArray) {
    uint32_t tid = threadIdx.x;
    
    // 레지스터 최적화 방지를 위한 sink 변수
    double sink = 0;
    double *ptr_base = posArray;

    // -------------------------------------------------------------------------
    // 1. Warm-up Phase
    // 데이터를 L2 캐시로 로드합니다.
    // ld.global.cg : Cache Global (L1 우회, L2 캐싱 강제)
    // -------------------------------------------------------------------------
    for (uint32_t i = tid; i < L2_ELEMS; i += THREADS_PER_BLOCK) {
        double *ptr = ptr_base + i;
        asm volatile (
            "{\n\t"
            ".reg .f64 data;\n\t"
            "ld.global.cg.f64 data, [%1];\n\t" // L1 우회, L2에서 로드
            "add.f64 %0, data, %0;\n\t"
            "}"
            : "+d"(sink) : "l"(ptr) : "memory"
        );
    }

    // 모든 스레드가 워밍업을 마칠 때까지 대기
    asm volatile ("bar.sync 0;");

    // -------------------------------------------------------------------------
    // 2. Measurement Phase
    // -------------------------------------------------------------------------
    uint32_t start = 0;
    asm volatile ("mov.u32 %0, %%clock;" : "=r"(start) :: "memory");

    #pragma unroll
    for (int r = 0; r < REPEAT_COUNT; r++) {
        for (uint32_t i = 0; i < L2_ELEMS; i += THREADS_PER_BLOCK) {
            uint32_t offset = (tid + i) % L2_ELEMS;
            double *ptr = ptr_base + offset;

            asm volatile (
                "{\n\t"
                ".reg .f64 data;\n\t"
                "ld.global.cg.f64 data, [%1];\n\t" // Critical: L2 Load
                "add.f64 %0, data, %0;\n\t"
                "}"
                : "+d"(sink) : "l"(ptr) : "memory"
            );
        }
    }

    asm volatile ("bar.sync 0;");

    uint32_t stop = 0;
    asm volatile ("mov.u32 %0, %%clock;" : "=r"(stop) :: "memory");

    // -------------------------------------------------------------------------
    // 3. Result Write-back
    // -------------------------------------------------------------------------
    startClk[tid] = start;
    stopClk[tid] = stop;
    dsink[tid] = sink;
}

int main() {
    uint32_t *d_startClk, *d_stopClk;
    double *d_dsink, *d_posArray;
    
    cudaMalloc(&d_startClk, THREADS_PER_BLOCK * sizeof(uint32_t));
    cudaMalloc(&d_stopClk, THREADS_PER_BLOCK * sizeof(uint32_t));
    cudaMalloc(&d_dsink, THREADS_PER_BLOCK * sizeof(double));
    cudaMalloc(&d_posArray, L2_SIZE_BYTES);

    cudaMemset(d_posArray, 0, L2_SIZE_BYTES);

    printf("=== L2 Data Cache Bandwidth Benchmark ===\n");
    printf("Config: %d Threads, %d MB Array, %d Repeats\n", THREADS_PER_BLOCK, L2_SIZE_BYTES/(1024*1024), REPEAT_COUNT);

    // L1 벤치마크와 동일하게 1개 Block(SM 1개)의 단일 성능 측정
    l2_bw_kernel<<<1, THREADS_PER_BLOCK>>>(d_startClk, d_stopClk, d_dsink, d_posArray);
    cudaDeviceSynchronize();

    uint32_t *h_startClk = (uint32_t*)malloc(THREADS_PER_BLOCK * sizeof(uint32_t));
    uint32_t *h_stopClk = (uint32_t*)malloc(THREADS_PER_BLOCK * sizeof(uint32_t));
    cudaMemcpy(h_startClk, d_startClk, THREADS_PER_BLOCK * sizeof(uint32_t), cudaMemcpyDeviceToHost);
    cudaMemcpy(h_stopClk, d_stopClk, THREADS_PER_BLOCK * sizeof(uint32_t), cudaMemcpyDeviceToHost);

    double total_cycles = 0;
    for (int i = 0; i < THREADS_PER_BLOCK; i++) {
        if (h_stopClk[i] >= h_startClk[i])
            total_cycles += (h_stopClk[i] - h_startClk[i]);
        else
            total_cycles += (h_stopClk[i] + (UINT32_MAX - h_startClk[i]));
    }
    double avg_cycles = total_cycles / THREADS_PER_BLOCK;

    uint64_t total_bytes_loaded = (uint64_t)REPEAT_COUNT * L2_ELEMS * sizeof(double);
    double bytes_per_cycle = total_bytes_loaded / avg_cycles;

    printf("--------------------------------------------------\n");
    printf("Total Bytes Loaded: %lu Bytes\n", total_bytes_loaded);
    printf("Average Cycles    : %.2f cycles\n", avg_cycles);
    printf("Measured L2 BW    : %.2f Bytes/Cycle (per SM)\n", bytes_per_cycle);
    printf("--------------------------------------------------\n");
    
    cudaFree(d_startClk); cudaFree(d_stopClk); cudaFree(d_dsink); cudaFree(d_posArray);
    free(h_startClk); free(h_stopClk);

    return 0;
}
