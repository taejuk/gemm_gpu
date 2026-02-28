#include <stdio.h>
#include <stdint.h>
#include <cuda_runtime.h>

// 설정: Volta V100 기준
// L1 Cache 크기 (128KB)에 맞춰서 배열 크기 설정 (Double은 8바이트이므로 16384개)
#define L1_SIZE_BYTES (128 * 1024) 
#define L1_ELEMS (L1_SIZE_BYTES / sizeof(double))
#define THREADS_PER_BLOCK 1024
#define WARP_SIZE 32
#define REPEAT_COUNT 100 // 측정 정확도를 위해 반복

// 논문의 Listing 3.1을 기반으로 한 L1 Bandwidth 측정 커널
__global__ void l1_bw_kernel(uint32_t *startClk, uint32_t *stopClk, double *dsink, double *posArray) {
    // Thread index
    uint32_t tid = threadIdx.x;

    // 레지스터 최적화 방지를 위한 sink 변수
    double sink = 0;
    double *ptr_base = posArray;

    // -------------------------------------------------------------------------
    // 1. Warm-up Phase
    // 데이터를 L1 캐시로 확실하게 로드하기 위해 미리 한 번 읽습니다.
    // ld.global.ca : Cache at All levels (L1 + L2)
    // -------------------------------------------------------------------------
    for (uint32_t i = tid; i < L1_ELEMS; i += THREADS_PER_BLOCK) {
        double *ptr = ptr_base + i;
        asm volatile (
            "{\n\t"
            ".reg .f64 data;\n\t"
            "ld.global.ca.f64 data, [%1];\n\t" // Load from L1
            "add.f64 %0, data, %0;\n\t"        // Accumulate to sink
            "}"
            : "+d"(sink) : "l"(ptr) : "memory"
        );
    }

    // 모든 스레드가 워밍업을 마칠 때까지 대기
    asm volatile ("bar.sync 0;");

    // -------------------------------------------------------------------------
    // 2. Measurement Phase
    // 실제 대역폭 측정 시작
    // -------------------------------------------------------------------------
    
    // 타이머 시작 (clock 레지스터 읽기)
    uint32_t start = 0;
    asm volatile ("mov.u32 %0, %%clock;" : "=r"(start) :: "memory");

    // Main Loop: L1 Cache Stress Test
    // 논문 방식: 쓰레드들이 L1 캐시의 모든 데이터를 반복적으로 긁어옴
    #pragma unroll
    for (int r = 0; r < REPEAT_COUNT; r++) {
        // 모든 스레드가 전체 배열을 훑음 (부하 극대화)
        for (uint32_t i = 0; i < L1_ELEMS; i += THREADS_PER_BLOCK) {
            // 오프셋 계산 (Bank Conflict 최소화 및 패턴 다양화)
            uint32_t offset = (tid + i) % L1_ELEMS;
            double *ptr = ptr_base + offset;

            asm volatile (
                "{\n\t"
                ".reg .f64 data;\n\t"
                "ld.global.ca.f64 data, [%1];\n\t" // Critical: L1 Load
                "add.f64 %0, data, %0;\n\t"        // Dependency chain to force load
                "}"
                : "+d"(sink) : "l"(ptr) : "memory"
            );
        }
    }

    // 동기화 (모든 스레드가 끝날 때까지)
    asm volatile ("bar.sync 0;");

    // 타이머 종료
    uint32_t stop = 0;
    asm volatile ("mov.u32 %0, %%clock;" : "=r"(stop) :: "memory");

    // -------------------------------------------------------------------------
    // 3. Result Write-back
    // 결과를 메모리에 써야 컴파일러가 위 코드를 삭제하지 않음
    // -------------------------------------------------------------------------
    startClk[tid] = start;
    stopClk[tid] = stop;
    dsink[tid] = sink;
}

int main() {
    // 메모리 할당
    uint32_t *d_startClk, *d_stopClk;
    double *d_dsink, *d_posArray;
    
    cudaMalloc(&d_startClk, THREADS_PER_BLOCK * sizeof(uint32_t));
    cudaMalloc(&d_stopClk, THREADS_PER_BLOCK * sizeof(uint32_t));
    cudaMalloc(&d_dsink, THREADS_PER_BLOCK * sizeof(double));
    cudaMalloc(&d_posArray, L1_SIZE_BYTES);

    // 데이터 초기화 (내용은 중요하지 않음)
    cudaMemset(d_posArray, 0, L1_SIZE_BYTES);

    printf("=== L1 Data Cache Bandwidth Benchmark ===\n");
    printf("Config: %d Threads, %d KB Array, %d Repeats\n", THREADS_PER_BLOCK, L1_SIZE_BYTES/1024, REPEAT_COUNT);

    // 커널 실행 (1 Block, 1024 Threads) -> SM 1개의 Peak 성능 측정
    l1_bw_kernel<<<1, THREADS_PER_BLOCK>>>(d_startClk, d_stopClk, d_dsink, d_posArray);
    cudaDeviceSynchronize();

    // 결과 가져오기
    uint32_t *h_startClk = (uint32_t*)malloc(THREADS_PER_BLOCK * sizeof(uint32_t));
    uint32_t *h_stopClk = (uint32_t*)malloc(THREADS_PER_BLOCK * sizeof(uint32_t));
    cudaMemcpy(h_startClk, d_startClk, THREADS_PER_BLOCK * sizeof(uint32_t), cudaMemcpyDeviceToHost);
    cudaMemcpy(h_stopClk, d_stopClk, THREADS_PER_BLOCK * sizeof(uint32_t), cudaMemcpyDeviceToHost);

    // 평균 시간 계산
    double total_cycles = 0;
    for (int i = 0; i < THREADS_PER_BLOCK; i++) {
        // 오버플로우 처리 포함
        if (h_stopClk[i] >= h_startClk[i])
            total_cycles += (h_stopClk[i] - h_startClk[i]);
        else
            total_cycles += (h_stopClk[i] + (UINT32_MAX - h_startClk[i]));
    }
    double avg_cycles = total_cycles / THREADS_PER_BLOCK;

    // 대역폭 계산
    // 총 전송량 = (스레드 수) * (반복 횟수) * (배열 크기 / 스레드 수 만큼 루프) * (데이터 크기 8B)
    // 코드 상: REPEAT_COUNT * L1_ELEMS * 8 Bytes * THREADS_PER_BLOCK (X)
    // 코드 논리:
    // 각 스레드는 REPEAT_COUNT * (L1_ELEMS / THREADS_PER_BLOCK) 만큼 Load를 수행
    // 총 Load 횟수 (전체 스레드 합) = REPEAT_COUNT * L1_ELEMS
    uint64_t total_bytes_loaded = (uint64_t)REPEAT_COUNT * L1_ELEMS * sizeof(double);
    
    // Bytes per Cycle
    double bytes_per_cycle = total_bytes_loaded / avg_cycles;

    printf("--------------------------------------------------\n");
    printf("Total Bytes Loaded: %lu Bytes\n", total_bytes_loaded);
    printf("Average Cycles    : %.2f cycles\n", avg_cycles);
    printf("Measured L1 BW    : %.2f Bytes/Cycle (per SM)\n", bytes_per_cycle);
    printf("--------------------------------------------------\n");
    
    // 정리
    cudaFree(d_startClk); cudaFree(d_stopClk); cudaFree(d_dsink); cudaFree(d_posArray);
    free(h_startClk); free(h_stopClk);

    return 0;
}
