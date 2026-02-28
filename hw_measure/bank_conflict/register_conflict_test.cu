#include <stdio.h>
#include <cuda_runtime.h>

// ---------------------------------------------------------
// 1. SASS 패칭 베이스 커널 A (No Conflict 목표)
// ---------------------------------------------------------
__global__ void no_conflict_kernel(float *out) {
    // 딱 3개의 스칼라 변수만 사용
    float a = threadIdx.x * 0.001f; // 상수 폴딩 방지
    float b = threadIdx.x * 0.002f;
    float c = 0;
    
    float a1 = threadIdx.x * 0.003f; // 상수 폴딩 방지
    float b1 = threadIdx.x * 0.004f;
    float c1 = 0;

    float a2 = threadIdx.x * 0.005f; // 상수 폴딩 방지
    float b2 = threadIdx.x * 0.006f;
    float c2 = 0;

    float a3 = threadIdx.x * 0.007f; // 상수 폴딩 방지
    float b3 = threadIdx.x * 0.008f;
    float c3 = 0;
    // 루프 삭제 방지
    #pragma unroll 1
    for (int i = 0; i < 1000000; i++) {
        // 패칭 타겟: 이 한 줄의 FFMA 명령어를 SASS에서 수정할 예정
        c = fmaf(a, b, c); 
	c1 = fmaf(a1, b1, c1);
	c2 = fmaf(a2, b2, c2);
        c3 = fmaf(a3, b3, c3);
    }
    out[blockIdx.x * blockDim.x + threadIdx.x] = c + a + b +  c1 + a1 + b1 +  c2 + a2 + b2 +  c3 + a3 + b3 ;
}

// ---------------------------------------------------------
// 2. SASS 패칭 베이스 커널 B (Max Conflict 목표)
// ---------------------------------------------------------
__global__ void conflict_kernel(float *out) {
    float a = threadIdx.x * 0.001f; // 상수 폴딩 방지
    float b = threadIdx.x * 0.002f;
    float c = 0;

    float a1 = threadIdx.x * 0.003f; // 상수 폴딩 방지
    float b1 = threadIdx.x * 0.004f;
    float c1 = 0;

    float a2 = threadIdx.x * 0.005f; // 상수 폴딩 방지
    float b2 = threadIdx.x * 0.006f;
    float c2 = 0;

    float a3 = threadIdx.x * 0.007f; // 상수 폴딩 방지
    float b3 = threadIdx.x * 0.008f;
    float c3 = 0;
    // 루프 삭제 방지
    #pragma unroll 1
    for (int i = 0; i < 1000000; i++) {
        // 패칭 타겟: 이 한 줄의 FFMA 명령어를 SASS에서 수정할 예정
        c = fmaf(a, b, c);
        c1 = fmaf(a1, b1, c1);
        c2 = fmaf(a2, b2, c2);
        c3 = fmaf(a3, b3, c3);
    }
    out[blockIdx.x * blockDim.x + threadIdx.x] = c + a + b +  c1 + a1 + b1 +  c2 + a2 + b2 +  c3 + a3 + b3;
}

int main() {
    int threads = 256;
    int blocks = 80;
    
    float *d_out;
    cudaMalloc(&d_out, blocks * threads * sizeof(float));

    cudaEvent_t start, stop;
    cudaEventCreate(&start);  cudaEventCreate(&stop);

    // Warm-up
    no_conflict_kernel<<<blocks, threads>>>(d_out);
    cudaDeviceSynchronize();

    // [측정 1] 패칭 후 No Conflict 커널이 될 부분
    cudaEventRecord(start);
    no_conflict_kernel<<<blocks, threads>>>(d_out);
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    float ms_normal = 0; cudaEventElapsedTime(&ms_normal, start, stop);

    // [측정 2] 패칭 후 Max Conflict 커널이 될 부분
    cudaEventRecord(start);
    conflict_kernel<<<blocks, threads>>>(d_out);
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    float ms_conflict = 0; cudaEventElapsedTime(&ms_conflict, start, stop);

    printf("Minimal 3-Var Register Bank Conflict Test (1,000,000 iterations)\n");
    printf("===================================================================\n");
    printf("No Conflict Kernel Time: %.3f ms\n", ms_normal);
    printf("Conflict Kernel Time   : %.3f ms\n", ms_conflict);
    
    if(ms_conflict > ms_normal) {
        printf("Slowdown Factor: %.2f x\n", ms_conflict / ms_normal);
    }

    cudaFree(d_out);
    return 0;
}
