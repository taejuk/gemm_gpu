#include <stdio.h>
#include <cuda_runtime.h>

template <int STRIDE>
__global__ void shared_bank_conflict_kernel(float* out) {
    volatile __shared__ float s_data[8192];

    int tid = threadIdx.x;

    int idx = tid * STRIDE;

    s_data[idx] = idx;
    __syncthreads();

    for(int i = 0; i < 100000; i++) s_data[idx] = s_data[idx] + 0.01f;

    out[tid] = s_data[idx];
}

int main() {
    int threads = 256; 
    int blocks = 1024; 

    float *d_out;
    cudaMalloc(&d_out, blocks * threads * sizeof(float));

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    shared_bank_conflict_kernel<1><<<blocks, threads>>>(d_out);
    cudaDeviceSynchronize();

    cudaEventRecord(start);
    shared_bank_conflict_kernel<1><<<blocks, threads>>>(d_out);
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    float ms_stride1 = 0;
    cudaEventElapsedTime(&ms_stride1, start, stop);

    cudaEventRecord(start);
    shared_bank_conflict_kernel<32><<<blocks, threads>>>(d_out);
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    float ms_stride32 = 0;
    cudaEventElapsedTime(&ms_stride32, start, stop);

    printf("Shared Memory Access Time (100,000 iterations)\n");
    printf("==============================================\n");
    printf("STRIDE 1  (No Conflict)  : %.3f ms\n", ms_stride1);
    printf("STRIDE 32 (32-way Conflict): %.3f ms\n", ms_stride32);
    printf("==============================================\n");
    printf("Slowdown Factor: %.2f x\n", ms_stride32 / ms_stride1);

    cudaFree(d_out);
    return 0;
}
