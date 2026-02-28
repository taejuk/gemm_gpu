#include <stdio.h>
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <mma.h>

using namespace nvcuda;

const int M = 256;
const int N = 256;
const int K = 256;

const int WMMA_M = 16;
const int WMMA_N = 16;
const int WMMA_K = 16;

__global__ void wmma_gemm_kernel(half* A, half* B, float* C) {
    int warpRow = blockIdx.y;
    int warpCol = blockIdx.x;

    wmma::fragment<wmma::accumulator, WMMA_M, WMMA_N, WMMA_K, float> frag_c;
    wmma::fill_fragment(frag_c, 0.0f);

    for (int k_step = 0; k_step < K; k_step += WMMA_K) {
        wmma::fragment<wmma::matrix_a, WMMA_M, WMMA_N, WMMA_K, half, wmma::row_major> frag_a;
	wmma::fragment<wmma::matrix_b, WMMA_M, WMMA_N, WMMA_K, half, wmma::row_major> frag_b;

	const half *a_tile_ptr = A + (warpRow * WMMA_M) * K + k_step;
	const half *b_tile_ptr = B + k_step * N + (warpCol * WMMA_N);

	wmma::load_matrix_sync(frag_a, a_tile_ptr, K);
	wmma::load_matrix_sync(frag_b, b_tile_ptr, N);
	
	wmma::mma_sync(frag_c, frag_a, frag_b, frag_c);
    }

    float *c_tile_ptr = C + (warpRow * WMMA_M) * N + (warpCol * WMMA_N);
    wmma::store_matrix_sync(c_tile_ptr, frag_c, N, wmma::mem_row_major);
}

int main() {
    int size_A = M * K * sizeof(half);
    int size_B = K * N * sizeof(half);
    int size_C = M * N * sizeof(float);

    half *h_a = (half*)malloc(size_A);
    half *h_b = (half*)malloc(size_B);
    float *h_c = (float*)malloc(size_C);

    for (int i = 0; i < M * K; i++) h_a[i] = __float2half(1.0f);
    for (int i = 0; i < K * N; i++) h_b[i] = __float2half(2.0f);

    half *d_a, *d_b;
    float *d_c;
    cudaMalloc(&d_a, size_A);
    cudaMalloc(&d_b, size_B);
    cudaMalloc(&d_c, size_C);

    cudaMemcpy(d_a, h_a, size_A, cudaMemcpyHostToDevice);
    cudaMemcpy(d_b, h_b, size_B, cudaMemcpyHostToDevice);

    dim3 threadsPerBlock(32,1);

    dim3 numBlocks(N / WMMA_N, M / WMMA_M);

    printf("Launching WMMA GEMM Kernel: Grid(%d, %d), Block(32)\n", numBlocks.x, numBlocks.y);
    wmma_gemm_kernel<<<numBlocks, threadsPerBlock>>>(d_a, d_b, d_c);
    cudaDeviceSynchronize();

    cudaMemcpy(h_c, d_c, size_C, cudaMemcpyDeviceToHost);

    printf("Result Matrix C (Top-Left 4x4):\n");
    for (int i = 0; i < 4; i++) {
        for (int j = 0; j < 4; j++) {
            printf("%6.1f ", h_c[i * N + j]);
        }
        printf("\n");
    }

    cudaFree(d_a); cudaFree(d_b); cudaFree(d_c);
    free(h_a); free(h_b); free(h_c);
    return 0;
}
