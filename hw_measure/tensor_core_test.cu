#include <iostream>
#include <cuda_fp16.h>
#include <mma.h>

using namespace nvcuda;

const int MATRIX_SIZE = 4096;
const int M = MATRIX_SIZE;
const int N = MATRIX_SIZE;
const int K = MATRIX_SIZE;

__global__ void naiveGemm(half *A, half *B, float *C, int M, int N, int K) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (row < M && col < N) {
        float sum = 0.0f;
        for (int i = 0; i < K; ++i) {
            sum += __half2float(A[row * K + i]) * __half2float(B[i * N + col]);
        }
        C[row * N + col] = sum;
    }
}

__global__ void wmmaGemm(half *A, half *B, float *C, int M, int N, int K) {
    int warpM = (blockIdx.y * blockDim.y + threadIdx.y) / 32;
    int warpN = (blockIdx.x * blockDim.x + threadIdx.x);

    int row = warpM * 16;
    int col = warpN * 16;

    if (row < M && col < N) {
        wmma::fragment<wmma::matrix_a, 16, 16, 16, half, wmma::row_major> a_frag;
	wmma::fragment<wmma::matrix_b, 16, 16, 16, half, wmma::col_major> b_frag;
	wmma::fragment<wmma::accumulator, 16, 16, 16, float> c_frag;

	wmma::fill_fragment(c_frag, 0.0f);

	for (int i = 0; i < K; i+= 16) {
	    wmma::load_matrix_sync(a_frag, A + row * K + i, K);
	    wmma::load_matrix_sync(b_frag, B + i * N + col, N);

	    wmma::mma_sync(c_frag, a_frag, b_frag, c_frag);
	}

	wmma::store_matrix_sync(C + row * N + col, c_frag, N, wmma::mem_row_major);
    }
}

__global__ void initMatrix(half *mat, int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        mat[idx] = __float2half(0.01f); // 임의의 작은 실수값 할당
    }
}

int main() {
    size_t size_AB = M * K * sizeof(half);
    size_t size_C = M * N * sizeof(float);

    half *d_A, *d_B;
    float *d_C;

    cudaMalloc(&d_A, size_AB);
    cudaMalloc(&d_B, size_AB);
    cudaMalloc(&d_C, size_C);

    int numElements = M * K;
    initMatrix<<<(numElements + 255) / 256, 256>>>(d_A, numElements);
    initMatrix<<<(numElements + 255) / 256, 256>>>(d_B, numElements);
    cudaDeviceSynchronize();

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    dim3 threadsNaive(16, 16);
    dim3 blocksNaive((N + 15) / 16, (M + 15) / 16);

    cudaEventRecord(start);
    naiveGemm<<<blocksNaive, threadsNaive>>>(d_A, d_B, d_C, M, N, K);
    cudaError_t err = cudaGetLastError();
    if(err != cudaSuccess) std::cout << cudaGetErrorString(err) << std::endl;
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    float naiveTime = 0;
    cudaEventElapsedTime(&naiveTime, start, stop);

    dim3 threadsWMMA(32, 4);
    dim3 blocksWMMA((N + 15) / 16, (M + (4 * 16) - 1) / (4 * 16));

    cudaEventRecord(start);
    wmmaGemm<<<blocksWMMA, threadsWMMA>>>(d_A, d_B, d_C, M, N, K);
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    float wmmaTime = 0;
    cudaEventElapsedTime(&wmmaTime, start, stop);

    std::cout << "=== Performance Comparison ===" << std::endl;
    std::cout << "Matrix Size: " << M << " x " << N << " x " << K << std::endl;
    std::cout << "CUDA Cores (Naive) Time: " << naiveTime << " ms" << std::endl;
    std::cout << "Tensor Cores (WMMA) Time : " << wmmaTime << " ms" << std::endl;
    std::cout << "Speedup Factor         : " << naiveTime / wmmaTime << " x" << std::endl;

    cudaFree(d_A); cudaFree(d_B); cudaFree(d_C);
    return 0;
}
