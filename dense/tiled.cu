#include "gemm.cuh"

// TILE_SIZE는 32로 설정 (Warp 크기와 일치, Bank Conflict 관리 용이)
#define TILE_SIZE 32

__global__ void tiled_gemm_kernel(float* A, float* B, float* C, int n) {
    // 1. [Shared Memory 할당]
    // A와 B 모두 Row-Major로 저장합니다.
    // [32][33] 패딩을 적용하여 Bank Conflict를 원천 차단합니다.
    __shared__ float tile_A[TILE_SIZE][TILE_SIZE + 1];
    __shared__ float tile_B[TILE_SIZE][TILE_SIZE + 1];

    // 2. [좌표 계산]
    // bx, by는 C 행렬의 타일(블록) 좌표입니다.
    // tx, ty는 타일 내부의 픽셀(스레드) 좌표입니다.
    int bx = blockIdx.x;
    int by = blockIdx.y;
    int tx = threadIdx.x;
    int ty = threadIdx.y;

    // 현재 스레드가 계산해야 할 C 행렬의 실제 위치 (Global Row/Col)
    int row = by * TILE_SIZE + ty;
    int col = bx * TILE_SIZE + tx;

    // 3. [레지스터 초기화]
    // atomicAdd를 안 쓰는 대신, 레지스터에 값을 누적합니다.
    // 이 'val'은 오직 나(현재 스레드)만 건드립니다. (Private)
    float val = 0.0f;

    // 4. [Main Loop] K 차원을 따라 타일 단위로 이동
    for (int k = 0; k < n; k += TILE_SIZE) {

        // --- Phase 1: Global Memory -> Shared Memory 로딩 (Packing) ---
        
        // [A 로딩]
        // A는 Row-Major이므로 가로로 읽어옵니다. (Coalesced Access)
        // A[row][k + tx] 위치의 값을 가져옵니다.
        if (row < n && (k + tx) < n) {
            tile_A[ty][tx] = A[row * n + (k + tx)];
        } else {
            tile_A[ty][tx] = 0.0f; // 범위 밖은 0으로 채움 (Padding)
        }

        // [B 로딩]
        // B도 Row-Major로 저장되어 있다고 가정합니다.
        // B[k + ty][col] 위치의 값을 가져옵니다.
        // 스레드들이 'col' 방향(가로)으로 연속된 값을 읽으므로 Coalesced Access가 유지됩니다.
        if (col < n && (k + ty) < n) {
            tile_B[ty][tx] = B[(k + ty) * n + col];
        } else {
            tile_B[ty][tx] = 0.0f;
        }

        // 로딩이 끝날 때까지 모든 스레드 대기
        __syncthreads();

        // --- Phase 2: Compute (Shared Memory 연산) ---
        
        // 여기가 성능의 핵심입니다 (Amortization).
        // HBM 접근 없이 SRAM(Shared Memory) 내부에서 32번의 곱셈/덧셈을 수행합니다.
        // loop 'i'는 타일 내부의 K 인덱스입니다.
        for (int i = 0; i < TILE_SIZE; i++) {
            // A[ty][i] : 같은 행(Row)의 데이터를 i에 따라 이동하며 읽음
            // B[i][tx] : 같은 열(Col)의 데이터를 i에 따라 이동하며 읽음
            
            // [Bank Conflict 분석]
            // tile_B[i][tx] 접근 시:
            // 워프 내 스레드들은 'i'가 같고 'tx'가 0~31로 다름.
            // Row-Major + Padding[33] 덕분에 서로 다른 Bank에 접근 -> Conflict Free!
            val += tile_A[ty][i] * tile_B[i][tx];
        }

        // 다음 타일 로딩을 위해, 현재 타일 사용이 끝날 때까지 대기
        __syncthreads();
    }

    // 5. [결과 저장]
    // 반복문이 끝나면 레지스터에 최종 결과가 모입니다.
    // 각 스레드는 서로 다른 (row, col) 위치에 쓰므로 경쟁 상태(Race Condition)가 없습니다.
    // -> 따라서 atomicAdd가 필요 없습니다!
    if (row < n && col < n) {
        C[row * n + col] = val;
    }
}

// Host 호출 함수
void run_tiled_gemm(float* d_A, float* d_B, float* d_C, int n) {
    dim3 block(TILE_SIZE, TILE_SIZE); // 32 x 32 스레드
    dim3 grid((n + TILE_SIZE - 1) / TILE_SIZE, (n + TILE_SIZE - 1) / TILE_SIZE);

    tiled_gemm_kernel<<<grid, block>>>(d_A, d_B, d_C, n);
}
