#include <stdio.h>
#include <cuda_runtime.h>

int main() {
    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, 0);
    printf("Warp Size: %d\n", prop.warpSize);
    printf("L1 Cache Support: %d\n", prop.globalL1CacheSupported);
    return 0;
}
