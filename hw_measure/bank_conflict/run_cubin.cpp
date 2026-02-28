#include <iostream>
#include <cuda.h>

// 💡 해커의 필수품: CUDA Driver API 에러 체커 매크로
#define CHECK_CU(call) \
    do { \
        CUresult res = call; \
        if (res != CUDA_SUCCESS) { \
            const char* errStr; \
            cuGetErrorName(res, &errStr); \
            std::cerr << "🔥 CUDA Error [" << errStr << "] at line " << __LINE__ << std::endl; \
            exit(1); \
        } \
    } while (0)

int main() {
    CHECK_CU(cuInit(0));
    CUdevice device;
    CHECK_CU(cuDeviceGet(&device, 0)); 
    CUcontext context;
    CHECK_CU(cuCtxCreate(&context, 0, device));

    CUmodule module;
    CHECK_CU(cuModuleLoad(&module, "patched.cubin"));

    CUfunction no_conflict_func, conflict_func;
    
    // ★★★ 주의: 여기서 에러가 난다면 이름이 틀린 것입니다! editable.sass에서 다시 찾아보세요 ★★★
    std::cout << "Loading functions..." << std::endl;
    CHECK_CU(cuModuleGetFunction(&no_conflict_func, module, "_Z18no_conflict_kernelPf")); 
    
    CHECK_CU(cuModuleGetFunction(&conflict_func, module, "_Z15conflict_kernelPf"));       
    std::cout << "Functions loaded successfully!" << std::endl;

    int threads = 256;
    int blocks = 80;
    CUdeviceptr d_out;
    CHECK_CU(cuMemAlloc(&d_out, blocks * threads * sizeof(float)));

    // 커널 인자값들 (float *out, float4, float4, float4)
    // float4는 메모리상에서 float 4개의 배열과 동일합니다.
    struct float4 { float x, y, z, w; };
    float4 val_a = {1.1f, 1.2f, 1.3f, 1.4f};
    float4 val_b = {2.1f, 2.2f, 2.3f, 2.4f};
    float4 val_c = {0.0f, 0.0f, 0.0f, 0.0f};

    //void* args[] = { &d_out, &val_a, &val_b, &val_c };
    void* args[] = {&d_out};
    CUevent start, stop;
    CHECK_CU(cuEventCreate(&start, CU_EVENT_DEFAULT));
    CHECK_CU(cuEventCreate(&stop, CU_EVENT_DEFAULT));
    float ms_normal = 0, ms_conflict = 0;

    std::cout << "Running No Conflict Kernel..." << std::endl;
    CHECK_CU(cuEventRecord(start, 0));
    CHECK_CU(cuLaunchKernel(no_conflict_func, blocks, 1, 1, threads, 1, 1, 0, 0, args, 0));
    CHECK_CU(cuEventRecord(stop, 0));
    CHECK_CU(cuEventSynchronize(stop));
    CHECK_CU(cuEventElapsedTime(&ms_normal, start, stop));

    std::cout << "Running Max Conflict Kernel..." << std::endl;
    CHECK_CU(cuEventRecord(start, 0));
    CHECK_CU(cuLaunchKernel(conflict_func, blocks, 1, 1, threads, 1, 1, 0, 0, args, 0));
    CHECK_CU(cuEventRecord(stop, 0));
    CHECK_CU(cuEventSynchronize(stop));
    CHECK_CU(cuEventElapsedTime(&ms_conflict, start, stop));

    std::cout << "\nSASS Patched Register Bank Conflict Test" << std::endl;
    std::cout << "=======================================" << std::endl;
    std::cout << "No Conflict Kernel Time: " << ms_normal << " ms" << std::endl;
    std::cout << "Conflict Kernel Time   : " << ms_conflict << " ms" << std::endl;
    
    if(ms_conflict > ms_normal) {
        std::cout << "Slowdown Factor: " << ms_conflict / ms_normal << " x" << std::endl;
    }

    CHECK_CU(cuMemFree(d_out));
    CHECK_CU(cuCtxDestroy(context));
    return 0;
}
