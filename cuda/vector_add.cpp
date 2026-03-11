#include "func.h"
#include <cuda_runtime.h>
#include <iostream>
#include <cstdlib>

int main() {
    const int N = 1024;           // 向量长度
    size_t size = N * sizeof(float);

    // 主机内存分配
    float *h_A = (float*)malloc(size);
    float *h_B = (float*)malloc(size);
    float *h_C = (float*)malloc(size);

    // 初始化数据
    for (int i = 0; i < N; i++) {
        h_A[i] = static_cast<float>(i);
        h_B[i] = static_cast<float>(i * 2);
    }

    // 设备内存分配
    float *d_A, *d_B, *d_C;
    cudaMalloc((void**)&d_A, size);
    cudaMalloc((void**)&d_B, size);
    cudaMalloc((void**)&d_C, size);

    // 数据拷贝：主机 -> 设备
    cudaMemcpy(d_A, h_A, size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, h_B, size, cudaMemcpyHostToDevice);

    // 调用CUDA函数
    vector_add(d_A, d_B, d_C, N);

    // 结果拷贝：设备 -> 主机
    cudaMemcpy(h_C, d_C, size, cudaMemcpyDeviceToHost);

    // 验证结果
    bool success = true;
    for (int i = 0; i < 10; i++) {  // 只打印前10个
        float expected = h_A[i] + h_B[i];
        std::cout << "C[" << i << "] = " << h_C[i] 
                  << " (expected: " << expected << ")" << std::endl;
        if (abs(h_C[i] - expected) > 1e-5) success = false;
    }
    std::cout << (success ? "✓ 验证通过!" : "✗ 验证失败!") << std::endl;

    // 释放资源
    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);
    free(h_A);
    free(h_B);
    free(h_C);

    return 0;
}