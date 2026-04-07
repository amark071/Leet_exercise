#include "func.h"
#include <cuda_runtime.h>

// 向量加法
__global__ void vector_add_kernel(const float* A, const float* B, float* C, int N) {
    int idx = blockDim.x * blockIdx.x + threadIdx.x;
    if (idx < N) {
        C[idx] = A[idx] + B[idx];
    }
}

extern "C" void vector_add(const float* A, const float* B, float* C, int N) {
    int threadsPerBlock = 256;
    int blocksPerGrid = (N + threadsPerBlock - 1) / threadsPerBlock;

    vector_add_kernel<<<blocksPerGrid, threadsPerBlock>>>(A, B, C, N);
    cudaDeviceSynchronize();
}

// 矩阵乘法
__global__ void matrix_multiplication_kernel(const float* A, const float* B, float* C, int M, int N,
                                             int K) {
    int col = blockDim.x * blockIdx.x + threadIdx.x;
    int row = blockDim.y * blockIdx.y + threadIdx.y;

    if(row < M && col < K){
        float sum = 0.0;
        for(int i = 0 ; i < N ;i++){
            sum += A[row*N+i]*B[K*i+col];
        }
        C[row*K+col] = sum;
    }
}

// 改进一：利用共享内存实现高速读取
__global__ void matrix_multiplication_kernel_1(const float* A, const float* B, float* C, int M, int N, int K){
  
  int BLOCK = 16;
  const int tx = threadIdx.x;
  const int ty = threadIdx.y;
  const int bx = blockIdx.x;
  const int by = blockIdx.y;

  // 注意到这种写法的前提是共享内存的块大小和线程块大小一致
  const int row = by * BLOCK + ty;
  const int col = bx * BLOCK + tx;

  if (row >= M || col >= K) return;

  __shared__ float Ashare[BLOCK][BLOCK];
  __shared__ float Bshare[BLOCK][BLOCK];

  float sum = 0.0f;

  for (int t = 0; t < N; t += BLOCK) {
    Ashare[ty][tx] = A[row * N + t + tx];
    Bshare[tx][ty] = B[(t + ty) * K + col];
    __syncthreads();
	// 注意这里的一个小巧思：我们输入的时候把B转置一下，这样在之后累加乘的时候，读取的就是行主元的B了

    #pragma unroll
    // 展开循环，加速 
    for (int k = 0; k < BLOCK; ++k) {
      sum += Ashare[ty][k] * Bshare[tx][k];
    }
    __syncthreads();
  }

  C[row * K + col] = sum;
}

// A, B, C are device pointers (i.e. pointers to memory on the GPU)
extern "C" void matrix_multiplication(const float* A, const float* B, float* C, int M, int N, int K) {
    dim3 threadsPerBlock(16, 16);
    dim3 blocksPerGrid((K + threadsPerBlock.x - 1) / threadsPerBlock.x,
                       (M + threadsPerBlock.y - 1) / threadsPerBlock.y);

    matrix_multiplication_kernel<<<blocksPerGrid, threadsPerBlock>>>(A, B, C, M, N, K);
    cudaDeviceSynchronize();
}
