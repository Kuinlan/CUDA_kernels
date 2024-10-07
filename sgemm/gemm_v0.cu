// sgemm naive implementation

#include <stdio.h>
#include <stdlib.h>
#include "assert.h"

// CUDA runtime
#include <cuda_runtime.h>
#include <cublas_v2.h>

// cal offset from row col and ld , in row-major matrix, ld is the width of the matrix
#define OFFSET(row, col, ld) ((row) * (ld) + (col))

// transfer float4
#define FETCH_FLOAT4(pointer) (reinterpret_cast<float4*>(&(pointer))[0])

#define checkCudaErrors(func)				\
{									\
    cudaError_t e = (func);			\
    if(e != cudaSuccess)						                \
        printf ("%s %d CUDA: %s\n", __FILE__,  __LINE__, cudaGetErrorString(e));		\
}

__global__ void Sgemm_naive(
    float * __restrict__ A,
    float * __restrict__ B,
    float * __restrict__ C,
    const int M,
    const int N,
    const int K) {
  // Block index
  int bx = blockIdx.x;
  int by = blockIdx.y;

  // Thread index
  int tx = threadIdx.x;
  int ty = threadIdx.y;

  // Global index
  int gx = bx * blockDim.x + tx;
  int gy = by * blockDim.y + ty;

  if (gy >= M || gx >= N) return;

  int sum = 0;
  for (int k = 0; k < K; ++k) {
    sum += A[gy * N + k] * B[k * N + gx];
  }
  C[gy * N + gx] = sum;
}

int main(int argc, char** argv) {
  if (argc != 4) {
    printf("usage: ./main [M] [K] [N]\n");
    exit(0);
  }
  size_t M = atoi(argv[1]);
  size_t K = atoi(argv[2]);
  size_t N = atoi(argv[3]);

  assert( M%8 == 0);
  assert( N%8 == 0);
  assert( K%8 == 0);

  size_t bytes_A = sizeof(float) * M * K;
  size_t bytes_B = sizeof(float) * K * N;
  size_t bytes_C = sizeof(float) * M * N;
  float* h_A = (float*)malloc(bytes_A);
  float* h_B = (float*)malloc(bytes_B);
  float* h_C = (float*)malloc(bytes_C);
  float* h_C1 = (float*)malloc(bytes_C);

  float* d_A;
  float* d_B;
  float* d_C;

  checkCudaErrors(cudaMalloc(&d_A, bytes_A));
  checkCudaErrors(cudaMalloc(&d_B, bytes_B));
  checkCudaErrors(cudaMalloc(&d_C, bytes_C));
  double msecPerMatrixMul[2] = {0, 0};
  double gigaFlops[2] = {0, 0};
  double flopsPerMatrixMul = 2.0 * M * N * K;

  // generate A
  for( int i = 0; i < M * K; i++ ){
    h_A[i] = i / 100;
  }

  // generate B
  for( int i = 0; i < K * N; i++ ) {
    h_B[i] = i % 13;
  }

  checkCudaErrors(cudaMemcpy( d_A, h_A, bytes_A, cudaMemcpyHostToDevice));
  checkCudaErrors(cudaMemcpy( d_B, h_B, bytes_B, cudaMemcpyHostToDevice));

  cudaEvent_t start, stop;
  checkCudaErrors(cudaEventCreate(&start));
  checkCudaErrors(cudaEventCreate(&stop));
  float msecTotal = 0;
  int nIter = 1;

  checkCudaErrors(cudaMemcpy( d_C, h_C, bytes_C, cudaMemcpyHostToDevice));
  checkCudaErrors(cudaEventRecord(start));
  for (int run = 0 ; run < nIter; run ++ ) {
    dim3 dimBlock(16, 16);
    dim3 dimGrid(N / 16, M / 16);
    Sgemm_naive<<< dimGrid, dimBlock >>>(d_A, d_B, d_C, M, N, K);
  }
  checkCudaErrors(cudaEventRecord(stop));
  checkCudaErrors(cudaEventSynchronize(stop));
  checkCudaErrors(cudaEventElapsedTime(&msecTotal, start, stop));


  checkCudaErrors(cudaMemcpy( h_C, d_C, bytes_C, cudaMemcpyDeviceToHost));

  msecPerMatrixMul[0] = msecTotal / nIter;
  gigaFlops[0] = (flopsPerMatrixMul * 1.0e-9f) / (msecPerMatrixMul[0] / 1000.0f);
  printf( "My gemm Performance= %.2f GFlop/s, Time= %.3f msec, Size= %.0f Ops,\n",
          gigaFlops[0],
          msecPerMatrixMul[0],
          flopsPerMatrixMul);

  // cublas
  cublasHandle_t blas_handle;
  cublasCreate(&blas_handle);
  float alpha = 1.0;
  float beta = 0;
  checkCudaErrors(cudaMemcpy( d_C, h_C, bytes_C, cudaMemcpyHostToDevice));

  // warmup
  cublasSgemm (blas_handle, CUBLAS_OP_T, CUBLAS_OP_T,
               M, N, K, &alpha,
               d_A, K, d_B, N, &beta, d_C, N
  );
  checkCudaErrors(cudaEventRecord(start));
  for (int run = 0 ; run < nIter; run ++ ) {
    cublasSgemm (blas_handle, CUBLAS_OP_T, CUBLAS_OP_T,
                 M, N, K, &alpha,
                 d_A, K, d_B, N, &beta, d_C, N
    );
  }
  checkCudaErrors(cudaEventRecord(stop));
  checkCudaErrors(cudaEventSynchronize(stop));
  checkCudaErrors(cudaEventElapsedTime(&msecTotal, start, stop));

  checkCudaErrors(cudaMemcpy( h_C1, d_C, bytes_C, cudaMemcpyDeviceToHost));

  msecPerMatrixMul[1] = msecTotal / nIter;
  gigaFlops[1] = (flopsPerMatrixMul * 1.0e-9f) / (msecPerMatrixMul[1] / 1000.0f);
  printf( "CuBlas Performance= %.2f GFlop/s, Time= %.3f msec, Size= %.0f Ops,\n",
          gigaFlops[1],
          msecPerMatrixMul[1],
          flopsPerMatrixMul);

  cublasDestroy(blas_handle);

  double eps = 1.e-6;  // machine zero
  bool correct = true;
  for (int i = 0; i < M * N; i++) {
    int row = i / N;
    int col = i % N;
    double abs_err = fabs(h_C[i] - h_C1[col * M + row]);
    double dot_length = M;
    double abs_val = fabs(h_C[i]);
    double rel_err = abs_err / abs_val / dot_length;
    if (rel_err > eps) {
      printf("Error! Matrix[%05d]=%.8f, ref=%.8f error term is > %E\n",
             i, h_C[i], h_C1[col * M + row], eps);
      correct = false;
      break;
    }
  }

  printf("%s\n", correct ? "Result= PASS" : "Result= FAIL");
  printf("ratio= %f\n", gigaFlops[0] / gigaFlops[1]);

  // Free Memory
  cudaFree(d_A);
  cudaFree(d_B);
  cudaFree(d_C);

  free(h_A);
  free(h_B);
  free(h_C);
  free(h_C1);
}
