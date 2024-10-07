// sgemm one thread for muti data point without prefetch
// blocks of A is stored in smem transposed

#include <stdio.h>
#include <stdlib.h>
#include "assert.h"

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

// K: ldA
// N: ldB
template <
    const int BLOCK_SIZE_M,  // height of block of C that each thread block calculate
    const int BLOCK_SIZE_K,  // width of block of A that each thread block load into shared memory
    const int BLOCK_SIZE_N,  // width of block of C that each thread block calculate
    const int THREAD_SIZE_Y, // height of block of C that each thread calculate
    const int THREAD_SIZE_X  // width of block of C that each thread calculate
>
__global__ void Sgemm(
    float * __restrict__ A,
    float * __restrict__ B,
    float * __restrict__ C,
    const int M,
    const int N,
    const int K) {

  int bx = blockIdx.x;
  int by = blockIdx.y;

  int tx = threadIdx.x;
  int ty = threadIdx.y;

  int tid = ty * blockDim.x + tx;

  // smem
  __shared__ float smemA[BLOCK_SIZE_K][BLOCK_SIZE_M];  // transposed A
  __shared__ float smemB[BLOCK_SIZE_K][BLOCK_SIZE_N];

  // register for C
  float accum[THREAD_SIZE_Y][THREAD_SIZE_X] = {0};

  //register for storing fragment of A and B
  float frag_a [THREAD_SIZE_Y];
  float frag_b [THREAD_SIZE_X];

  // one thread writes mutiple data points
  const int A_TILE_THREAD_PER_ROW = BLOCK_SIZE_K / 4; // a thread load 4 data from HBM to smem
  const int B_TILE_THREAD_PER_ROW = BLOCK_SIZE_N / 4; // 32

  // row number and col number that needs to be loaded by this thread
  const int A_TILE_ROW_START = tid / A_TILE_THREAD_PER_ROW;
  const int B_TILE_ROW_START = tid / B_TILE_THREAD_PER_ROW;

  const int A_TILE_COL = (tid % A_TILE_THREAD_PER_ROW) * 4;
  const int B_TILE_COL = (tid % B_TILE_THREAD_PER_ROW) * 4;

  const int A_TILE_ROW_STRIDE = (blockDim.x * blockDim.y) / A_TILE_THREAD_PER_ROW; // 8
  const int B_TILE_ROW_STRIDE = (blockDim.x * blockDim.y) / B_TILE_THREAD_PER_ROW; // 8

  // local address for a block to perform k-split
  A = &A[BLOCK_SIZE_M * by * K];
  B = &B[BLOCK_SIZE_N * bx];

  #pragma unroll
  // K split
  for (int k = 0; k < K; k += BLOCK_SIZE_K) {
    // load from global to smem
    // block size stride
    // 针对线程块中的线程需要多批次地将数据从 gmem 搬运到 smem, 循环次数等于搬运次数
    #pragma unroll
    for (int i = A_TILE_ROW_START; i < BLOCK_SIZE_M; i += A_TILE_ROW_STRIDE) {
      // note A is transposed after loading
      // use register to buffer
      float4 buffer = FETCH_FLOAT4(A[OFFSET(i, A_TILE_COL + k, K)]);
      smemA[A_TILE_COL][i] = buffer.x;
      smemA[A_TILE_COL + 1][i] = buffer.y;
      smemA[A_TILE_COL + 2][i] = buffer.z;
      smemA[A_TILE_COL + 3][i] = buffer.w;
    }

    #pragma unroll
    for (int i = B_TILE_ROW_START; i < BLOCK_SIZE_K; i += B_TILE_ROW_STRIDE) {
      FETCH_FLOAT4(smemB[i][B_TILE_COL]) = FETCH_FLOAT4(B[OFFSET(i + k, B_TILE_COL, N)]);
    }

    __syncthreads(); // 保证读写顺序

    // 一个线程计算 [THREAD_SIZE_Y, THREAD_SIZE_X] 个数据
    #pragma unroll
    for (int i = 0; i < BLOCK_SIZE_K; ++i) {
      #pragma unroll
      for (int j = 0; j < THREAD_SIZE_Y; j += 4) {
        FETCH_FLOAT4(frag_a[j]) = FETCH_FLOAT4(smemA[i][ty * THREAD_SIZE_Y + j]);
      }
      #pragma unroll
      for (int j = 0; j < THREAD_SIZE_X; j += 4) {
        FETCH_FLOAT4(frag_b[j]) = FETCH_FLOAT4(smemB[i][tx * THREAD_SIZE_X + j]);
      }
      #pragma unroll
      for (int y = 0; y < THREAD_SIZE_Y; ++y) {
        #pragma unroll
        for (int x = 0; x < THREAD_SIZE_X; ++x) {
          accum[y][x] += frag_a[y] * frag_b[x];
        }
      }
    }
    __syncthreads();
  }
  // write from register back to C
#pragma unroll
  for (int thread_y = 0; thread_y < THREAD_SIZE_Y; ++thread_y) {
#pragma unroll
    for (int thread_x = 0; thread_x < THREAD_SIZE_X; thread_x += 4) {
      FETCH_FLOAT4(C[OFFSET(by * BLOCK_SIZE_M + ty * THREAD_SIZE_Y + thread_y,
                            bx * BLOCK_SIZE_N + tx * THREAD_SIZE_X + thread_x,
                            N)]) = FETCH_FLOAT4(accum[thread_y][thread_x]);
    }
  }

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

  const int BLOCK_SIZE_M = 128;
  const int BLOCK_SIZE_K = 8;
  const int BLOCK_SIZE_N = 128;
  const int THREAD_SIZE_X = 8;
  const int THREAD_SIZE_Y = 8;

  // generate A
  for( int i = 0; i < M * K; i++ ){
    h_A[i] = i / 666;
  }

  // generate B
  for( int i = 0; i < K * N; i++ ) {
    h_B[i] = i % 666;
  }

  checkCudaErrors(cudaMemcpy( d_A, h_A, bytes_A, cudaMemcpyHostToDevice));
  checkCudaErrors(cudaMemcpy( d_B, h_B, bytes_B, cudaMemcpyHostToDevice));

  cudaEvent_t start, stop;
  checkCudaErrors(cudaEventCreate(&start));
  checkCudaErrors(cudaEventCreate(&stop));
  float msecTotal = 0;
  int nIter = 1000;

  checkCudaErrors(cudaMemcpy( d_C, h_C, bytes_C, cudaMemcpyHostToDevice));
  checkCudaErrors(cudaEventRecord(start));
  for (int run = 0 ; run < nIter; run ++ ) {
    dim3 dimBlock(BLOCK_SIZE_N / THREAD_SIZE_X, BLOCK_SIZE_M / THREAD_SIZE_Y);
    dim3 dimGrid(N / BLOCK_SIZE_N, M / BLOCK_SIZE_M);
    Sgemm<BLOCK_SIZE_M, BLOCK_SIZE_K, BLOCK_SIZE_N, THREAD_SIZE_Y, THREAD_SIZE_X>
    <<< dimGrid, dimBlock >>>(d_A, d_B, d_C, M, N, K);
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
