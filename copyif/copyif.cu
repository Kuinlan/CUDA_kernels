#include <bits/stdc++.h>
#include <cuda.h>
#include <cuda_runtime.h>

#define BLOCK_SIZE  128
#define BATCH_SIZE    4
#define          H 4800
#define          W 4800

// [H * W, W, 1]
// [i / batchsize, i, j]
//#define check_runtime(op) CHECK(op, #op, __FILE__, __LINE__)
//
//bool CHECK(cudaError_t code, )
__device__ int atomicAggInc(int *ctr) {
  unsigned int active = __activemask();
  int leader = __ffs(active) - 1;
  int change = __popc(active);//warp mask中为1的数量
  int lane_mask_lt;
  asm("mov.u32 %0, %%lanemask_lt;" : "=r"(lane_mask_lt));
  unsigned int rank = __popc(active & lane_mask_lt); // 比当前线程id小且值为1的mask之和
  int warp_res = 0;
  if(rank == 0)//leader thread of every warp
    warp_res = atomicAdd(ctr, change);//compute global offset of warp
  warp_res = __shfl_sync(active, warp_res, leader);//broadcast warp_res of leader thread to every active thread
  return warp_res + rank; // global offset + local offset = final offset，即L91表示的atomicAggInc(nres), 为src[i]的最终的写入到dst的位置
}

__global__ void copy_if(int *input, int *output,
                        const int size, const int batch,
                        const int height, const int width) {
  __shared__ int blockRes;
  int thread_res;
  const int bid = blockIdx.x;
  const int tid = threadIdx.x;
  const int warp_lane = tid % 32;

  int grid_size = gridDim.x;

  int src;
  int* dst_row;

  for (int i = bid; i < batch * height; i += grid_size) {
    if (tid == 0) {
      blockRes = 0;
    }
    __syncthreads();

    for (int j = tid; j - warp_lane < width; j += BLOCK_SIZE) {
      src = *(input + (i / BATCH_SIZE) * H * W + i * W + j);
      dst_row = input + (i / BATCH_SIZE) * H * W + i * W;
      if (j < width && src > 0) {
        thread_res = atomicAggInc(&blockRes);
        *(dst_row + thread_res) = src;
      }
    }
  }

}

int main() {
  cudaError_t code;
  int cuda_device_id;
  code = cudaGetDeviceCount(&cuda_device_id);
  cudaSetDevice(cuda_device_id);

  int size = BATCH_SIZE * H * W;
  float milisecond = 0.0f;

  int *in_h;
  int *out_h;
  int *in_d;
  int *out_d;

  in_h = (int*)malloc(size * sizeof(int));
  out_h = (int*)malloc(size * sizeof(int));

  cudaMalloc((void **)&in_d, size * sizeof(int));
  cudaMalloc((void **)&out_d, size * sizeof(int));


  dim3 Grid(128);
  dim3 Block(128);
  cudaEvent_t start, end;
  cudaEventCreate(&start);
  cudaEventCreate(&end);
  cudaEventRecord(start);
  copy_if<<<Grid, Block>>>(in_d, out_d, size, BATCH_SIZE, H, W);
  cudaEventRecord(end);
  cudaEventElapsedTime(&milisecond, start, end);
  printf("elaped time: %f", milisecond);
}