#include <bits/stdc++.h>
#include <cuda.h>
#include "cuda_runtime.h"

#define WarpSize 32

template <int blockSize>
__device__ WarpShuffle(float sum) {
    if (blockSize >= 32) sum += __shfl_down_sync(0xffffffff, sum, 16);
    if (blockSize >= 16) sum += __shfl_down_sync(0xffffffff, sum,  8);
    if (blockSize >=  8) sum += __shfl_down_sync(0xffffffff, sum,  4);
    if (blockSize >=  4) sum += __shfl_down_sync(0xffffffff, sum,  2);
    if (blockSize >=  2) sum += __shfl_down_sync(0xffffffff, sum,  1);
    return sum;
}

template <int blockSize>
__global__ void reduce_warp_level(float *d_in, float *d_out, unsigned int n) {
    float sum = 0;

    uint32_t tid = threadIdx.x;
    uint32_t gtid = blockIdx.x * blockDim.x + threadIdx.x;
    uint32_t coverSize = gridDim.x * blockDim.x;

    for (uint32_t index = gtid; index < n; index += coverSize) {
        sum += d_in[index];
    }

    __shared__ float WarpSums[blockSize / WarpSize];
    const int laneId = tid % WarpSize;
    const int warpId = tid / WarpSize;
    sum = WarpShuffle<blockSize>(sum);
    if (laneId == 0) {
        WarpSums[warpId] = sum;
    }
    __syncthreads();

    sum = (tid < blockSize / WarpSize) ? WarpSums[tid] : 0;
    if (warpId == 0) {
        sum = WarpShuffle<blockSize / WarpSize>(sum);
    }

    if (tid == 0) {
        d_out[blockIdx.x] = sum;
    }
}