#include <cuda_runtime.h>
#include <cuda.h>
#include <bits/stdc++.h>

// v1: 消除存在的 bank conflict
// latency: 0.441280 ms (相邻), 0.266016 ms （交错） 

template<int blockSize>
__global__ void reduce_v1(const float* d_in, float* d_out) {
    __shared__ float smem[blockSize];
    int tid = threadIdx.x;
    int gtid = blockIdx.x * blockDim.x + threadIdx.x;

    smem[tid] = d_in[gtid];
    __syncthreads();

    // 可解决 bank conflict，由原先一个线程访问2个存储体变为只访问一个存储体
    for (int stride = blockDim.x / 2; stride >= 1; stride >>= 1) {
        if (tid < stride) {
            smem[tid] += smem[tid + stride];
        }
        __syncthreads();
    }

    if (tid == 0) {
        d_out[blockIdx.x] = smem[0];
    }
}

bool checkResult(const float* out, const float groundtruth, int n) {
    float sum = 0.0f;
    for (int i = 0; i < n; ++i) {
        sum += out[i];
    }
    if (sum != groundtruth) return false;
    return true;
}


int main() {
    float milliseconds = 0;
    const int N = 25600000;

    cudaSetDevice(0);
    cudaDeviceProp deviceProp;
    cudaGetDeviceProperties(&deviceProp, 0);

    const int blockSize = 256;
    int gridSize = std::min((N + blockSize - 1) / blockSize, deviceProp.maxGridSize[0]);
    // maxGridSize = 100000

    float* a = (float *)malloc(N * sizeof(float));
    float* d_a;
    cudaMalloc((void **)&d_a, N * sizeof(float));

    float* out = (float *)malloc(gridSize * sizeof(float));
    float* d_out;
    cudaMalloc((void **)&d_out, gridSize * sizeof(float));

    for (int i = 0; i < N; ++i) {
        a[i] = 1.0f;
    }

    float groundtruth = N * 1.0f;

    cudaMemcpy(d_a, a, N * sizeof(float), cudaMemcpyHostToDevice);

    dim3 Grid(gridSize);
    dim3 Block(blockSize);

    // warmup 
    reduce_v0<blockSize><<<Grid, Block>>>(d_a, d_out, N);
    cudaDeviceSynchronize();
    
    // kernel
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start);
    reduce_v0<blockSize><<<Grid, Block>>>(d_a, d_out, N);
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&milliseconds, start, stop);
    
    cudaMemcpy(out, d_out, gridSize * sizeof(float), cudaMemcpyDeviceToHost);

    printf("allocated %d blocks, data counts are %d \n", gridSize, N);
    bool is_right = checkResult(out, groundtruth, gridSize);
    if (is_right) {
        printf("the answer is right \n");
    }
    else {
        printf("the answer is wrong \n");
        // for (int i = 0; i < gridSize; i++) {
        //     printf("res per block : %f ", out[i]);
        // }
    }
    printf("reduce baseline latency is %f ms\n", milliseconds);

    cudaFree(d_a);
    cudaFree(d_out);
    free(a);
    free(out);
}