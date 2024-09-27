#include <bits/stdc++.h>
#include <cuda.h>
#include <cuda_runtime.h>

// 0.211904 ms
//v5: 基于 gridSize stop 的 loop, 多线程块级规约
// 分配 threads 总数小于元素数量
// 好处：1. 更灵活，可以处理大于启动线程数量的 problem size
//      2. 复用线程，减小线程创建和销毁的开销
//      3. 方便debug：设置block数量和thread数量为 1 时，为串行程序
template<int blockSize>
__device__ void blockReduce(float * sdata) {
    if (blockSize >= 1024) {
        if (threadIdx.x < 512) {
            sdata[threadIdx.x] += sdata[threadIdx.x + 512];
        }
    }
    __syncthreads();
    if (blockSize >= 512) {
        if (threadIdx.x < 256) {
            sdata[threadIdx.x] += sdata[threadIdx.x + 256];
        }
    }
    __syncthreads();
    if (blockSize >= 256) {
        if (threadIdx.x < 128) {
            sdata[threadIdx.x] += sdata[threadIdx.x + 128];
        }
    }
    __syncthreads();
    if (blockSize >= 128) {
        if (threadIdx.x < 64) {
            sdata[threadIdx.x] += sdata[threadIdx.x + 64];
        }
    }
    __syncthreads();
}

__device__ void warpReduce(volatile float* smem, int tid) {
    float x = smem[tid];
    if (blockDim.x >= 64) {
        // 分离读写操作，先读后写
        x += smem[tid + 32]; __syncwarp();
        smem[tid] = x; __syncwarp();
    }
    x += smem[tid + 16]; __syncwarp();
    smem[tid] = x; __syncwarp();

    x += smem[tid + 8]; __syncwarp();
    smem[tid] = x; __syncwarp();

    x += smem[tid + 4]; __syncwarp();
    smem[tid] = x; __syncwarp();

    x += smem[tid + 2]; __syncwarp();
    smem[tid] = x; __syncwarp();

    x += smem[tid + 1]; __syncwarp();
    smem[tid] = x; __syncwarp();
}

template<int blockSize>
__global__ void reduce_v6(float * d_in, float * d_out, uint32_t N) {
    __shared__ float smem[blockSize];

    uint32_t tid = threadIdx.x;

    // cross block
    uint32_t gtid = blockIdx.x * blockDim.x + threadIdx.x;
    uint32_t coverSize = blockDim.x * gridDim.x;
    uint32_t sum = 0;
    for (uint32_t index = gtid; index < N; index += coverSize) {
        sum += d_in[index];
    }
    smem[tid] = sum;
    // for (unsigned int stride = blockDim.x / 2; stride > 32; stride >>= 1) {
    //     if (tid < stride) { 
    //         smem[tid] += smem[tid + stride];
    //     }
    //     __syncthreads();
    // }
    blockReduce<blockSize>(smem);

    if (tid < 32) {
        warpReduce(smem, tid);
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

    float* out = (float *)malloc(1 * sizeof(float));
    float* d_out;
    cudaMalloc((void **)&d_out, 1 * sizeof(float));
    float* d_out_part;
    cudaMalloc((void **)&d_out_part, (gridSize / 2) * sizeof(float));

    for (int i = 0; i < N; ++i) {
        a[i] = 1.0f;
    }

    float groundtruth = N * 1.0f;

    cudaMemcpy(d_a, a, N * sizeof(float), cudaMemcpyHostToDevice);

    dim3 Grid(gridSize / 2);
    dim3 Block(blockSize);

    // warmup 
    reduce_v6<blockSize><<<Grid, Block>>>(d_a, d_out_part, N);
    cudaDeviceSynchronize();
    
    // kernel
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start);
    reduce_v6<blockSize><<<Grid, Block>>>(d_a, d_out_part, N);
    reduce_v6<blockSize><<<   1, Block>>>(d_out_part, d_out, gridSize / 2);
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&milliseconds, start, stop);
    
    cudaMemcpy(out, d_out, 1 * sizeof(float), cudaMemcpyDeviceToHost);

    printf("allocated %d blocks, data counts are %d \n", gridSize, N);
    bool is_right = checkResult(out, groundtruth, 1);
    if (is_right) {
        printf("the answer is right \n");
    }
    else {
        printf("the answer is wrong \n");
        // for (int i = 0; i < gridSize; i++) {
        //     printf("res per block : %f ", out[i]);
        // }
    }
    printf("reduce_v5 latency is %f ms\n", milliseconds);

    cudaFree(d_a);
    cudaFree(d_out);
    free(a);
    free(out);
}