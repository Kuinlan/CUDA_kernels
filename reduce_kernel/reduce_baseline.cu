#include <bits/stdc++.h>
#include <cuda.h>
#include <cuda_runtime.h>

// baseline 版本
// latency: 452ms 
// single device thread processing
__global__ void reduce_baseline(const int* input, int* output, uint32_t size) {
    int sum = 0;
    for (int i = 0; i < size; ++i) {
        sum += input[i];
    }
    *output = sum;
}

__global__ void reduce_baseline_warmup(const int* input, int* output, uint32_t size) {
    int sum = 0;
    for (int i = 0; i < size; ++i) {
        sum += input[i];
    }
    *output = sum;
}

bool CheckResult(int *out, int groundtruth, int n) {
    for (int i = 0; i < n; i++) {
        if (out[i] != groundtruth) return false;
    }
    return true;
}

int main() {
    float milliseconds = 0;
    const int N = 25600000;

    cudaSetDevice(0);
    cudaDeviceProp deviceProp;
    cudaGetDeviceProperties(&deviceProp, 0);

    const int blockSize = 1;
    int gridSize = 1;

    int* a = (int *)malloc(N * sizeof(int));
    int* d_a;
    cudaMalloc((void **)&d_a, N * sizeof(int));

    int* out = (int *)malloc(gridSize * sizeof(int));
    int* d_out;
    cudaMalloc((void **)&d_out, gridSize * sizeof(int));

    for (int i = 0; i < N; ++i) {
        a[i] = 1;
    }

    int groundtruth = N * 1;

    cudaMemcpy(d_a, a, N * sizeof(int), cudaMemcpyHostToDevice);

    dim3 Grid(gridSize);
    dim3 Block(blockSize);

    // warmup 
    reduce_baseline_warmup<<<Grid, Block>>>(d_a, d_out, N);
    cudaDeviceSynchronize();
    
    // kernel
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start);
    reduce_baseline<<<Grid, Block>>>(d_a, d_out, N);
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&milliseconds, start, stop);
    
    cudaMemcpy(out, d_out, gridSize * sizeof(int), cudaMemcpyDeviceToHost);
    printf("allocated %d blocks, data counts are %d \n", gridSize, N);
    bool is_right = CheckResult(out, groundtruth, gridSize);
    if (is_right) {
        printf("the answer is right \n");
    }
    else {
        printf("the answer is wrong \n");
        for (int i = 0; i < gridSize; i++) {
            printf("res per block : %lf ", out[i]);
        }
    }
    printf("reduce baseline latency is %f ms\n", milliseconds);

    cudaFree(d_a);
    cudaFree(d_out);
    free(a);
    free(out);
}