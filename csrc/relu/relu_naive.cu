#include "cuda_runtime.h"

__global__ relu_naive(float* x, float* y, N) {
    int tid = threadIdx.x + blockIdx.x * blockDim.x;
    if (tid < N) {
        y[tid] = max(0, x[tid]);
    }
}

int main() {
    dim3 numBlocks = 
    dim3 threadsPerBlock = 
    relu_naive<<<numBlocks, threadsPerBlock>>>();
    return 0;
}
