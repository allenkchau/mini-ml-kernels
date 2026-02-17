#include <cuda_runtime.h>

// start with naive version

__global__ void layernorm() {

}

int main() {
    // launch kernel
    dim3 threadsPerBlock = dim3();
    dim3 numBlocks = dim3();
    layernorm<<<>>>();

    return 0;
}
