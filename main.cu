#include <iostream>
#include <iomanip>
#include <cstdlib>
#include <cmath>

#define CSC(call)                                                           \
do {                                                                        \
    cudaError_t res = call;                                                 \
    if (res != cudaSuccess) {                                               \
        std::cerr << "ERROR in " << __FILE__ << ":" << __LINE__              \
                  << ". Message: " << cudaGetErrorString(res) << std::endl;\
        std::exit(0);                                                       \
    }                                                                       \
} while(0)

__global__ void odd_even_sort_phase(float* arr, int n, int phase) {
    int idx = blockDim.x * blockIdx.x + threadIdx.x;
    int offset = blockDim.x * gridDim.x;
    for (int i = 2 * idx + phase; i + 1 < n; i += 2 * offset) {
        if (arr[i] > arr[i + 1]) {
            float tmp = arr[i];
            arr[i] = arr[i + 1];
            arr[i + 1] = tmp;
        }
    }
}

int main() {
    std::ios_base::sync_with_stdio(false);
    std::cin.tie(nullptr);

    int n;
    std::cin >> n;

    float* h_arr = (float*)std::malloc(sizeof(float) * n);
    for (int i = 0; i < n; ++i) {
        std::cin >> h_arr[i];
    }

    float* d_arr = nullptr;
    CSC(cudaMalloc(&d_arr, sizeof(float) * n));
    CSC(cudaMemcpy(d_arr, h_arr, sizeof(float) * n, cudaMemcpyHostToDevice));

    const int threads = 256;
    int blocks = (n / 2 + threads - 1) / threads;
    if (blocks == 0) blocks = 1;
    if (blocks > 1024) blocks = 1024;

    for (int step = 0; step < n; ++step) {
        odd_even_sort_phase<<<blocks, threads>>>(d_arr, n, step % 2);
    }

    CSC(cudaGetLastError());
    CSC(cudaDeviceSynchronize());

    CSC(cudaMemcpy(h_arr, d_arr, sizeof(float) * n, cudaMemcpyDeviceToHost));
    CSC(cudaFree(d_arr));

    std::cout << std::scientific << std::setprecision(6);
    for (int i = 0; i < n; ++i) {
        if (std::abs(h_arr[i]) == 0.0f) {
            h_arr[i] = 0.0f;
        }
        std::cout << h_arr[i] << (i == n - 1 ? "" : " ");
    }
    std::cout << std::endl;

    std::free(h_arr);
    return 0;
}
