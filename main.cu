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

__global__ void vector_add(double* a, const double* b, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int offset = blockDim.x * gridDim.x;
    for (int i = idx; i < n; i += offset) {
        a[i] += b[i];
    }
}

int main() {
    std::ios_base::sync_with_stdio(false);
    std::cin.tie(nullptr);

    int n;
    if (!(std::cin >> n) || n <= 0) {
        std::cout << "\n";
        return 0;
    }

    double* h_a = (double*)std::malloc(sizeof(double) * n);
    double* h_b = (double*)std::malloc(sizeof(double) * n);

    for (int i = 0; i < n; ++i) {
        std::cin >> h_a[i];
    }

    for (int i = 0; i < n; ++i) {
        std::cin >> h_b[i];
    }

    double* d_a = nullptr;
    double* d_b = nullptr;

    CSC(cudaMalloc(&d_a, sizeof(double) * n));
    CSC(cudaMalloc(&d_b, sizeof(double) * n));

    CSC(cudaMemcpy(d_a, h_a, sizeof(double) * n, cudaMemcpyHostToDevice));
    CSC(cudaMemcpy(d_b, h_b, sizeof(double) * n, cudaMemcpyHostToDevice));

    const int threads = 256;
    const int blocks = 256;

    vector_add<<<blocks, threads>>>(d_a, d_b, n);
    CSC(cudaGetLastError());
    CSC(cudaDeviceSynchronize());

    CSC(cudaMemcpy(h_a, d_a, sizeof(double) * n, cudaMemcpyDeviceToHost));

    CSC(cudaFree(d_a));
    CSC(cudaFree(d_b));

    std::cout << std::scientific << std::setprecision(10);
    for (int i = 0; i < n; ++i) {
        if (std::abs(h_a[i]) == 0.0) {
            h_a[i] = 0.0;
        }
        std::cout << h_a[i] << (i == n - 1 ? "" : " ");
    }
    std::cout << "\n";

    std::free(h_a);
    std::free(h_b);

    return 0;
}
