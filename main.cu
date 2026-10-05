#include <iostream>
#include <iomanip>
#include <cmath>
#include <cstdlib>

#define CSC(call)                                                           \
do {                                                                        \
    cudaError_t res = call;                                                 \
    if (res != cudaSuccess) {                                               \
        std::cerr << "ERROR in " << __FILE__ << ":" << __LINE__              \
                  << ". Message: " << cudaGetErrorString(res) << std::endl;\
        std::exit(0);                                                       \
    }                                                                       \
} while(0)

enum class SolutionType {
    TwoRoots,
    OneRoot,
    Imaginary,
    Any,
    Incorrect
};

struct Result {
    SolutionType type{SolutionType::Incorrect};
    float x1{0.0f};
    float x2{0.0f};
};

__global__ void solve_quadratic(const float* in, Result* out) {
    const float a = in[0];
    const float b = in[1];
    const float c = in[2];

    if (a == 0.0f) {
        if (b == 0.0f) {
            if (c == 0.0f) {
                out->type = SolutionType::Any;
            } else {
                out->type = SolutionType::Incorrect;
            }
        } else {
            float x = -c / b;
            if (fabsf(x) == 0.0f) {
                x = 0.0f;
            }
            out->type = SolutionType::OneRoot;
            out->x1 = x;
        }
    } else {
        const float D = b * b - 4.0f * a * c;
        if (D > 0.0f) {
            const float sqrtD = sqrtf(D);
            float x1 = (-b + sqrtD) / (2.0f * a);
            float x2 = (-b - sqrtD) / (2.0f * a);
            if (fabsf(x1) == 0.0f) {
                x1 = 0.0f;
            }
            if (fabsf(x2) == 0.0f) {
                x2 = 0.0f;
            }
            out->type = SolutionType::TwoRoots;
            out->x1 = x1;
            out->x2 = x2;
        } else if (D == 0.0f) {
            float x = -b / (2.0f * a);
            if (fabsf(x) == 0.0f) {
                x = 0.0f;
            }
            out->type = SolutionType::OneRoot;
            out->x1 = x;
        } else {
            out->type = SolutionType::Imaginary;
        }
    }
}

int main() {
    float h_coeffs[3];
    if (!(std::cin >> h_coeffs[0] >> h_coeffs[1] >> h_coeffs[2])) {
        return 0;
    }

    float* d_coeffs = nullptr;
    Result* d_res = nullptr;

    CSC(cudaMalloc(&d_coeffs, sizeof(float) * 3));
    CSC(cudaMalloc(&d_res, sizeof(Result)));

    CSC(cudaMemcpy(d_coeffs, h_coeffs, sizeof(float) * 3, cudaMemcpyHostToDevice));

    solve_quadratic<<<1, 1>>>(d_coeffs, d_res);
    CSC(cudaGetLastError());
    CSC(cudaDeviceSynchronize());

    Result h_res;
    CSC(cudaMemcpy(&h_res, d_res, sizeof(Result), cudaMemcpyDeviceToHost));

    CSC(cudaFree(d_coeffs));
    CSC(cudaFree(d_res));

    std::cout << std::fixed << std::setprecision(6);

    switch (h_res.type) {
        case SolutionType::TwoRoots:
            if (std::abs(h_res.x1) == 0.0f) h_res.x1 = 0.0f;
            if (std::abs(h_res.x2) == 0.0f) h_res.x2 = 0.0f;
            std::cout << h_res.x1 << " " << h_res.x2 << "\n";
            break;
        case SolutionType::OneRoot:
            if (std::abs(h_res.x1) == 0.0f) h_res.x1 = 0.0f;
            std::cout << h_res.x1 << "\n";
            break;
        case SolutionType::Imaginary:
            std::cout << "imaginary\n";
            break;
        case SolutionType::Any:
            std::cout << "any\n";
            break;
        case SolutionType::Incorrect:
            std::cout << "incorrect\n";
            break;
    }

    return 0;
}
