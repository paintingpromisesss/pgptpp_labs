#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <ctime>

#include <cuda_runtime.h>

#define CSC(call)                                                           \
do {                                                                        \
    cudaError_t res = call;                                                 \
    if (res != cudaSuccess) {                                               \
        std::fprintf(stderr, "ERROR in %s:%d. Message: %s\n",                \
                     __FILE__, __LINE__, cudaGetErrorString(res));          \
        std::exit(0);                                                       \
    }                                                                       \
} while (0)

// Normalized Gaussian kernel w[i] = exp(-i^2 / (2 r^2)), built on the host.
// Separable two-pass scheme: complexity O(w*h*r) instead of O(w*h*r^2).

#define blur_tex(__tex, __i, __j) tex2D<uchar4>((__tex), (__j) + 0.5f, (__i) + 0.5f)
#define tmp_tex(__tex, __i, __j) tex2D<float4>((__tex), (__j) + 0.5f, (__i) + 0.5f)

__global__ void gauss_horizontal(cudaTextureObject_t src, float4* dst,
                                 int w, int h, const float* wts, int r) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int idy = blockIdx.y * blockDim.y + threadIdx.y;
    int offx = blockDim.x * gridDim.x;
    int offy = blockDim.y * gridDim.y;

    for (int i = idy; i < h; i += offy) {
        for (int j = idx; j < w; j += offx) {
            float3 acc = make_float3(0.0f, 0.0f, 0.0f);
            for (int k = -r; k <= r; ++k) {
                float f = wts[k + r];
                uchar4 p = blur_tex(src, i, j + k); // texture clamp extends the image borders
                acc.x += f * p.x;
                acc.y += f * p.y;
                acc.z += f * p.z;
            }
            dst[i * w + j] = make_float4(acc.x, acc.y, acc.z, 0.0f);
        }
    }
}

__global__ void gauss_vertical(cudaTextureObject_t tmp, uchar4* dst,
                               int w, int h, const float* wts, int r) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int idy = blockIdx.y * blockDim.y + threadIdx.y;
    int offx = blockDim.x * gridDim.x;
    int offy = blockDim.y * gridDim.y;

    for (int i = idy; i < h; i += offy) {
        for (int j = idx; j < w; j += offx) {
            float3 acc = make_float3(0.0f, 0.0f, 0.0f);
            for (int k = -r; k <= r; ++k) {
                float f = wts[k + r];
                float4 p = tmp_tex(tmp, i + k, j); // texture clamp extends the image borders
                acc.x += f * p.x;
                acc.y += f * p.y;
                acc.z += f * p.z;
            }
            dst[i * w + j] = make_uchar4(
                (unsigned char)fminf(255.5f, fmaxf(0.0f, acc.x)),
                (unsigned char)fminf(255.5f, fmaxf(0.0f, acc.y)),
                (unsigned char)fminf(255.5f, fmaxf(0.0f, acc.z)), 0);
        }
    }
}

void read_image(const char* path, int& w, int& h, uchar4*& data) {
    FILE* fp = fopen(path, "rb");
    if (!fp) {
        std::fprintf(stderr, "ERROR: cannot open %s\n", path);
        std::exit(0);
    }
    if (fread(&w, sizeof(int), 1, fp) < 1 || fread(&h, sizeof(int), 1, fp) < 1) {
        std::fprintf(stderr, "ERROR: broken image file %s\n", path);
        std::exit(0);
    }
    data = (uchar4*)std::malloc(sizeof(uchar4) * w * h);
    if (fread(data, sizeof(uchar4), (size_t)w * h, fp) < (size_t)w * h) {
        std::fprintf(stderr, "ERROR: broken image file %s\n", path);
        std::exit(0);
    }
    fclose(fp);
}

void write_image(const char* path, int w, int h, uchar4* data) {
    FILE* fp = fopen(path, "wb");
    fwrite(&w, sizeof(int), 1, fp);
    fwrite(&h, sizeof(int), 1, fp);
    fwrite(data, sizeof(uchar4), (size_t)w * h, fp);
    fclose(fp);
}

// Texture object over a cudaArray with Clamp addressing:
// border pixels are duplicated beyond the image bounds (ip = max(min(i, h), 1)).
void make_object_uchar4(uchar4* host, size_t w, size_t h,
                        cudaArray_t& arr, cudaTextureObject_t& tex) {
    cudaChannelFormatDesc desc = cudaCreateChannelDesc<uchar4>();
    CSC(cudaMallocArray(&arr, &desc, w, h));
    CSC(cudaMemcpy2DToArray(arr, 0, 0, host, sizeof(uchar4) * w,
                            sizeof(uchar4) * w, h, cudaMemcpyHostToDevice));

    cudaResourceDesc rd = {};
    rd.resType = cudaResourceTypeArray;
    rd.res.array.array = arr;

    cudaTextureDesc td = {};
    td.addressMode[0] = cudaAddressModeClamp;
    td.addressMode[1] = cudaAddressModeClamp;
    td.filterMode = cudaFilterModePoint;
    td.normalizedCoords = false;

    CSC(cudaCreateTextureObject(&tex, &rd, &td, nullptr));
}

void make_object_float4(float4* devFrom, size_t w, size_t h,
                        cudaArray_t& arr, cudaTextureObject_t& tex) {
    cudaChannelFormatDesc desc = cudaCreateChannelDesc<float4>();
    CSC(cudaMallocArray(&arr, &desc, w, h));
    CSC(cudaMemcpy2DToArray(arr, 0, 0, devFrom, sizeof(float4) * w,
                            sizeof(float4) * w, h, cudaMemcpyDeviceToDevice));

    cudaResourceDesc rd = {};
    rd.resType = cudaResourceTypeArray;
    rd.res.array.array = arr;

    cudaTextureDesc td = {};
    td.addressMode[0] = cudaAddressModeClamp;
    td.addressMode[1] = cudaAddressModeClamp;
    td.filterMode = cudaFilterModePoint;
    td.normalizedCoords = false;

    CSC(cudaCreateTextureObject(&tex, &rd, &td, nullptr));
}

int main() {
    const int MAX_PATH = 1024;
    char path_in[MAX_PATH], path_out[MAX_PATH];
    if (!fgets(path_in, MAX_PATH, stdin) || !fgets(path_out, MAX_PATH, stdin)) {
        std::fprintf(stderr, "ERROR: expected input and output paths\n");
        return 0;
    }
    // strip newline
    for (char* s : {path_in, path_out}) {
        for (int i = 0; s[i]; ++i) {
            if (s[i] == '\n' || s[i] == '\r') s[i] = '\0';
        }
    }

    int r;
    std::scanf("%d", &r);

    // Grid config for benchmarks (default: 32x32 blocks, 16x16 threads)
    int bx = 32, by = 32, tx = 16, ty = 16;
    if (std::scanf("%d %d", &bx, &by) == 2) {
        std::scanf("%d %d", &tx, &ty);
    }

    int w, h;
    uchar4* h_src = nullptr;
    read_image(path_in, w, h, h_src);

    // Gaussian kernel weights
    int k = 2 * r + 1;
    float* h_wts = (float*)std::malloc(sizeof(float) * k);
    double sum = 0.0;
    for (int i = 0; i < k; ++i) {
        double x = i - r;
        h_wts[i] = (float)std::exp(-x * x / (2.0 * r * r));
        sum += h_wts[i];
    }
    for (int i = 0; i < k; ++i) h_wts[i] /= (float)sum;

    float* d_wts = nullptr;
    CSC(cudaMalloc(&d_wts, sizeof(float) * k));
    CSC(cudaMemcpy(d_wts, h_wts, sizeof(float) * k, cudaMemcpyHostToDevice));

    float4* d_tmp = nullptr;
    uchar4* d_out = nullptr;
    CSC(cudaMalloc(&d_tmp, sizeof(float4) * w * h));
    CSC(cudaMalloc(&d_out, sizeof(uchar4) * w * h));

    cudaArray_t arr_src = nullptr;
    cudaTextureObject_t tex_src;
    make_object_uchar4(h_src, w, h, arr_src, tex_src);

    const dim3 block(tx, ty);
    dim3 grid(bx, by);

    cudaEvent_t start_all, start_k, stop_k, stop_all;
    CSC(cudaEventCreate(&start_all));
    CSC(cudaEventCreate(&start_k));
    CSC(cudaEventCreate(&stop_k));
    CSC(cudaEventCreate(&stop_all));
    CSC(cudaEventRecord(start_all));

    CSC(cudaEventRecord(start_k));
    gauss_horizontal<<<grid, block>>>(tex_src, d_tmp, w, h, d_wts, r);
    CSC(cudaGetLastError());

    cudaArray_t arr_tmp = nullptr;
    cudaTextureObject_t tex_tmp;
    make_object_float4(d_tmp, w, h, arr_tmp, tex_tmp);
    gauss_vertical<<<grid, block>>>(tex_tmp, d_out, w, h, d_wts, r);
    CSC(cudaGetLastError());
    CSC(cudaEventRecord(stop_k));

    uchar4* h_out = (uchar4*)std::malloc(sizeof(uchar4) * w * h);
    CSC(cudaMemcpy(h_out, d_out, sizeof(uchar4) * w * h, cudaMemcpyDeviceToHost));

    CSC(cudaEventRecord(stop_all));
    CSC(cudaEventSynchronize(stop_all));
    float ms_all, ms_k;
    CSC(cudaEventElapsedTime(&ms_all, start_all, stop_all));
    CSC(cudaEventElapsedTime(&ms_k, start_k, stop_k));
    std::fprintf(stderr, "kernel_ms = %.6f total_ms = %.6f\n", ms_k, ms_all);

    write_image(path_out, w, h, h_out);

    CSC(cudaDestroyTextureObject(tex_src));
    CSC(cudaDestroyTextureObject(tex_tmp));
    CSC(cudaFreeArray(arr_src));
    CSC(cudaFreeArray(arr_tmp));
    CSC(cudaFree(d_wts));
    CSC(cudaFree(d_tmp));
    CSC(cudaFree(d_out));
    std::free(h_src);
    std::free(h_out);
    std::free(h_wts);
    return 0;
}
