#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <ctime>

#include <cuda_runtime.h>

#define CSC(call)                                                           \
do {                                                                        \
    cudaError_t res = call;                                                 \
    if (res != cudaSuccess) {                                               \
        std::fprintf(stderr, "ERROR: %s:%d: %s\n",                          \
                     __FILE__, __LINE__, cudaGetErrorString(res));          \
        std::exit(0);                                                       \
    }                                                                       \
} while (0)


__device__ float brightness(cudaTextureObject_t tex, int x, int y) {
    uchar4 p = tex2D<uchar4>(tex, x + 0.5f, y + 0.5f);
    return 0.299f * p.x + 0.587f * p.y + 0.114f * p.z;
}

__global__ void sobel(cudaTextureObject_t tex, uchar4* dst, int w, int h) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int idy = blockIdx.y * blockDim.y + threadIdx.y;
    int offx = blockDim.x * gridDim.x;
    int offy = blockDim.y * gridDim.y;
    for (int y = idy; y < h; y += offy) {
        for (int x = idx; x < w; x += offx) {
            float p00 = brightness(tex, x - 1, y - 1);
            float p01 = brightness(tex, x, y - 1);
            float p02 = brightness(tex, x + 1, y - 1);
            float p10 = brightness(tex, x - 1, y);
            float p12 = brightness(tex, x + 1, y);
            float p20 = brightness(tex, x - 1, y + 1);
            float p21 = brightness(tex, x, y + 1);
            float p22 = brightness(tex, x + 1, y + 1);
            float gx = (p02 + 2.0f * p12 + p22) - (p00 + 2.0f * p10 + p20);
            float gy = (p20 + 2.0f * p21 + p22) - (p00 + 2.0f * p01 + p02);
            unsigned char value = (unsigned char)fminf(255.0f, sqrtf(gx * gx + gy * gy));
            dst[y * w + x] = make_uchar4(value, value, value, 0);
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
    if (w <= 0 || h <= 0 || w >= 65536 || h >= 65536 || (size_t)w * h > 100000000) {
        std::fprintf(stderr, "ERROR: invalid image dimensions\n");
        std::exit(0);
    }
    data = (uchar4*)std::malloc(sizeof(uchar4) * w * h);
    if (!data) {
        std::fprintf(stderr, "ERROR: cannot allocate image\n");
        std::exit(0);
    }
    if (fread(data, sizeof(uchar4), (size_t)w * h, fp) < (size_t)w * h) {
        std::fprintf(stderr, "ERROR: broken image file %s\n", path);
        std::exit(0);
    }
    fclose(fp);
}

void write_image(const char* path, int w, int h, uchar4* data) {
    FILE* fp = fopen(path, "wb");
    if (!fp) {
        std::fprintf(stderr, "ERROR: cannot open %s for writing\n", path);
        std::exit(0);
    }
    bool ok = fwrite(&w, sizeof(int), 1, fp) == 1 &&
              fwrite(&h, sizeof(int), 1, fp) == 1 &&
              fwrite(data, sizeof(uchar4), (size_t)w * h, fp) == (size_t)w * h;
    if (fclose(fp) != 0) ok = false;
    if (!ok) {
        std::fprintf(stderr, "ERROR: cannot write %s\n", path);
        std::exit(0);
    }
}

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

int main(int argc, char** argv) {
    char path_in[1024], path_out[1024];
    if (!fgets(path_in, sizeof(path_in), stdin) || !fgets(path_out, sizeof(path_out), stdin)) {
        std::fprintf(stderr, "ERROR: expected input and output paths\n");
        return 0;
    }
    for (char* s : {path_in, path_out}) {
        for (int i = 0; s[i]; ++i) {
            if (s[i] == '\n' || s[i] == '\r') s[i] = '\0';
        }
    }
    int bx = 32, by = 32, tx = 16, ty = 16;
    if (argc != 1) {
        char extra;
        if (argc != 5 || std::sscanf(argv[1], "%d%c", &bx, &extra) != 1 ||
            std::sscanf(argv[2], "%d%c", &by, &extra) != 1 ||
            std::sscanf(argv[3], "%d%c", &tx, &extra) != 1 ||
            std::sscanf(argv[4], "%d%c", &ty, &extra) != 1 ||
            bx < 1 || by < 1 || tx < 1 || ty < 1) {
            std::fprintf(stderr, "ERROR: invalid launch configuration\n");
            return 0;
        }
    }
    int w, h;
    uchar4* img = nullptr;
    read_image(path_in, w, h, img);
    cudaArray_t arr = nullptr;
    cudaTextureObject_t tex;
    make_object_uchar4(img, w, h, arr, tex);
    uchar4* out = nullptr;
    CSC(cudaMalloc(&out, sizeof(uchar4) * w * h));
    cudaEvent_t start, stop;
    CSC(cudaEventCreate(&start));
    CSC(cudaEventCreate(&stop));
    CSC(cudaEventRecord(start));
    sobel<<<dim3(bx, by), dim3(tx, ty)>>>(tex, out, w, h);
    CSC(cudaGetLastError());
    CSC(cudaEventRecord(stop));
    CSC(cudaEventSynchronize(stop));
    float ms;
    CSC(cudaEventElapsedTime(&ms, start, stop));
    std::fprintf(stderr, "kernel_ms = %.6f\n", ms);
    CSC(cudaMemcpy(img, out, sizeof(uchar4) * w * h, cudaMemcpyDeviceToHost));
    write_image(path_out, w, h, img);
    CSC(cudaEventDestroy(start));
    CSC(cudaEventDestroy(stop));
    CSC(cudaDestroyTextureObject(tex));
    CSC(cudaFreeArray(arr));
    CSC(cudaFree(out));
    std::free(img);
    return 0;
}
