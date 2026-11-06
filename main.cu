#include <cstdio>
#include <cstdlib>
#include <cstdint>
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

struct Center { uint32_t r, g, b, count; };
__constant__ Center centers[32];

__global__ void classify(uchar4* pixels, size_t count, int classes) {
    size_t index = blockIdx.x * blockDim.x + threadIdx.x;
    size_t stride = blockDim.x * gridDim.x;
    for (size_t i = index; i < count; i += stride) {
        uchar4 pixel = pixels[i];
        uint64_t best = 0, best_denominator = 1;
        int label = 0;
        for (int j = 0; j < classes; ++j) {
            Center c = centers[j];
            int64_t dr = (int64_t)pixel.x * c.count - c.r;
            int64_t dg = (int64_t)pixel.y * c.count - c.g;
            int64_t db = (int64_t)pixel.z * c.count - c.b;
            uint64_t distance = dr * dr + dg * dg + db * db;
            uint64_t denominator = (uint64_t)c.count * c.count;
            uint64_t left_hi = __umul64hi(distance, best_denominator);
            uint64_t right_hi = __umul64hi(best, denominator);
            if (j == 0 || left_hi < right_hi ||
                (left_hi == right_hi && distance * best_denominator < best * denominator)) {
                best = distance;
                best_denominator = denominator;
                label = j;
            }
        }
        pixel.w = (unsigned char)label;
        pixels[i] = pixel;
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
    if (w <= 0 || h <= 0 || (size_t)w * h > 400000000) {
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
    int blocks = 256, threads = 256;
    if (argc != 1) {
        char extra;
        if (argc != 3 || std::sscanf(argv[1], "%d%c", &blocks, &extra) != 1 ||
            std::sscanf(argv[2], "%d%c", &threads, &extra) != 1 ||
            blocks < 1 || threads < 1) {
            std::fprintf(stderr, "ERROR: invalid launch configuration\n");
            return 0;
        }
    }
    int w, h;
    uchar4* img = nullptr;
    read_image(path_in, w, h, img);
    int classes;
    if (std::scanf("%d", &classes) != 1 || classes < 1 || classes > 32) {
        std::fprintf(stderr, "ERROR: invalid class count\n");
        return 0;
    }
    Center means[32];
    for (int j = 0; j < classes; ++j) {
        int n;
        if (std::scanf("%d", &n) != 1 || n < 1 || n > (1 << 19)) {
            std::fprintf(stderr, "ERROR: invalid sample count\n");
            return 0;
        }
        uint64_t r = 0, g = 0, b = 0;
        for (int i = 0; i < n; ++i) {
            int x, y;
            if (std::scanf("%d%d", &x, &y) != 2 || x < 0 || x >= w || y < 0 || y >= h) {
                std::fprintf(stderr, "ERROR: invalid sample coordinate\n");
                return 0;
            }
            uchar4 p = img[(size_t)y * w + x];
            r += p.x; g += p.y; b += p.z;
        }
        means[j] = {(uint32_t)r, (uint32_t)g, (uint32_t)b, (uint32_t)n};
    }
    size_t bytes = sizeof(uchar4) * (size_t)w * h;
    uchar4* dev = nullptr;
    CSC(cudaMalloc(&dev, bytes));
    CSC(cudaMemcpy(dev, img, bytes, cudaMemcpyHostToDevice));
    CSC(cudaMemcpyToSymbol(centers, means, sizeof(Center) * classes));
    cudaEvent_t start, stop;
    CSC(cudaEventCreate(&start));
    CSC(cudaEventCreate(&stop));
    CSC(cudaEventRecord(start));
    classify<<<blocks, threads>>>(dev, (size_t)w * h, classes);
    CSC(cudaGetLastError());
    CSC(cudaEventRecord(stop));
    CSC(cudaEventSynchronize(stop));
    float ms;
    CSC(cudaEventElapsedTime(&ms, start, stop));
    std::fprintf(stderr, "kernel_ms = %.6f\n", ms);
    CSC(cudaMemcpy(img, dev, bytes, cudaMemcpyDeviceToHost));
    write_image(path_out, w, h, img);
    CSC(cudaEventDestroy(start));
    CSC(cudaEventDestroy(stop));
    CSC(cudaFree(dev));
    std::free(img);
    return 0;
}
