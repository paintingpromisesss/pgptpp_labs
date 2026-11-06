#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <ctime>
#include <vector>

struct Pixel { unsigned char r, g, b, a; };
struct Mean { uint32_t r, g, b, count; };

int main(int argc, char** argv) {
    if (argc != 3) return 1;
    FILE* in = std::fopen(argv[1], "rb");
    if (!in) return 1;
    int w, h, nc;
    if (std::fread(&w, 4, 1, in) != 1 || std::fread(&h, 4, 1, in) != 1 || w <= 0 || h <= 0) return 1;
    size_t n = (size_t)w * h;
    std::vector<Pixel> pixels(n);
    if (std::fread(pixels.data(), sizeof(Pixel), n, in) != n) return 1;
    std::fclose(in);
    FILE* samples = std::fopen(argv[2], "r");
    if (!samples || std::fscanf(samples, "%d", &nc) != 1 || nc < 1 || nc > 32) return 1;
    Mean means[32];
    for (int j = 0; j < nc; ++j) {
        int count;
        if (std::fscanf(samples, "%d", &count) != 1 || count < 1) return 1;
        uint64_t r = 0, g = 0, b = 0;
        for (int i = 0; i < count; ++i) {
            int x, y;
            if (std::fscanf(samples, "%d%d", &x, &y) != 2 || x < 0 || x >= w || y < 0 || y >= h) return 1;
            Pixel p = pixels[(size_t)y * w + x]; r += p.r; g += p.g; b += p.b;
        }
        means[j] = {(uint32_t)r, (uint32_t)g, (uint32_t)b, (uint32_t)count};
    }
    std::fclose(samples);
    timespec start, stop;
    clock_gettime(CLOCK_MONOTONIC, &start);
    for (auto& p : pixels) {
        uint64_t best = 0, best_denominator = 1;
        int label = 0;
        for (int j = 0; j < nc; ++j) {
            Mean c = means[j];
            int64_t dr = (int64_t)p.r * c.count - c.r;
            int64_t dg = (int64_t)p.g * c.count - c.g;
            int64_t db = (int64_t)p.b * c.count - c.b;
            uint64_t distance = dr*dr + dg*dg + db*db;
            uint64_t denominator = (uint64_t)c.count * c.count;
            if (j == 0 || (__uint128_t)distance * best_denominator < (__uint128_t)best * denominator) {
                best = distance; best_denominator = denominator; label = j;
            }
        }
        p.a = (unsigned char)label;
    }
    clock_gettime(CLOCK_MONOTONIC, &stop);
    std::fprintf(stderr, "cpu_ms = %.6f\n", (stop.tv_sec-start.tv_sec)*1000.0 + (stop.tv_nsec-start.tv_nsec)/1000000.0);
    return 0;
}
