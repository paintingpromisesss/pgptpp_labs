#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <ctime>
#include <fstream>
#include <vector>

struct Pixel { uint8_t r, g, b, a; };

int main(int argc, char** argv) {
    if (argc != 3) return 1;
    std::ifstream in(argv[1], std::ios::binary);
    int w, h;
    if (!in.read((char*)&w, 4) || !in.read((char*)&h, 4) || w <= 0 || h <= 0) return 1;
    size_t n = (size_t)w * h;
    std::vector<Pixel> input(n), output(n);
    if (!in.read((char*)input.data(), n * sizeof(Pixel))) return 1;
    std::vector<float> gray(n);
    timespec t0, t1;
    clock_gettime(CLOCK_MONOTONIC, &t0);
    for (size_t i = 0; i < n; ++i)
        gray[i] = 0.299f * input[i].r + 0.587f * input[i].g + 0.114f * input[i].b;
    for (int y = 0; y < h; ++y) {
        for (int x = 0; x < w; ++x) {
            float p[3][3];
            for (int dy = -1; dy <= 1; ++dy)
                for (int dx = -1; dx <= 1; ++dx) {
                    int iy = std::max(0, std::min(y + dy, h - 1));
                    int ix = std::max(0, std::min(x + dx, w - 1));
                    p[dy + 1][dx + 1] = gray[(size_t)iy * w + ix];
                }
            float gx = (p[0][2] + 2.0f * p[1][2] + p[2][2]) - (p[0][0] + 2.0f * p[1][0] + p[2][0]);
            float gy = (p[2][0] + 2.0f * p[2][1] + p[2][2]) - (p[0][0] + 2.0f * p[0][1] + p[0][2]);
            uint8_t value = static_cast<uint8_t>(std::min(255.0f, std::sqrt(gx * gx + gy * gy)));
            output[(size_t)y * w + x] = {value, value, value, 0};
        }
    }
    clock_gettime(CLOCK_MONOTONIC, &t1);
    std::fprintf(stderr, "cpu_ms = %.3f\n", (t1.tv_sec-t0.tv_sec)*1e3 + (t1.tv_nsec-t0.tv_nsec)/1e6);
    std::ofstream out(argv[2], std::ios::binary);
    out.write((char*)&w, 4); out.write((char*)&h, 4);
    out.write((char*)output.data(), n * sizeof(Pixel));
    return out ? 0 : 1;
}
