#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <ctime>

// CPU-референс для бенчмарка: двухпроходное гауссово размытие (те же формулы, что и в main.cu).
struct F3 { float x, y, z; };
struct U4 { unsigned char x, y, z, w; };
using float3 = F3; using uchar4 = U4;
static float3 make_float3(float a, float b, float c) { return {a, b, c}; }
static uchar4 make_uchar4(unsigned char a, unsigned char b, unsigned char c, unsigned char d) { return {a, b, c, d}; }

static void blur(int w, int h, const uchar4* src, uchar4* dst, int r) {
    int k = 2 * r + 1;
    float* wts = (float*)malloc(sizeof(float) * k);
    double sum = 0.0;
    for (int i = 0; i < k; ++i) {
        double x = i - r;
        wts[i] = r == 0 ? 1.0f : (float)exp(-x * x / (2.0 * r * r));
        sum += wts[i];
    }
    for (int i = 0; i < k; ++i) wts[i] /= (float)sum;

    auto cl = [](int v, int m) { return v < 0 ? 0 : (v > m ? m : v); };
    float3* tmp = (float3*)malloc(sizeof(float3) * w * h);
    for (int y = 0; y < h; ++y)
        for (int x = 0; x < w; ++x) {
            float a = 0, b = 0, c = 0;
            for (int k = -r; k <= r; ++k) {
                float f = wts[k + r];
                uchar4 p = src[y * w + cl(x + k, w - 1)];
                a += f * p.x; b += f * p.y; c += f * p.z;
            }
            tmp[y * w + x] = make_float3(a, b, c);
        }
    for (int y = 0; y < h; ++y)
        for (int x = 0; x < w; ++x) {
            float a = 0, b = 0, c = 0;
            for (int k = -r; k <= r; ++k) {
                float f = wts[k + r];
                float3 t = tmp[cl(y + k, h - 1) * w + x];
                a += f * t.x; b += f * t.y; c += f * t.z;
            }
            int s = y * w + x;
            dst[s] = make_uchar4((unsigned char)fminf(255.5f, fmaxf(0.0f, a)),
                                 (unsigned char)fminf(255.5f, fmaxf(0.0f, b)),
                                 (unsigned char)fminf(255.5f, fmaxf(0.0f, c)), 0);
        }
    free(wts);
    free(tmp);
}

int main(int argc, char** argv) {
    if (argc < 4) { fprintf(stderr, "usage: cpu ref.data out.data r\n"); return 1; }
    FILE* fp = fopen(argv[1], "rb");
    int w, h;
    if (fread(&w, 4, 1, fp) < 1 || fread(&h, 4, 1, fp) < 1) return 1;
    uchar4* src = (uchar4*)malloc(sizeof(uchar4) * w * h);
    if (fread(src, sizeof(uchar4), (size_t)w * h, fp) < (size_t)w * h) return 1;
    fclose(fp);
    int r = atoi(argv[3]);
    uchar4* dst = (uchar4*)malloc(sizeof(uchar4) * w * h);

    struct timespec t0, t1;
    clock_gettime(CLOCK_MONOTONIC, &t0);
    blur(w, h, src, dst, r);
    clock_gettime(CLOCK_MONOTONIC, &t1);
    double ms = (t1.tv_sec - t0.tv_sec) * 1e3 + (t1.tv_nsec - t0.tv_nsec) / 1e6;
    fprintf(stderr, "cpu_ms = %.3f\n", ms);

    FILE* fo = fopen(argv[2], "wb");
    fwrite(&w, 4, 1, fo); fwrite(&h, 4, 1, fo);
    fwrite(dst, sizeof(uchar4), (size_t)w * h, fo);
    fclose(fo);
    return 0;
}
