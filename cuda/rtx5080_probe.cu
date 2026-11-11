#include <cuda_runtime.h>

#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <cstring>

#define CUDA_CHECK(call)                                                        \
    do {                                                                        \
        cudaError_t _err = (call);                                              \
        if (_err != cudaSuccess) {                                              \
            std::fprintf(stderr, "CUDA error %s:%d: %s\\n", __FILE__, __LINE__, \
                         cudaGetErrorString(_err));                             \
            return 1;                                                           \
        }                                                                       \
    } while (0)

static double gbps(size_t bytes, int reps, float ms)
{
    if (ms <= 0.0f) return 0.0;
    return ((double)bytes * (double)reps / 1.0e9) / ((double)ms / 1000.0);
}

static int timed_copy(void *dst, const void *src, size_t bytes,
                      cudaMemcpyKind kind, int reps, float *elapsed_ms)
{
    cudaEvent_t start = nullptr, stop = nullptr;
    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));

    CUDA_CHECK(cudaMemcpyAsync(dst, src, bytes, kind, 0));
    CUDA_CHECK(cudaStreamSynchronize(0));

    CUDA_CHECK(cudaEventRecord(start, 0));
    for (int i = 0; i < reps; ++i)
        CUDA_CHECK(cudaMemcpyAsync(dst, src, bytes, kind, 0));
    CUDA_CHECK(cudaEventRecord(stop, 0));
    CUDA_CHECK(cudaEventSynchronize(stop));
    CUDA_CHECK(cudaEventElapsedTime(elapsed_ms, start, stop));

    CUDA_CHECK(cudaEventDestroy(start));
    CUDA_CHECK(cudaEventDestroy(stop));
    return 0;
}

int main()
{
    std::puts("K3_RTX_PROBE_VERSION=1");

    int driver = 0, runtime = 0, count = 0;
    CUDA_CHECK(cudaDriverGetVersion(&driver));
    CUDA_CHECK(cudaRuntimeGetVersion(&runtime));
    CUDA_CHECK(cudaGetDeviceCount(&count));

    std::printf("cuda.driver=%d\\n", driver);
    std::printf("cuda.runtime=%d\\n", runtime);
    std::printf("gpu.count=%d\\n", count);

    if (count < 1) {
        std::fprintf(stderr, "No CUDA device found.\\n");
        return 2;
    }

    int selected = -1;
    for (int i = 0; i < count; ++i) {
        cudaDeviceProp p{};
        CUDA_CHECK(cudaGetDeviceProperties(&p, i));
        std::printf("gpu.%d.name=%s\\n", i, p.name);
        std::printf("gpu.%d.cc=%d.%d\\n", i, p.major, p.minor);
        std::printf("gpu.%d.vram_gib=%.3f\\n", i,
                    (double)p.totalGlobalMem / (1024.0 * 1024.0 * 1024.0));
        std::printf("gpu.%d.multiprocessors=%d\\n", i, p.multiProcessorCount);
        std::printf("gpu.%d.memory_bus_bits=%d\\n", i, p.memoryBusWidth);

        if (selected < 0 && std::strstr(p.name, "RTX 5080"))
            selected = i;
    }

    if (selected < 0) {
        std::fprintf(stderr,
                     "RTX 5080 was not found by name. Refusing to silently benchmark "
                     "a different GPU.\\n");
        return 3;
    }

    CUDA_CHECK(cudaSetDevice(selected));

    cudaDeviceProp p{};
    CUDA_CHECK(cudaGetDeviceProperties(&p, selected));
    if (p.major != 12 || p.minor != 0) {
        std::fprintf(stderr,
                     "Expected RTX 5080 compute capability 12.0, got %d.%d. "
                     "Investigate driver/toolkit/device detection before proceeding.\\n",
                     p.major, p.minor);
        return 4;
    }

    size_t free_bytes = 0, total_bytes = 0;
    CUDA_CHECK(cudaMemGetInfo(&free_bytes, &total_bytes));
    std::printf("selected.device=%d\\n", selected);
    std::printf("selected.free_vram_gib=%.3f\\n",
                (double)free_bytes / (1024.0 * 1024.0 * 1024.0));

    const size_t max_probe = 256ull * 1024ull * 1024ull;
    const size_t conservative = free_bytes / 8;
    const size_t bytes = std::max<size_t>(
        32ull * 1024ull * 1024ull,
        std::min(max_probe, conservative));

    void *host = nullptr;
    void *device = nullptr;
    CUDA_CHECK(cudaHostAlloc(&host, bytes, cudaHostAllocDefault));
    CUDA_CHECK(cudaMalloc(&device, bytes));
    std::memset(host, 0x5a, bytes);

    constexpr int reps = 10;
    float h2d_ms = 0.0f, d2h_ms = 0.0f;

    if (timed_copy(device, host, bytes, cudaMemcpyHostToDevice, reps, &h2d_ms))
        return 5;
    if (timed_copy(host, device, bytes, cudaMemcpyDeviceToHost, reps, &d2h_ms))
        return 6;

    std::printf("probe.bytes=%zu\\n", bytes);
    std::printf("probe.repetitions=%d\\n", reps);
    std::printf("pcie.h2d_gbps=%.3f\\n", gbps(bytes, reps, h2d_ms));
    std::printf("pcie.d2h_gbps=%.3f\\n", gbps(bytes, reps, d2h_ms));

    CUDA_CHECK(cudaFree(device));
    CUDA_CHECK(cudaFreeHost(host));
    CUDA_CHECK(cudaDeviceSynchronize());

    std::puts("probe.status=PASS");
    return 0;
}
