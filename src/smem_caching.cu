#include <cblas.h>
#include <cuda_runtime.h>

#include <algorithm>
#include <cmath>
#include <iomanip>
#include <iostream>
#include <random>
#include <vector>

using uint = unsigned int;

template <typename T, typename accT>
__global__ void smemCachingGEMM(uint A, uint B, uint C,
                                T *ptrA, T *ptrB, T *ptrC)
{
    constexpr uint cacheDim = 16;

    __shared__ T aCache[cacheDim][cacheDim];
    __shared__ T bCache[cacheDim][cacheDim];

    accT tmp = accT(0);

    T *locA = ptrA + cacheDim * blockIdx.y * B;
    T *locB = ptrB + cacheDim * blockIdx.x;
    T *locC = ptrC + cacheDim * blockIdx.y * C + cacheDim * blockIdx.x;

    for (int i = 0; i < B; i += cacheDim)
    {
        aCache[threadIdx.y][threadIdx.x] = locA[threadIdx.y * B + i + threadIdx.x];
        bCache[threadIdx.y][threadIdx.x] = locB[(i + threadIdx.y) * C + threadIdx.x];
        __syncthreads();

        for (int cacheIdx = 0; cacheIdx < cacheDim; cacheIdx++)
        {
            tmp += aCache[threadIdx.y][cacheIdx] * bCache[cacheIdx][threadIdx.x];
        }
        __syncthreads();
    }
    locC[threadIdx.y * C + threadIdx.x] = tmp;
}

template __global__ void smemCachingGEMM<float, float>(
    uint A, uint B, uint C,
    float *ptrA, float *ptrB, float *ptrC);

void benchmark(uint M, uint K, uint N)
{
    constexpr uint cacheDim = 16;

    if (M % cacheDim != 0 ||
        K % cacheDim != 0 ||
        N % cacheDim != 0)
    {
        std::cerr
            << "Skipping "
            << M << " x " << K << " x " << N
            << " (dimensions must be multiples of "
            << cacheDim << ")\n";

        return;
    }

    size_t sizeA = static_cast<size_t>(M) * K;
    size_t sizeB = static_cast<size_t>(K) * N;
    size_t sizeC = static_cast<size_t>(M) * N;

    std::vector<float> h_A(sizeA);
    std::vector<float> h_B(sizeB);
    std::vector<float> h_C(sizeC);
    std::vector<float> h_ref(sizeC);

    // Random initialization
    std::mt19937 rng(42);
    std::uniform_real_distribution<float> dist(-1.0f, 1.0f);

    for (float &x : h_A)
        x = dist(rng);

    for (float &x : h_B)
        x = dist(rng);

    // CPU reference result
    cblas_sgemm(
        CblasRowMajor,
        CblasNoTrans,
        CblasNoTrans,
        M, N, K,
        1.0f,
        h_A.data(), K,
        h_B.data(), N,
        0.0f,
        h_ref.data(), N);

    // Allocate GPU memory
    float *d_A;
    float *d_B;
    float *d_C;

    cudaMalloc(&d_A, sizeA * sizeof(float));
    cudaMalloc(&d_B, sizeB * sizeof(float));
    cudaMalloc(&d_C, sizeC * sizeof(float));

    // Copy inputs to GPU
    cudaMemcpy(
        d_A,
        h_A.data(),
        sizeA * sizeof(float),
        cudaMemcpyHostToDevice);

    cudaMemcpy(
        d_B,
        h_B.data(),
        sizeB * sizeof(float),
        cudaMemcpyHostToDevice);

    // 16 x 16 threads per block
    dim3 blockDim(cacheDim, cacheDim);

    dim3 gridDim(
        N / cacheDim,
        M / cacheDim);

    // Warmup
    smemCachingGEMM<float, float>
        <<<gridDim, blockDim>>>(
            M, K, N,
            d_A, d_B, d_C);

    cudaDeviceSynchronize();

    // Benchmark
    cudaEvent_t start, stop;

    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    constexpr int iterations = 10;

    cudaEventRecord(start);

    for (int i = 0; i < iterations; i++)
    {
        smemCachingGEMM<float, float>
            <<<gridDim, blockDim>>>(
                M, K, N,
                d_A, d_B, d_C);
    }

    cudaEventRecord(stop);
    cudaEventSynchronize(stop);

    float total_ms;

    cudaEventElapsedTime(
        &total_ms,
        start,
        stop);

    double time_ms =
        static_cast<double>(total_ms) / iterations;

    // Copy result back
    cudaMemcpy(
        h_C.data(),
        d_C,
        sizeC * sizeof(float),
        cudaMemcpyDeviceToHost);

    // Verify result
    float max_error = 0.0f;

    for (size_t i = 0; i < sizeC; i++)
    {
        max_error = std::max(
            max_error,
            std::abs(h_C[i] - h_ref[i]));
    }

    bool correct = max_error < 1e-3f;

    // GFLOP/s
    double gflops =
        (2.0 * static_cast<double>(M) *
         static_cast<double>(K) *
         static_cast<double>(N)) /
        (time_ms * 1e6);

    // Output
    std::cout << std::left
              << std::setw(26)
              << (std::to_string(M) + " x " +
                  std::to_string(K) + " x " +
                  std::to_string(N))

              << std::setw(16)
              << (std::to_string(time_ms) + " ms")

              << std::setw(22)
              << (std::to_string(gflops) + " GFLOP/s")

              << std::setw(10)
              << (correct ? "PASS" : "FAIL")

              << std::scientific
              << max_error
              << '\n';

    cudaEventDestroy(start);
    cudaEventDestroy(stop);

    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);
}

int main()
{
    std::cout << std::left
              << std::setw(26) << "Matrix"
              << std::setw(16) << "Time"
              << std::setw(22) << "GFLOP/s"
              << std::setw(10) << "Result"
              << "Max Error\n";

    benchmark(128, 128, 128);
    benchmark(256, 256, 256);
    benchmark(512, 512, 512);
    benchmark(1024, 1024, 1024);
    benchmark(2048, 2048, 2048);
    benchmark(4096, 4096, 4096);

    return 0;
}