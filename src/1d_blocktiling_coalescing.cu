#include <cblas.h>
#include <cuda_runtime.h>

#include <algorithm>
#include <cmath>
#include <iomanip>
#include <iostream>
#include <random>
#include <vector>

using uint = unsigned int;

constexpr uint regBlockHeight = 4;
constexpr uint cacheDim = 16; // cacheDim * cacheDim = cacheSize

constexpr uint cK = cacheDim / regBlockHeight; // K = width of A, height of B, shared Dim
constexpr uint cM = cacheDim * regBlockHeight; // M = height of A
constexpr uint cN = cacheDim * regBlockHeight; // N = width  of B

template <typename T, typename accT>
__global__ void oneDBlockTiling(uint A, uint B, uint C,
                                T *ptrA, T *ptrB, T *ptrC)
{
    // thread.x = [0, 4]
    // thread.y = [0, 16]

    __shared__ T aCache[cM][cK]; // 64 x 4
    __shared__ T bCache[cK][cN]; // 4 x 64

    constexpr uint TM = cM / cK;

    // Row = [0, 64], Col = [0, 4]
    uint cacheRowA = threadIdx.x / cK;
    uint cacheColA = threadIdx.x % cK;

    // Row = [0, 4], Col = [0, 64]
    uint cacheRowB = threadIdx.x / cN;
    uint cacheColB = threadIdx.x % cN;

    accT tmpVec[TM] = {accT(0)};

    T *locA = ptrA + cM * blockIdx.y * B;
    T *locB = ptrB + cN * blockIdx.x;
    T *locC = ptrC + cM * blockIdx.y * C + cN * blockIdx.x;

    for (int blockTileIdx = 0; blockTileIdx < B; blockTileIdx += cK)
    {
        aCache[cacheRowA][cacheColA] = locA[cacheRowA * C + blockTileIdx + cacheColA];
        bCache[cacheRowB][cacheColB] = locB[(cacheRowB + blockTileIdx) * B + cacheColB];

        __syncthreads();

        for (int cacheIdx = 0; cacheIdx < cK; cacheIdx++)
        {
            accT tmpB = bCache[cacheIdx][cacheColB];
            for (int warpIdx = 0; warpIdx < TM; warpIdx++)
            {
                tmpVec[warpIdx] +=
                    aCache[cacheRowB * TM + warpIdx][cacheIdx] * tmpB;
            }
        }

        __syncthreads();
    }
    for (int warpIdx = 0; warpIdx < TM; warpIdx++)
    {
        locC[(cacheRowB * TM + warpIdx) * C + cacheColB] =
            tmpVec[warpIdx];
    }
}

void benchmark(uint M, uint K, uint N)
{
    // M = multiple of 64
    // N = multiple of 64
    // K = multiple of 4

    if (M % cM != 0 ||
        K % cK != 0 ||
        N % cN != 0)
    {
        std::cerr
            << "Skipping "
            << M << " x " << K << " x " << N
            << " (M/N must be multiples of 64, "
            << "K must be a multiple of 4)\n";

        return;
    }

    size_t sizeA =
        static_cast<size_t>(M) * K;

    size_t sizeB =
        static_cast<size_t>(K) * N;

    size_t sizeC =
        static_cast<size_t>(M) * N;

    std::vector<float> h_A(sizeA);
    std::vector<float> h_B(sizeB);
    std::vector<float> h_C(sizeC);
    std::vector<float> h_ref(sizeC);

    // Random initialization
    std::mt19937 rng(42);

    std::uniform_real_distribution<float> dist(
        -1.0f, 1.0f);

    for (float &x : h_A)
        x = dist(rng);

    for (float &x : h_B)
        x = dist(rng);

    // CPU reference
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

    // GPU memory
    float *d_A;
    float *d_B;
    float *d_C;

    cudaMalloc(
        &d_A,
        sizeA * sizeof(float));

    cudaMalloc(
        &d_B,
        sizeB * sizeof(float));

    cudaMalloc(
        &d_C,
        sizeC * sizeof(float));

    // Copy inputs
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

    dim3 blockDim(cK * cN);

    dim3 gridDim(N / cN, M / cM);

    // Warmup
    oneDBlockTiling<float, float>
        <<<gridDim, blockDim>>>(
            M, K, N,
            d_A, d_B, d_C);

    cudaError_t err =
        cudaDeviceSynchronize();

    if (err != cudaSuccess)
    {
        std::cerr
            << "Warmup kernel failed: "
            << cudaGetErrorString(err)
            << '\n';

        cudaFree(d_A);
        cudaFree(d_B);
        cudaFree(d_C);

        return;
    }

    cudaEvent_t start;
    cudaEvent_t stop;

    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    constexpr int iterations = 10;

    cudaEventRecord(start);

    for (int i = 0; i < iterations; i++)
    {
        oneDBlockTiling<float, float>
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
        static_cast<double>(total_ms) /
        iterations;

    // Check kernel errors
    err = cudaGetLastError();

    if (err != cudaSuccess)
    {
        std::cerr
            << "Kernel launch failed: "
            << cudaGetErrorString(err)
            << '\n';

        cudaEventDestroy(start);
        cudaEventDestroy(stop);

        cudaFree(d_A);
        cudaFree(d_B);
        cudaFree(d_C);

        return;
    }

    cudaMemcpy(
        h_C.data(),
        d_C,
        sizeC * sizeof(float),
        cudaMemcpyDeviceToHost);

    float max_error = 0.0f;

    for (size_t i = 0; i < sizeC; i++)
    {
        max_error =
            std::max(
                max_error,
                std::abs(h_C[i] - h_ref[i]));
    }

    bool correct =
        max_error < 1e-3f;

    double gflops =
        (2.0 *
         static_cast<double>(M) *
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
              << (std::to_string(gflops) +
                  " GFLOP/s")

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
