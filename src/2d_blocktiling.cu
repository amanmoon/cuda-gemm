#include <cblas.h>
#include <cuda_runtime.h>

#include <algorithm>
#include <cmath>
#include <iomanip>
#include <iostream>
#include <random>
#include <vector>

using uint = unsigned int;

constexpr uint THREADS_PER_BLOCK = 256;

constexpr uint cK = 16;  // 4
constexpr uint cM = 128; // 64
constexpr uint cN = 128; // 64

constexpr uint TM = 8;
constexpr uint TN = 8;

template <typename T, typename accT>
__global__ void twoDBlockTiling(uint A, uint B, uint C,
                                T *ptrA, T *ptrB, T *ptrC)
{
    __shared__ T aCache[cM][cK];
    __shared__ T bCache[cK][cN];

    uint loadRowA = threadIdx.x / cK; // [0, 32]
    uint loadColA = threadIdx.x % cK; // [0, 8]

    uint loadRowB = threadIdx.x / cN; // [0, 128]
    uint loadColB = threadIdx.x % cN; // [0, 2]

    uint threadRow = threadIdx.x / (cN / TN); // [0, 16]
    uint threadCol = threadIdx.x % (cN / TN); // [0, 16]

    constexpr uint loadAOffSet = THREADS_PER_BLOCK / cK;
    constexpr uint loadBOffSet = THREADS_PER_BLOCK / cN;

    T tmpMat[TM][TN] = {T(0)};

    T *locA = ptrA + blockIdx.y * cM * B;
    T *locB = ptrB + blockIdx.x * cN;
    T *locC = ptrC + cM * blockIdx.y * C + cN * blockIdx.x;

    for (int blockTileIdx = 0; blockTileIdx < B; blockTileIdx += cK)
    {
        for (uint offset = 0; offset < cM; offset += loadAOffSet)
        {
            aCache[loadRowA + offset][loadColA] = locA[(loadRowA + offset) * B + blockTileIdx + loadColA];
        }

        for (uint offset = 0; offset < cK; offset += loadBOffSet)
        {
            bCache[loadRowB + offset][loadColB] = locB[(loadRowB + offset + blockTileIdx) * C + loadColB];
        }

        __syncthreads();

        accT regA[TM] = {accT(0)};
        accT regB[TN] = {accT(0)};

        for (int cacheIdx = 0; cacheIdx < cK; cacheIdx++)
        {
            for (int regIdx = 0; regIdx < TM; regIdx++)
            {
                regA[regIdx] = aCache[threadRow * TM + regIdx][cacheIdx];
            }

            for (int regIdx = 0; regIdx < TN; regIdx++)
            {
                regB[regIdx] = bCache[cacheIdx][threadCol * TN + regIdx];
            }

            for (int col = 0; col < TN; col++)
            {
                for (int row = 0; row < TM; row++)
                {
                    tmpMat[row][col] += regA[row] * regB[col];
                }
            }
        }

        __syncthreads();
    }

    for (int col = 0; col < TN; col++)
    {
        for (int row = 0; row < TM; row++)
        {
            locC[((threadRow * TM) + row) * C + (threadCol * TN) + col] = tmpMat[row][col];
        }
    }
}

void benchmark(uint M, uint K, uint N)
{
    // M = multiple of 64
    // N = multiple of 64
    // K = multiple of 4
    if (M % cM != 0 || K % cK != 0 || N % cN != 0)
    {
        std::cerr << "Skipping " << M << " x " << K << " x " << N
                  << " (M/N must be multiples of 64, K must be a multiple of 4)\n";
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

    // CPU reference (OpenBLAS)
    cblas_sgemm(
        CblasRowMajor, CblasNoTrans, CblasNoTrans,
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

    cudaMalloc(&d_A, sizeA * sizeof(float));
    cudaMalloc(&d_B, sizeB * sizeof(float));
    cudaMalloc(&d_C, sizeC * sizeof(float));

    // Copy inputs
    cudaMemcpy(d_A, h_A.data(), sizeA * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, h_B.data(), sizeB * sizeof(float), cudaMemcpyHostToDevice);

    // --- Kernel Configuration ---
    // 256 threads per block map to a 64x64 block computing 4x4 elements per thread
    dim3 blockDim(THREADS_PER_BLOCK);
    dim3 gridDim((N + cN - 1) / cN, (M + cM - 1) / cM);

    // Warmup
    twoDBlockTiling<float, float><<<gridDim, blockDim>>>(M, K, N, d_A, d_B, d_C);

    cudaError_t err = cudaDeviceSynchronize();
    if (err != cudaSuccess)
    {
        std::cerr << "Warmup kernel failed: " << cudaGetErrorString(err) << '\n';
        cudaFree(d_A);
        cudaFree(d_B);
        cudaFree(d_C);
        return;
    }

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    constexpr int iterations = 10;
    cudaEventRecord(start);

    for (int i = 0; i < iterations; i++)
    {
        twoDBlockTiling<float, float><<<gridDim, blockDim>>>(M, K, N, d_A, d_B, d_C);
    }

    cudaEventRecord(stop);
    cudaEventSynchronize(stop);

    float total_ms;
    cudaEventElapsedTime(&total_ms, start, stop);
    double time_ms = static_cast<double>(total_ms) / iterations;

    // Check kernel errors
    err = cudaGetLastError();
    if (err != cudaSuccess)
    {
        std::cerr << "Kernel launch failed: " << cudaGetErrorString(err) << '\n';
        cudaEventDestroy(start);
        cudaEventDestroy(stop);
        cudaFree(d_A);
        cudaFree(d_B);
        cudaFree(d_C);
        return;
    }

    // Verify Results
    cudaMemcpy(h_C.data(), d_C, sizeC * sizeof(float), cudaMemcpyDeviceToHost);

    float max_error = 0.0f;
    for (size_t i = 0; i < sizeC; i++)
    {
        max_error = std::max(max_error, std::abs(h_C[i] - h_ref[i]));
    }

    bool correct = max_error < 1e-3f;

    // Calculate throughput
    double gflops = (2.0 * static_cast<double>(M) * static_cast<double>(K) * static_cast<double>(N)) / (time_ms * 1e6);

    // Output
    std::cout << std::left
              << std::setw(26) << (std::to_string(M) + " x " + std::to_string(K) + " x " + std::to_string(N))
              << std::setw(16) << (std::to_string(time_ms) + " ms")
              << std::setw(22) << (std::to_string(gflops) + " GFLOP/s")
              << std::setw(10) << (correct ? "PASS" : "FAIL")
              << std::scientific << max_error << '\n';

    // Cleanup
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
    benchmark(1536, 1536, 1536);
    benchmark(2048, 2048, 2048);
    benchmark(3072, 3072, 3072);
    benchmark(4096, 4096, 4096);
    benchmark(5120, 5120, 5120);
    benchmark(6144, 6144, 6144);
    benchmark(7168, 7168, 7168);
    benchmark(8192, 8192, 8192);

    return 0;
}