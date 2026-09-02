#include <cblas.h>
#include <cuda_runtime.h>

#include <cmath>
#include <iomanip>
#include <iostream>
#include <random>
#include <vector>

using uint = unsigned int;

template <typename T, typename accT>
__global__ void naiveCoalescingGEMM(uint A, uint B, uint C,
                                    T *ptrA, T *ptrB, T *ptrC)
{
    const uint x = blockIdx.x * blockDim.x + threadIdx.x;
    const uint y = blockIdx.y * blockDim.y;

    accT temp = accT(0);

    if (x < C && y < A)
    {
        for (int i = 0; i < B; i++)
        {
            temp += ptrA[y * B + i] * ptrB[i * C + x];
        }

        ptrC[y * C + x] = static_cast<T>(temp);
    }
}

template __global__ void naiveCoalescingGEMM<float, float>(
    uint A, uint B, uint C,
    float *ptrA, float *ptrB, float *ptrC);

void benchmark(uint M, uint K, uint N)
{
    size_t sizeA = static_cast<size_t>(M) * K;
    size_t sizeB = static_cast<size_t>(K) * N;
    size_t sizeC = static_cast<size_t>(M) * N;

    std::vector<float> h_A(sizeA);
    std::vector<float> h_B(sizeB);
    std::vector<float> h_C(sizeC);
    std::vector<float> h_ref(sizeC);

    std::mt19937 rng(42);
    std::uniform_real_distribution<float> dist(-1.0f, 1.0f);

    for (float &x : h_A)
        x = dist(rng);

    for (float &x : h_B)
        x = dist(rng);

    // Reference result
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
    float *d_A, *d_B, *d_C;

    cudaMalloc(&d_A, sizeA * sizeof(float));
    cudaMalloc(&d_B, sizeB * sizeof(float));
    cudaMalloc(&d_C, sizeC * sizeof(float));

    cudaMemcpy(
        d_A, h_A.data(),
        sizeA * sizeof(float),
        cudaMemcpyHostToDevice);

    cudaMemcpy(
        d_B, h_B.data(),
        sizeB * sizeof(float),
        cudaMemcpyHostToDevice);

    // 253 threads per block
    dim3 blockDim(256);

    dim3 gridDim(
        (N + blockDim.x - 1) / blockDim.x,
        (M + blockDim.y - 1) / blockDim.y);

    // Warmup
    naiveCoalescingGEMM<float, float><<<gridDim, blockDim>>>(
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
        naiveCoalescingGEMM<float, float><<<gridDim, blockDim>>>(
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

    double time_ms = total_ms / iterations;

    // Copy result back
    cudaMemcpy(
        h_C.data(),
        d_C,
        sizeC * sizeof(float),
        cudaMemcpyDeviceToHost);

    // Verify
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
        (2.0 * M * K * N) /
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