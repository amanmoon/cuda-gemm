#include <cblas.h>
#include <chrono>
#include <cmath>
#include <iomanip>
#include <iostream>
#include <random>
#include <vector>

using uint = unsigned int;

template <typename T, typename accT>
void GEMM(uint A, uint B, uint C,
          T *ptrA, T *ptrB, T *ptrC)
{
    for (int row = 0; row < A; row++)
    {
        for (int col = 0; col < C; col++)
        {
            accT temp = accT(0);

            for (int k = 0; k < B; k++)
            {
                temp += ptrA[row * B + k] *
                        ptrB[k * C + col];
            }

            ptrC[row * C + col] = static_cast<T>(temp);
        }
    }
}

template void GEMM<float, double>(
    uint A, uint B, uint C,
    float *ptrA, float *ptrB, float *ptrC);

void benchmark(uint M, uint K, uint N)
{
    std::vector<float> A(M * K);
    std::vector<float> B(K * N);
    std::vector<float> C(M * N);
    std::vector<float> C_ref(M * N);

    // Random input
    std::mt19937 rng(42);
    std::uniform_real_distribution<float> dist(-1.0f, 1.0f);

    for (float &x : A)
        x = dist(rng);

    for (float &x : B)
        x = dist(rng);

    cblas_sgemm(
        CblasRowMajor,
        CblasNoTrans,
        CblasNoTrans,
        M, N, K,
        1.0f,
        A.data(), K,
        B.data(), N,
        0.0f,
        C_ref.data(), N);

    GEMM<float, double>(
        M, K, N,
        A.data(),
        B.data(),
        C.data());

    float max_error = 0.0f;

    for (size_t i = 0; i < C.size(); i++)
    {
        max_error = std::max(
            max_error,
            std::abs(C[i] - C_ref[i]));
    }

    bool correct = max_error < 1e-4f;

    // Warmup
    GEMM<float, double>(
        M, K, N,
        A.data(),
        B.data(),
        C.data());

    constexpr int iterations = 5;

    auto start = std::chrono::high_resolution_clock::now();

    for (int i = 0; i < iterations; i++)
    {
        GEMM<float, double>(
            M, K, N,
            A.data(),
            B.data(),
            C.data());
    }

    auto end = std::chrono::high_resolution_clock::now();

    double time_ms =
        std::chrono::duration<double, std::milli>(
            end - start)
            .count() /
        iterations;

    double gflops =
        (2.0 * M * K * N) /
        (time_ms * 1e6);

    std::cout << std::left
              << std::setw(26) << (std::to_string(M) + " x " + std::to_string(K) + " x " + std::to_string(N));

    std::cout << std::setw(16)
              << (std::to_string(time_ms) + " ms");

    std::cout << std::setw(18)
              << (std::to_string(gflops) + " GFLOP/s");

    std::cout << std::setw(10)
              << (correct ? "PASS" : "FAIL");

    std::cout << std::scientific
              << max_error
              << '\n';
}

int main()
{
    std::cout << std::left
              << std::setw(26) << "Matrix"
              << std::setw(16) << "Time"
              << std::setw(18) << "GFLOP/s"
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