#include <cblas.h>
#include <cuda_runtime.h>

#include <algorithm>
#include <iomanip>
#include <iostream>
#include <random>
#include <vector>
#include <cmath>

using uint = unsigned int;

constexpr uint WARP_SIZE = 32;
constexpr uint PAD = 4;

constexpr uint cacheM = 128;
constexpr uint cacheK = 32;  // should be divisible by 4; for float4 loading of data
constexpr uint cacheN = 128; // should be divisible by 4; for float4 loading of data

// numWrapsXDim * numWrapsYDim < 32
constexpr uint numWrapsPerXDim = 4;
constexpr uint numWrapsPerYDim = 2;

constexpr uint numSubSecInWrapPerXDim = 2;
constexpr uint numSubSecInWrapPerYDim = 2;

// numThreadsPerXDim * numThreadsPerYDim == WRAP_SIZE
// we should use all the threads in a wrap for computing
constexpr uint numThreadsPerXDim = 4;
constexpr uint numThreadsPerYDim = WARP_SIZE / numThreadsPerXDim;

// precompute based on above listed values; dont change this.
// wrapM * wrapN = wrap tile (tile which is computed by one wrap)
constexpr uint wrapM = cacheM / numWrapsPerYDim; // no of rows; height of each wrap
constexpr uint wrapN = cacheN / numWrapsPerXDim; // no of columns; width of each wrap

// subsections in wrap
constexpr uint wrapSubM = wrapM / numSubSecInWrapPerYDim;
constexpr uint wrapSubN = wrapN / numSubSecInWrapPerXDim;

// threadM * threadN = thread tile (tile which is computed by one thread)
constexpr uint threadM = wrapSubM / numThreadsPerYDim;
constexpr uint threadN = wrapSubN / numThreadsPerXDim;

constexpr uint THREADS_PER_BLOCK = numWrapsPerXDim * numWrapsPerYDim * WARP_SIZE;

// A = M x K
// B = K x N
// C = M x N

template <typename T, typename accT>
__global__ void wrapTiling(const uint M, const uint K, const uint N, T *__restrict__ ptrA, T *__restrict__ ptrB, T *__restrict__ ptrC)
{
    uint cacheRow = blockIdx.y;
    uint cacheCol = blockIdx.x;

    uint wrapIdx = threadIdx.x / WARP_SIZE;

    uint wrapCol = wrapIdx % numWrapsPerXDim;
    uint wrapRow = wrapIdx / numWrapsPerXDim;

    // thread number inside of the wrap
    uint innerThreadIdx = threadIdx.x % WARP_SIZE;

    uint innerThreadCol = innerThreadIdx % numThreadsPerXDim;
    uint innerThreadRow = innerThreadIdx / numThreadsPerXDim;

    // smem cache and cache loading ptrs
    __shared__ T cacheA[cacheK][cacheM + PAD];
    __shared__ T cacheB[cacheK][cacheN];

    uint loadACol = threadIdx.x % (cacheK / 4); // divide by 4 cause we will use float4 (128bit load) for loading data from gmem to smem
    uint loadARow = threadIdx.x / (cacheK / 4);

    uint loadBCol = threadIdx.x % (cacheN / 4);
    uint loadBRow = threadIdx.x / (cacheN / 4);

    constexpr uint movOffsetA = THREADS_PER_BLOCK / (cacheK / 4);
    constexpr uint movOffsetB = THREADS_PER_BLOCK / (cacheN / 4);

    T tmpMat[numSubSecInWrapPerXDim * numSubSecInWrapPerYDim * threadM * threadN] = {T(0)};

    // move all the pointers to the start of the cache tile to be computed.
    ptrA += cacheRow * cacheM * K;
    ptrB += cacheCol * cacheN;
    ptrC += cacheRow * cacheM * N + cacheCol * cacheN;

    #pragma unroll
    for (int blockTileIdx = 0; blockTileIdx < K; blockTileIdx += cacheK)
    {

        // load data in smem cache
        #pragma unroll
        for (int offset = 0; offset < cacheM; offset += movOffsetA)
        {
            float4 tmp = reinterpret_cast<const float4 *>(&ptrA[(loadARow + offset) * K + loadACol * 4])[0];

            // cacheA is loaded in a transposed manner
            cacheA[loadACol * 4 + 0][loadARow + offset] = tmp.x;
            cacheA[loadACol * 4 + 1][loadARow + offset] = tmp.y;
            cacheA[loadACol * 4 + 2][loadARow + offset] = tmp.z;
            cacheA[loadACol * 4 + 3][loadARow + offset] = tmp.w;
        }

        #pragma unroll
        for (int offset = 0; offset < cacheK; offset += movOffsetB)
        {
            reinterpret_cast<float4 *>(&cacheB[loadBRow + offset][loadBCol * 4])[0] = reinterpret_cast<const float4 *>(&ptrB[(loadBRow + offset) * N + loadBCol * 4])[0];
        }
        __syncthreads();

        accT registerA[numSubSecInWrapPerYDim][threadM] = {accT(0)};
        accT registerB[numSubSecInWrapPerXDim][threadN] = {accT(0)};

        // move inside a subsection of the cache tile
        #pragma unroll
        for (int cacheTileIdx = 0; cacheTileIdx < cacheK; cacheTileIdx++)
        {

            // load register A
            #pragma unroll
            for (int subWrapYIdx = 0; subWrapYIdx < numSubSecInWrapPerYDim; subWrapYIdx++)
            {

                #pragma unroll
                for (int regIdx = 0; regIdx < threadM; regIdx++)
                {
                    registerA[subWrapYIdx][regIdx] = cacheA[cacheTileIdx][(wrapRow * wrapM) + subWrapYIdx * wrapSubM + innerThreadRow * threadM + regIdx];
                }
            }

            // load register B
            #pragma unroll
            for (int subWrapXIdx = 0; subWrapXIdx < numSubSecInWrapPerXDim; subWrapXIdx++)
            {

                #pragma unroll
                for (int regIdx = 0; regIdx < threadN; regIdx++)
                {
                    registerB[subWrapXIdx][regIdx] = cacheB[cacheTileIdx][(wrapCol * wrapN) + subWrapXIdx * wrapSubN + innerThreadCol * threadN + regIdx];
                }
            }

            // compute outer product
            #pragma unroll
            for (int subWrapYIdx = 0; subWrapYIdx < numSubSecInWrapPerYDim; subWrapYIdx++)
            {

                #pragma unroll
                for (int subWrapXIdx = 0; subWrapXIdx < numSubSecInWrapPerXDim; subWrapXIdx++)
                {

                    #pragma unroll
                    for (int regIdY = 0; regIdY < threadM; regIdY++)
                    {

                        #pragma unroll
                        for (int regIdX = 0; regIdX < threadN; regIdX++)
                        {
                            tmpMat[(subWrapYIdx * threadM + regIdY) * (numSubSecInWrapPerXDim * threadN) + (subWrapXIdx * threadN) + regIdX] += registerA[subWrapYIdx][regIdY] * registerB[subWrapXIdx][regIdX];
                        }
                    }
                }
            }
        }

        ptrA += cacheK;
        ptrB += cacheK * N;

        __syncthreads();
    }

    // move value from tmpMat to C
    #pragma unroll
    for (int subWrapYIdx = 0; subWrapYIdx < numSubSecInWrapPerYDim; subWrapYIdx++)
    {
        #pragma unroll
        for (int subWrapXIdx = 0; subWrapXIdx < numSubSecInWrapPerXDim; subWrapXIdx++)
        {
            #pragma unroll
            for (int regIdY = 0; regIdY < threadM; regIdY++)
            {
                #pragma unroll
                for (int regIdX = 0; regIdX < threadN; regIdX += 4)
                {
                    float *Cptr = ptrC + (wrapRow * wrapM + subWrapYIdx * wrapSubM + innerThreadRow * threadM + regIdY) * N + (wrapCol * wrapN + subWrapXIdx * wrapSubN + innerThreadCol * threadN + regIdX);

                    const int base = (subWrapYIdx * threadM + regIdY) * (numSubSecInWrapPerXDim * threadN) + (subWrapXIdx * threadN) + regIdX;

                    float4 tmp;
                    tmp.x = tmpMat[base + 0];
                    tmp.y = tmpMat[base + 1];
                    tmp.z = tmpMat[base + 2];
                    tmp.w = tmpMat[base + 3];

                    reinterpret_cast<float4 *>(Cptr)[0] = tmp;
                }
            }
        }
    }
}

void benchmark(uint M, uint K, uint N)
{
    // This kernel requires complete tiles.
    //
    // M must be a multiple of cacheM  = 64
    // N must be a multiple of cacheN  = 64
    // K must be a multiple of cacheK  = 32
    //
    // K/N also need to be compatible with float4 accesses.

    if (M % cacheM != 0 ||
        K % cacheK != 0 ||
        N % cacheN != 0)
    {
        std::cerr
            << "Skipping "
            << M << " x " << K << " x " << N
            << " (M must be a multiple of 64, "
            << "N must be a multiple of 64, "
            << "K must be a multiple of 32)\n";

        return;
    }

    const size_t sizeA =
        static_cast<size_t>(M) * K;

    const size_t sizeB =
        static_cast<size_t>(K) * N;

    const size_t sizeC =
        static_cast<size_t>(M) * N;

    std::vector<float> h_A(sizeA);
    std::vector<float> h_B(sizeB);
    std::vector<float> h_C(sizeC, 0.0f);
    std::vector<float> h_ref(sizeC, 0.0f);

    // --------------------------------------------------------
    // Random initialization
    // --------------------------------------------------------

    std::mt19937 rng(42);
    std::uniform_real_distribution<float> dist(-1.0f, 1.0f);

    for (float &x : h_A)
        x = dist(rng);

    for (float &x : h_B)
        x = dist(rng);

    // --------------------------------------------------------
    // CPU reference
    // --------------------------------------------------------

    cblas_sgemm(
        CblasRowMajor,
        CblasNoTrans,
        CblasNoTrans,
        M,
        N,
        K,
        1.0f,
        h_A.data(),
        K,
        h_B.data(),
        N,
        0.0f,
        h_ref.data(),
        N);

    // --------------------------------------------------------
    // GPU allocation
    // --------------------------------------------------------

    float *d_A = nullptr;
    float *d_B = nullptr;
    float *d_C = nullptr;

    cudaError_t err;

    err = cudaMalloc(
        &d_A,
        sizeA * sizeof(float));

    if (err != cudaSuccess)
    {
        std::cerr
            << "cudaMalloc(d_A) failed: "
            << cudaGetErrorString(err) << '\n';
        return;
    }

    err = cudaMalloc(
        &d_B,
        sizeB * sizeof(float));

    if (err != cudaSuccess)
    {
        std::cerr
            << "cudaMalloc(d_B) failed: "
            << cudaGetErrorString(err) << '\n';

        cudaFree(d_A);
        return;
    }

    err = cudaMalloc(
        &d_C,
        sizeC * sizeof(float));

    if (err != cudaSuccess)
    {
        std::cerr
            << "cudaMalloc(d_C) failed: "
            << cudaGetErrorString(err) << '\n';

        cudaFree(d_A);
        cudaFree(d_B);
        return;
    }

    // --------------------------------------------------------
    // Copy inputs
    // --------------------------------------------------------

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

    // --------------------------------------------------------
    // Kernel configuration
    //
    // One block computes one 64x64 C tile.
    // --------------------------------------------------------

    dim3 blockDim(THREADS_PER_BLOCK);

    dim3 gridDim(
        N / cacheN,
        M / cacheM);

    // --------------------------------------------------------
    // Warmup
    // --------------------------------------------------------

    wrapTiling<float, float>
        <<<gridDim, blockDim>>>(
            M,
            K,
            N,
            d_A,
            d_B,
            d_C);

    err = cudaDeviceSynchronize();

    if (err != cudaSuccess)
    {
        std::cerr
            << "Warmup kernel failed for "
            << M << " x " << K << " x " << N
            << ": "
            << cudaGetErrorString(err)
            << '\n';

        cudaFree(d_A);
        cudaFree(d_B);
        cudaFree(d_C);

        return;
    }

    // --------------------------------------------------------
    // CUDA events
    // --------------------------------------------------------

    cudaEvent_t start;
    cudaEvent_t stop;

    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    constexpr int iterations = 10;

    cudaEventRecord(start);

    for (int i = 0; i < iterations; i++)
    {
        wrapTiling<float, float>
            <<<gridDim, blockDim>>>(
                M,
                K,
                N,
                d_A,
                d_B,
                d_C);
    }

    cudaEventRecord(stop);
    cudaEventSynchronize(stop);

    float total_ms = 0.0f;

    cudaEventElapsedTime(
        &total_ms,
        start,
        stop);

    const double time_ms =
        static_cast<double>(total_ms) /
        iterations;

    // --------------------------------------------------------
    // Check kernel launch
    // --------------------------------------------------------

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

    // --------------------------------------------------------
    // Copy result back
    // --------------------------------------------------------

    cudaMemcpy(
        h_C.data(),
        d_C,
        sizeC * sizeof(float),
        cudaMemcpyDeviceToHost);

    // --------------------------------------------------------
    // Verify
    // --------------------------------------------------------

    float max_error = 0.0f;

    for (size_t i = 0; i < sizeC; i++)
    {
        max_error =
            std::max(
                max_error,
                std::abs(h_C[i] - h_ref[i]));
    }

    const bool correct =
        max_error < 1e-3f;

    // --------------------------------------------------------
    // GFLOP/s
    // --------------------------------------------------------

    const double flops =
        2.0 *
        static_cast<double>(M) *
        static_cast<double>(K) *
        static_cast<double>(N);

    const double gflops =
        flops /
        (time_ms * 1.0e6);

    // --------------------------------------------------------
    // Output
    // --------------------------------------------------------

    std::cout
        << std::left
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

    // --------------------------------------------------------
    // Cleanup
    // --------------------------------------------------------

    cudaEventDestroy(start);
    cudaEventDestroy(stop);

    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);
}

// ------------------------------------------------------------
// Main
// ------------------------------------------------------------

int main()
{
    std::cout
        << std::left
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
    benchmark(8192, 8192, 8192);

    return 0;
}