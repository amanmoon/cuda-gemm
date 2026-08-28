# CUDA GEMM Benchmarks & Optimizations

An iterative optimization study of **General Matrix Multiplication (GEMM)** in CUDA, comparing progressive CUDA kernel implementations against CPU (OpenBLAS verified) and NVIDIA **cuBLAS** performance.

---

## Overview

Matrix multiplication ($C = A \times B$) is the foundational computational kernel for deep learning and high-performance computing. This project benchmarks and analyzes key optimization techniques on GPUs:

1. **CPU Baseline** (`src/cpu.cpp`): Triple-nested loop CPU reference.
2. **Naive CUDA Kernel** (`src/naive.cu`): Baseline 2D grid implementation without coalescing.
3. **Coalesced Memory Access** (`src/naive_coalescing.cu`): Optimizing global memory access patterns to align with GPU warp read transactions.
4. **Shared Memory Tiling** (`src/smem_caching.cu`): Block-level caching in shared memory ($16 \times 16$ tile size) to drastically reduce global memory bandwidth bottleneck.
5. **1D Block Tiling (Naive)** (`src/1d_blocktiling_naive.cu`): Thread-level 1D register tiling ($4 \times 1$ per thread work allocation) with shared memory caching, but uncoalesced global memory loads.
6. **1D Block Tiling (Coalesced)** (`src/1d_blocktiling_coalescing.cu`): Combining 1D thread block tiling with coalesced global memory transfers to drastically improve arithmetic intensity and bandwidth efficiency.
7. **2D Block Tiling** (`src/2d_blocktiling.cu`): 2D register tiling ($4 \times 4$ output tile per thread) paired with shared memory caching ($64 \times 64$ block tiles) to maximize register reuse and memory bandwidth utilization.
8. **NVIDIA cuBLAS** (`src/cublas.cu`): Production-grade vendor library performance benchmark (`cublasSgemm`).

All implementations compute $C = A \times B$ for square single-precision floating-point matrices ($M = K = N$) and automatically validate their output against OpenBLAS `cblas_sgemm` to verify floating-point precision correctness.

---

## Performance Summary

Throughput is measured in **GFLOP/s** ($\text{Floating Point Operations per Second}$), calculated as:

$$\text{GFLOP/s} = \frac{2 \times M \times K \times N}{\text{Time (seconds)} \times 10^9}$$

### Hardware & Benchmark Environment

All benchmarks were recorded on the following system:

- **GPU**: NVIDIA GeForce RTX 5080 Laptop / Mobile (Max-Q)
- **VRAM**: 16 GB
- **FP32 Theoretical Limits**: 27.6 TFLOP/s (1800 MHz) | 35.1 TFLOP/s (2285 MHz max boost)
- **CPU**: Intel® Core™ Ultra 9 275HX (24 cores)
- **OS & Kernel**: Ubuntu 26.04 LTS x86_64 (Linux kernel 7.0.0-29-generic)
- **Memory**: 32 GB RAM

---

### Comparative Benchmark Results

| Matrix Size ($M \times K \times N$) | CPU<br>(OpenBLAS Ref) | Naive<br>CUDA | Naive Coalesced<br>CUDA | Shared Memory<br>(2D Tile) | 1D Blocktiling<br>(Naive) | 1D Blocktiling<br>(Coalesced) | 2D Blocktiling | NVIDIA<br>cuBLAS |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **$128 \times 128 \times 128$** | 1.13&nbsp;GFLOP/s<br>*(3.72 ms)* | 269.97&nbsp;GFLOP/s<br>*(0.016 ms)* | 891.04&nbsp;GFLOP/s<br>*(0.0047 ms)* | **1027.21&nbsp;GFLOP/s**<br>*(0.0041 ms)* | 132.46&nbsp;GFLOP/s<br>*(0.0317 ms)* | 346.02&nbsp;GFLOP/s<br>*(0.0121 ms)* | 580.74&nbsp;GFLOP/s<br>*(0.0072 ms)* | 635.96&nbsp;GFLOP/s<br>*(0.0066 ms)* |
| **$256 \times 256 \times 256$** | 2.92&nbsp;GFLOP/s<br>*(11.49 ms)* | 522.36&nbsp;GFLOP/s<br>*(0.064 ms)* | 1164.05&nbsp;GFLOP/s<br>*(0.0288 ms)* | 2197.35&nbsp;GFLOP/s<br>*(0.0153 ms)* | 543.92&nbsp;GFLOP/s<br>*(0.0617 ms)* | 1451.72&nbsp;GFLOP/s<br>*(0.0231 ms)* | **2509.16&nbsp;GFLOP/s**<br>*(0.0134 ms)* | **3921.38&nbsp;GFLOP/s**<br>*(0.0086 ms)* |
| **$512 \times 512 \times 512$** | 2.34&nbsp;GFLOP/s<br>*(114.90 ms)* | 601.35&nbsp;GFLOP/s<br>*(0.446 ms)* | 1691.49&nbsp;GFLOP/s<br>*(0.1587 ms)* | 3150.30&nbsp;GFLOP/s<br>*(0.0852 ms)* | 1253.23&nbsp;GFLOP/s<br>*(0.2142 ms)* | 5253.39&nbsp;GFLOP/s<br>*(0.0511 ms)* | **7884.77&nbsp;GFLOP/s**<br>*(0.0340 ms)* | **9747.40&nbsp;GFLOP/s**<br>*(0.0275 ms)* |
| **$1024 \times 1024 \times 1024$** | 1.36&nbsp;GFLOP/s<br>*(1582.75 ms)* | 630.45&nbsp;GFLOP/s<br>*(3.406 ms)* | 1593.81&nbsp;GFLOP/s<br>*(1.347 ms)* | 3425.77&nbsp;GFLOP/s<br>*(0.6269 ms)* | 2073.89&nbsp;GFLOP/s<br>*(1.0355 ms)* | 9204.09&nbsp;GFLOP/s<br>*(0.2333 ms)* | **14162.17&nbsp;GFLOP/s**<br>*(0.1516 ms)* | **19375.47&nbsp;GFLOP/s**<br>*(0.1108 ms)* |
| **$2048 \times 2048 \times 2048$** | 0.73&nbsp;GFLOP/s<br>*(23483.45 ms)* | 637.28&nbsp;GFLOP/s<br>*(26.96 ms)* | 1369.91&nbsp;GFLOP/s<br>*(12.54 ms)* | 3482.72&nbsp;GFLOP/s<br>*(4.933 ms)* | 2347.49&nbsp;GFLOP/s<br>*(7.318 ms)* | 11094.55&nbsp;GFLOP/s<br>*(1.548 ms)* | **16109.67&nbsp;GFLOP/s**<br>*(1.066 ms)* | **25001.50&nbsp;GFLOP/s**<br>*(0.687 ms)* |
| **$4096 \times 4096 \times 4096$** | 0.35&nbsp;GFLOP/s<br>*(395621.22 ms)* | 628.64&nbsp;GFLOP/s<br>*(218.63 ms)* | 878.98&nbsp;GFLOP/s<br>*(156.36 ms)* | 3206.16&nbsp;GFLOP/s<br>*(42.87 ms)* | 2445.39&nbsp;GFLOP/s<br>*(56.20 ms)* | 10805.50&nbsp;GFLOP/s<br>*(12.72 ms)* | **14379.79&nbsp;GFLOP/s**<br>*(9.558 ms)* | **23706.17&nbsp;GFLOP/s**<br>*(5.798 ms)* |

> All benchmarks report **PASS** with maximum absolute numerical error bounded under $3.5 \times 10^{-4}$ against OpenBLAS reference outputs.

### GFLOP/s vs. Matrix Size Chart

![CUDA GEMM Performance (GFLOP/s vs Matrix Size)](./benchmarks/gflops_vs_matrix_size.png)

---

## Performance Analysis & Insights

### $2048 \times 2048 \times 2048$ Matrix Performance Breakdown

| Implementation | Execution Time | Throughput | Performance vs. cuBLAS (%) |
| :--- | :---: | :---: | :---: |
| **CPU (OpenBLAS Ref)** | 23483.45 ms | 0.73 GFLOP/s | 0.003% |
| **Naive CUDA** | 26.96 ms | 637.28 GFLOP/s | 2.55% |
| **Naive Coalesced CUDA** | 12.54 ms | 1369.91 GFLOP/s | 5.48% |
| **1D Block Tiling (Naive)** | 7.318 ms | 2347.49 GFLOP/s | 9.39% |
| **Shared Memory (Smem) Tiled** | 4.933 ms | 3482.72 GFLOP/s | 13.93% |
| **1D Block Tiling (Coalesced)** | 1.548 ms | 11094.55 GFLOP/s | 44.38% |
| **2D Block Tiling** | 1.066 ms | 16109.67 GFLOP/s | **64.43%** |
| **NVIDIA cuBLAS** | 0.687 ms | 25001.50 GFLOP/s | **100.00%** |

1. **CPU vs GPU Baseline**: Even the naive uncoalesced CUDA kernel achieves a **~870x speedup** over single-threaded C++ GEMM at $2048 \times 2048$ matrix size due to massive parallel execution across GPU cores.
2. **Coalesced Memory Access**: Reordering memory indexing to align contiguous thread accesses within a warp yields a **~2.1x speedup** over naive indexing at $2048 \times 2048$, resolving unaligned memory transactions.
3. **Shared Memory Tiling (2D Tile)**: By loading $16 \times 16$ matrix tiles into high-speed `__shared__` memory and synchronizing threads (`__syncthreads()`), global memory traffic is reduced by a factor of 16. This increases throughput from **1369.91 GFLOP/s** (Coalesced) to **3482.72 GFLOP/s** (Shared Memory)—a **2.54x improvement**.
4. **1D Block Tiling & Register Reuse**: Assigning multiple output elements ($TM = 16$) to each thread's register space significantly improves arithmetic intensity. When paired with coalesced global memory loads (`1d_blocktiling_coalescing.cu`), performance reaches **11094.55 GFLOP/s** (11.1 TFLOP/s)—a **3.19x speedup** over 2D Shared Memory Tiling and reaching **44.38% of cuBLAS performance**. Without coalesced loading (`1d_blocktiling_naive.cu`), unaligned memory accesses constrain performance to **2347.49 GFLOP/s**, highlighting that aligned memory transfers remain critical even with register caching.
5. **2D Block Tiling (2D Thread/Register Tiling)**: Extending register caching into two dimensions ($4 \times 4$ per-thread tile) alongside 2D shared memory caching (`src/2d_blocktiling.cu`) reuses cached values across both rows and columns. This significantly increases arithmetic intensity and reduces total memory access overhead, boosting performance to **16109.67 GFLOP/s** (16.11 TFLOP/s) at $2048 \times 2048$—a **1.45x speedup** over 1D Coalesced Blocktiling and achieving **64.43% of cuBLAS performance**.
6. **cuBLAS Peak Performance**: cuBLAS reaches **25 TFLOP/s** throughput at $2048 \times 2048$. It achieves this level by leveraging hardware-level features such as 2D thread tiling (register tiling in both dimensions), vectorized memory operations (`float4` / `LDG.128`), warp shuffle instructions, double buffering/pipelining, and hardware Tensor Cores.

---

## Project Structure

```
cuda-gemm/
├── benchmarks/                        # Pre-computed benchmark execution logs & plots
│   ├── 1d_blocktiling_benchmark       # Benchmark log for 1D Blocktiling implementations
│   ├── 2d_blocktiling_benchmark       # Benchmark log for 2D Blocktiling implementation
│   ├── cpu_benchmark.txt
│   ├── cublas_benchmark.txt
│   ├── gflops_vs_matrix_size.png      # Generated benchmark chart
│   ├── naive_benchmark.txt
│   ├── naive_coalescing_benchmark.txt
│   └── smem_benchmark.txt
├── src/                               # Source implementations
│   ├── 1d_blocktiling_coalescing.cu   # 1D blocktiling CUDA kernel with coalesced loads
│   ├── 1d_blocktiling_naive.cu        # 1D blocktiling CUDA kernel (uncoalesced)
│   ├── 2d_blocktiling.cu              # 2D blocktiling CUDA kernel (4x4 thread tiling)
│   ├── cpu.cpp                        # CPU OpenBLAS reference & GEMM baseline
│   ├── cublas.cu                      # cuBLAS benchmark
│   ├── naive.cu                       # Uncoalesced CUDA kernel
│   ├── naive_coalescing.cu            # Memory-coalesced CUDA kernel
│   └── smem_caching.cu                # Tiled Shared Memory CUDA kernel
└── README.md                          # Project documentation
```

---

## Detailed Benchmark Reports

Individual execution logs are stored in the [`benchmarks/`](./benchmarks/) folder:

- [`benchmarks/cpu_benchmark.txt`](./benchmarks/cpu_benchmark.txt)
- [`benchmarks/naive_benchmark.txt`](./benchmarks/naive_benchmark.txt)
- [`benchmarks/naive_coalescing_benchmark.txt`](./benchmarks/naive_coalescing_benchmark.txt)
- [`benchmarks/smem_benchmark.txt`](./benchmarks/smem_benchmark.txt)
- [`benchmarks/1d_blocktiling_benchmark`](./benchmarks/1d_blocktiling_benchmark)
- [`benchmarks/2d_blocktiling_benchmark`](./benchmarks/2d_blocktiling_benchmark)
- [`benchmarks/cublas_benchmark.txt`](./benchmarks/cublas_benchmark.txt)

---

## Acknowledgments

- Concepts and optimization steps were inspired by Simon Boehm's excellent guide: [How to Optimize a CUDA Matmul Kernel for cuBLAS-like Performance](https://siboehm.com/articles/22/CUDA-MMM). All implementations in this repository were written from scratch.
