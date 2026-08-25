# CUDA GEMM Benchmarks & Optimizations

An iterative optimization study of **General Matrix Multiplication (GEMM)** in CUDA, comparing progressive CUDA kernel implementations against CPU (OpenBLAS verified) and NVIDIA **cuBLAS** performance.

---

## Overview

Matrix multiplication ($C = A \times B$) is the foundational computational kernel for deep learning and high-performance computing. This project benchmarks and analyzes key optimization techniques on GPUs:

1. **CPU Baseline** (`src/cpu.cpp`): Triple-nested loop CPU reference.
2. **Naive CUDA Kernel** (`src/naive.cu`): Baseline 2D grid implementation without coalescing.
3. **Coalesced Memory Access** (`src/naive_coalescing.cu`): Optimizing global memory access patterns to align with GPU warp read transactions.
4. **Shared Memory Tiling** (`src/smem_caching.cu`): Block-level caching in shared memory ($16 \times 16$ tile size) to drastically reduce global memory bandwidth bottleneck.
5. **NVIDIA cuBLAS** (`src/cublas.cu`): Production-grade vendor library performance benchmark (`cublasSgemm`).

All implementations compute $C = A \times B$ for square single-precision floating-point matrices ($M = K = N$) and automatically validate their output against OpenBLAS `cblas_sgemm` to verify floating-point precision correctness.

---

## Performance Summary

Throughput is measured in **GFLOP/s** ($\text{Floating Point Operations per Second}$), calculated as:

$$\text{GFLOP/s} = \frac{2 \times M \times K \times N}{\text{Time (seconds)} \times 10^9}$$

### Hardware & Benchmark Environment

All benchmarks were recorded on the following system:

- **GPU**: NVIDIA GeForce RTX 5080 Laptop / Mobile (Max-Q)
- **VRAM**: 16 GB
- **CPU**: Intel® Core™ Ultra 9 275HX (24 cores)
- **OS & Kernel**: Ubuntu 26.04 LTS x86_64 (Linux kernel 7.0.0-29-generic)
- **Memory**: 32 GB RAM

---

### Comparative Benchmark Results

| Matrix Size ($M \times K \times N$) | CPU (OpenBLAS Ref) | Naive CUDA | Naive Coalesced CUDA | Shared Memory (Smem) | NVIDIA cuBLAS |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **$128 \times 128 \times 128$** | 1.13 GFLOP/s *(3.72 ms)* | 269.97 GFLOP/s *(0.016 ms)* | 891.04 GFLOP/s *(0.0047 ms)* | **1027.21 GFLOP/s** *(0.0041 ms)* | 635.96 GFLOP/s *(0.0066 ms)* |
| **$256 \times 256 \times 256$** | 2.92 GFLOP/s *(11.49 ms)* | 522.36 GFLOP/s *(0.064 ms)* | 1164.05 GFLOP/s *(0.0288 ms)* | **2197.35 GFLOP/s** *(0.0153 ms)* | **3921.38 GFLOP/s** *(0.0086 ms)* |
| **$512 \times 512 \times 512$** | 2.34 GFLOP/s *(114.90 ms)* | 601.35 GFLOP/s *(0.446 ms)* | 1691.49 GFLOP/s *(0.1587 ms)* | **3150.30 GFLOP/s** *(0.0852 ms)* | **9747.40 GFLOP/s** *(0.0275 ms)* |
| **$1024 \times 1024 \times 1024$** | 1.36 GFLOP/s *(1582.75 ms)* | 630.45 GFLOP/s *(3.406 ms)* | 1593.81 GFLOP/s *(1.347 ms)* | **3425.77 GFLOP/s** *(0.6269 ms)* | **19375.47 GFLOP/s** *(0.1108 ms)* |
| **$2048 \times 2048 \times 2048$** | 0.73 GFLOP/s *(23483.45 ms)* | 637.28 GFLOP/s *(26.96 ms)* | 1369.91 GFLOP/s *(12.54 ms)* | **3482.72 GFLOP/s** *(4.933 ms)* | **25001.50 GFLOP/s** *(0.687 ms)* |
| **$4096 \times 4096 \times 4096$** | 0.35 GFLOP/s *(395621.22 ms)* | 628.64 GFLOP/s *(218.63 ms)* | 878.98 GFLOP/s *(156.36 ms)* | **3206.16 GFLOP/s** *(42.87 ms)* | **23706.17 GFLOP/s** *(5.798 ms)* |

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
| **Shared Memory (Smem) Tiled** | 4.933 ms | 3482.72 GFLOP/s | 13.93% |
| **NVIDIA cuBLAS** | 0.687 ms | 25001.50 GFLOP/s | **100.00%** |

1. **CPU vs GPU Baseline**: Even the naive uncoalesced CUDA kernel achieves a **~870x speedup** over single-threaded C++ GEMM at $2048 \times 2048$ matrix size due to massive parallel execution across GPU cores.
2. **Coalesced Memory Access**: Reordering memory indexing to align contiguous thread accesses within a warp yields a **~2.1x speedup** over naive indexing at $2048 \times 2048$, resolving unaligned memory transactions.
3. **Shared Memory Tiling**: By loading $16 \times 16$ matrix tiles into high-speed `__shared__` memory and synchronizing threads (`__syncthreads()`), global memory traffic is reduced by a factor of 16. This increases throughput from **1369.91 GFLOP/s** (Coalesced) to **3482.72 GFLOP/s** (Shared Memory)—a **2.54x improvement**.
4. **cuBLAS Peak Performance**: cuBLAS reaches **25 TFLOP/s** throughput at $2048 \times 2048$. It achieves this level by leveraging hardware-level features such as register tiling (2D thread tiling), vectorized memory operations (`float4` / `LDG.128`), warp shuffle instructions, and hardware Tensor Cores.

---

## Project Structure

```
cuda-gemm/
├── benchmarks/                        # Pre-computed benchmark execution logs & plots
│   ├── cpu_benchmark.txt
│   ├── cublas_benchmark.txt
│   ├── gflops_vs_matrix_size.png      # Generated benchmark chart
│   ├── naive_benchmark.txt
│   ├── naive_coalescing_benchmark.txt
│   └── smem_benchmark.txt
├── src/                               # Source implementations
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
- [`benchmarks/cublas_benchmark.txt`](./benchmarks/cublas_benchmark.txt)

---

## Acknowledgments

- Concepts and optimization steps were inspired by Simon Boehm's excellent guide: [How to Optimize a CUDA Matmul Kernel for cuBLAS-like Performance](https://siboehm.com/articles/22/CUDA-MMM). All implementations in this repository were written from scratch.
