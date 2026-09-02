import os
import re
import matplotlib.pyplot as plt
import numpy as np

# Matrix sizes
sizes = np.array([128, 256, 512, 1024, 2048, 4096, 8192])


def parse_benchmark_file(filepath):
    """Parses a benchmark text file and extracts matrix size -> GFLOP/s mapping."""
    data = {}
    if not os.path.exists(filepath):
        print(f"Warning: File not found: {filepath}")
        return data

    pattern = re.compile(r"(\d+)\s*x\s*\d+\s*x\s*\d+\s+[\d\.]+\s*ms\s+([\d\.]+)\s*GFLOP/s")

    with open(filepath, 'r') as f:
        for line in f:
            match = pattern.search(line)
            if match:
                size = int(match.group(1))
                gflops = float(match.group(2))
                data[size] = gflops
    return data


def parse_1d_blocktiling(filepath):
    """Parses 1d_blocktiling_benchmark.txt containing 'Without coalescing' and 'With coalescing' sections."""
    naive_data = {}
    coalesced_data = {}
    if not os.path.exists(filepath):
        print(f"Warning: File not found: {filepath}")
        return naive_data, coalesced_data

    pattern = re.compile(r"(\d+)\s*x\s*\d+\s*x\s*\d+\s+[\d\.]+\s*ms\s+([\d\.]+)\s*GFLOP/s")

    current_section = "without"
    with open(filepath, 'r') as f:
        for line in f:
            if "With coalescing" in line:
                current_section = "with"
            match = pattern.search(line)
            if match:
                size = int(match.group(1))
                gflops = float(match.group(2))
                if current_section == "without":
                    naive_data[size] = gflops
                else:
                    coalesced_data[size] = gflops
    return naive_data, coalesced_data


def load_benchmark_data(benchmarks_dir):
    """Loads all benchmark data from the benchmarks directory."""
    cpu_data = parse_benchmark_file(os.path.join(benchmarks_dir, 'cpu_benchmark.txt'))
    naive_data = parse_benchmark_file(os.path.join(benchmarks_dir, 'naive_benchmark.txt'))
    naive_coalesced_data = parse_benchmark_file(os.path.join(benchmarks_dir, 'naive_coalescing_benchmark.txt'))
    smem_data = parse_benchmark_file(os.path.join(benchmarks_dir, 'smem_benchmark.txt'))
    b1d_naive_data, b1d_coalesced_data = parse_1d_blocktiling(os.path.join(benchmarks_dir, '1d_blocktiling_benchmark.txt'))
    b2d_data = parse_benchmark_file(os.path.join(benchmarks_dir, '2d_blocktiling_benchmark.txt'))
    vectorized_smem_data = parse_benchmark_file(os.path.join(benchmarks_dir, 'vectorized_smem_loading_benchmark.txt'))
    wraptiling_data = parse_benchmark_file(os.path.join(benchmarks_dir, 'wraptiling_benchmark.txt'))
    wraptiling_float4_data = parse_benchmark_file(os.path.join(benchmarks_dir, 'wraptiling_float4_load_benchmark.txt'))
    cublas_data = parse_benchmark_file(os.path.join(benchmarks_dir, 'cublas_benchmark.txt'))

    def to_list(data_map):
        return [data_map.get(s, None) for s in sizes]

    return {
        'cpu': to_list(cpu_data),
        'naive': to_list(naive_data),
        'naive_coalesced': to_list(naive_coalesced_data),
        'smem': to_list(smem_data),
        'b1d_naive': to_list(b1d_naive_data),
        'b1d_coalesced': to_list(b1d_coalesced_data),
        'b2d': to_list(b2d_data),
        'vectorized_smem': to_list(vectorized_smem_data),
        'wraptiling': to_list(wraptiling_data),
        'wraptiling_float4': to_list(wraptiling_float4_data),
        'cublas': to_list(cublas_data),
    }


colors = {
    'CPU': '#64748b',
    'Naive CUDA': '#dc2626',
    'Naive Coalesced': '#ea580c',
    '1D Blocktiling (Naive)': '#d97706',
    'Shared Memory Tiling': '#65a30d',
    '1D Blocktiling (Coalesced)': '#0891b2',
    '2D Blocktiling': '#2563eb',
    'Vectorized Smem Loading': '#9333ea',
    'Warp Tiling': '#d946ef',
    'Warp Tiling (float4 Load)': '#0284c7',
    'NVIDIA cuBLAS': '#059669'
}


def create_plot(data, scaled_x=True, output_filename=''):
    plt.style.use('default')
    fig, ax = plt.subplots(figsize=(13, 7.5), dpi=300)

    fig.patch.set_facecolor('#ffffff')
    ax.set_facecolor('#ffffff')

    x = sizes if scaled_x else np.arange(len(sizes))

    cpu_x = [x_val for x_val, v in zip(x, data['cpu']) if v is not None]
    cpu_y = [v for v in data['cpu'] if v is not None]
    ax.plot(cpu_x, cpu_y, label='CPU (OpenBLAS Ref)', color=colors['CPU'], marker='o', linewidth=1.5, linestyle=':')

    ax.plot(x, data['naive'], label='Naive CUDA', color=colors['Naive CUDA'], marker='s', linewidth=1.8)
    ax.plot(x, data['naive_coalesced'], label='Naive Coalesced CUDA', color=colors['Naive Coalesced'], marker='^', linewidth=1.8)
    ax.plot(x, data['b1d_naive'], label='1D Blocktiling (Naive)', color=colors['1D Blocktiling (Naive)'], marker='v', linewidth=1.8)
    ax.plot(x, data['smem'], label='Shared Memory Tiling', color=colors['Shared Memory Tiling'], marker='d', linewidth=1.8)
    ax.plot(x, data['b1d_coalesced'], label='1D Blocktiling (Coalesced)', color=colors['1D Blocktiling (Coalesced)'], marker='p', linewidth=2.0)
    ax.plot(x, data['b2d'], label='2D Blocktiling', color=colors['2D Blocktiling'], marker='h', linewidth=2.2)
    ax.plot(x, data['vectorized_smem'], label='Vectorized Smem Loading', color=colors['Vectorized Smem Loading'], marker='D', linewidth=2.2)
    ax.plot(x, data['wraptiling'], label='Warp Tiling', color=colors['Warp Tiling'], marker='X', linewidth=2.5)
    ax.plot(x, data['wraptiling_float4'], label='Warp Tiling (float4 Load)', color=colors['Warp Tiling (float4 Load)'], marker='P', linewidth=2.5)
    ax.plot(x, data['cublas'], label='NVIDIA cuBLAS', color=colors['NVIDIA cuBLAS'], marker='*', linewidth=2.5, linestyle='--')

    # Boost Clock Speed line (32 TFLOP/s = 32000 GFLOP/s)
    boost_peak_gflops = 32000
    ax.axhline(y=boost_peak_gflops, color='#e11d48', linestyle='--', alpha=0.75, linewidth=1.8,
               label='32 Tflops/s (FP32 Theoretical Peak)')

    if scaled_x:
        text_x = sizes[-1] - 2500
    else:
        text_x = len(sizes) - 2.6

    ax.text(x=text_x, y=boost_peak_gflops + 500, s='32 Tflops/s (FP32 Theoretical peak)',
            color='#e11d48', fontsize=10, fontweight='bold', ha='left', va='bottom')

    ax.set_title('CUDA GEMM Performance: GFLOP/s vs Matrix Size (M=N=K)', fontsize=15, fontweight='bold', pad=15, color='#0f172a')
    ax.set_xlabel('Matrix Size (N x N)', fontsize=12, fontweight='semibold', color='#1e293b', labelpad=10)
    ax.set_ylabel('Throughput (GFLOP/s)', fontsize=12, fontweight='semibold', color='#1e293b', labelpad=10)

    ax.set_xticks(x)
    ax.set_xticklabels(sizes, fontsize=10, color='#334155')
    ax.tick_params(colors='#334155', labelsize=10)

    ax.grid(True, which='both', linestyle='--', alpha=0.3, color='#94a3b8')
    ax.set_ylim(0, 36000)

    ax.legend(loc='upper left', facecolor='#ffffff', edgecolor='#cbd5e1', fontsize=8.5, labelcolor='#0f172a', framealpha=0.95)

    plt.tight_layout()
    plt.savefig(output_filename, dpi=300)
    plt.close()
    print(f"Generated plot: {output_filename}")


if __name__ == '__main__':
    script_dir = os.path.dirname(os.path.abspath(__file__))
    benchmarks_dir = os.path.join(script_dir, 'benchmarks')
    data = load_benchmark_data(benchmarks_dir)

    create_plot(data, scaled_x=False, output_filename=os.path.join(benchmarks_dir, 'gflops_vs_matrix_size.png'))
    create_plot(data, scaled_x=True, output_filename=os.path.join(benchmarks_dir, 'gflops_vs_matrix_size_scaled.png'))

