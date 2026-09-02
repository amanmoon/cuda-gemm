import os
import re
import matplotlib.pyplot as plt
import numpy as np

# Matrix sizes
sizes = np.array([128, 256, 512, 1024, 1536, 2048, 3072, 4096, 5120, 6144, 7168, 8192])


def parse_benchmark_file(filepath):
    """Parses a benchmark text file and extracts matrix size -> (gflops, time_ms) mapping."""
    gflops_data = {}
    time_data = {}
    if not os.path.exists(filepath):
        print(f"Warning: File not found: {filepath}")
        return gflops_data, time_data

    pattern = re.compile(r"(\d+)\s*x\s*\d+\s*x\s*\d+\s+([\d\.]+)\s*ms\s+([\d\.]+)\s*GFLOP/s")

    with open(filepath, 'r') as f:
        for line in f:
            match = pattern.search(line)
            if match:
                size = int(match.group(1))
                time_ms = float(match.group(2))
                gflops = float(match.group(3))
                gflops_data[size] = gflops
                time_data[size] = time_ms
    return gflops_data, time_data


def parse_1d_blocktiling(filepath):
    """Parses 1d_blocktiling_benchmark.txt containing 'Without coalescing' and 'With coalescing' sections."""
    naive_gflops, naive_time = {}, {}
    coalesced_gflops, coalesced_time = {}, {}
    if not os.path.exists(filepath):
        print(f"Warning: File not found: {filepath}")
        return naive_gflops, naive_time, coalesced_gflops, coalesced_time

    pattern = re.compile(r"(\d+)\s*x\s*\d+\s*x\s*\d+\s+([\d\.]+)\s*ms\s+([\d\.]+)\s*GFLOP/s")

    current_section = "without"
    with open(filepath, 'r') as f:
        for line in f:
            if "With coalescing" in line:
                current_section = "with"
            match = pattern.search(line)
            if match:
                size = int(match.group(1))
                time_ms = float(match.group(2))
                gflops = float(match.group(3))
                if current_section == "without":
                    naive_gflops[size] = gflops
                    naive_time[size] = time_ms
                else:
                    coalesced_gflops[size] = gflops
                    coalesced_time[size] = time_ms
    return naive_gflops, naive_time, coalesced_gflops, coalesced_time


def load_benchmark_data(benchmarks_dir):
    """Loads all benchmark data from the benchmarks directory."""
    cpu_g, cpu_t = parse_benchmark_file(os.path.join(benchmarks_dir, 'cpu_benchmark.txt'))
    naive_g, naive_t = parse_benchmark_file(os.path.join(benchmarks_dir, 'naive_benchmark.txt'))
    naive_c_g, naive_c_t = parse_benchmark_file(os.path.join(benchmarks_dir, 'naive_coalescing_benchmark.txt'))
    smem_g, smem_t = parse_benchmark_file(os.path.join(benchmarks_dir, 'smem_benchmark.txt'))
    b1d_n_g, b1d_n_t, b1d_c_g, b1d_c_t = parse_1d_blocktiling(os.path.join(benchmarks_dir, '1d_blocktiling_benchmark.txt'))
    b2d_g, b2d_t = parse_benchmark_file(os.path.join(benchmarks_dir, '2d_blocktiling_benchmark.txt'))
    vec_smem_g, vec_smem_t = parse_benchmark_file(os.path.join(benchmarks_dir, 'vectorized_smem_loading_benchmark.txt'))
    wrap_g, wrap_t = parse_benchmark_file(os.path.join(benchmarks_dir, 'wraptiling_benchmark.txt'))
    wrap_f4_g, wrap_f4_t = parse_benchmark_file(os.path.join(benchmarks_dir, 'wraptiling_float4_load_benchmark.txt'))
    cublas_g, cublas_t = parse_benchmark_file(os.path.join(benchmarks_dir, 'cublas_benchmark.txt'))

    def to_list(data_map):
        return [data_map.get(s, None) for s in sizes]

    gflops_dict = {
        'cpu': to_list(cpu_g),
        'naive': to_list(naive_g),
        'naive_coalesced': to_list(naive_c_g),
        'smem': to_list(smem_g),
        'b1d_naive': to_list(b1d_n_g),
        'b1d_coalesced': to_list(b1d_c_g),
        'b2d': to_list(b2d_g),
        'vectorized_smem': to_list(vec_smem_g),
        'wraptiling': to_list(wrap_g),
        'wraptiling_float4': to_list(wrap_f4_g),
        'cublas': to_list(cublas_g),
    }

    time_dict = {
        'cpu': to_list(cpu_t),
        'naive': to_list(naive_t),
        'naive_coalesced': to_list(naive_c_t),
        'smem': to_list(smem_t),
        'b1d_naive': to_list(b1d_n_t),
        'b1d_coalesced': to_list(b1d_c_t),
        'b2d': to_list(b2d_t),
        'vectorized_smem': to_list(vec_smem_t),
        'wraptiling': to_list(wrap_t),
        'wraptiling_float4': to_list(wrap_f4_t),
        'cublas': to_list(cublas_t),
    }

    return gflops_dict, time_dict


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


def create_gflops_plot(data, scaled_x=True, output_filename=''):
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
        text_x = len(sizes) - 3.2

    ax.text(x=text_x, y=boost_peak_gflops + 500, s='32 Tflops/s (FP32 Theoretical peak)',
            color='#e11d48', fontsize=10, fontweight='bold', ha='left', va='bottom')

    ax.set_title('CUDA GEMM Performance: GFLOP/s vs Matrix Size (M=N=K)', fontsize=15, fontweight='bold', pad=15, color='#0f172a')
    ax.set_xlabel('Matrix Size (N x N)', fontsize=12, fontweight='semibold', color='#1e293b', labelpad=10)
    ax.set_ylabel('Throughput (GFLOP/s)', fontsize=12, fontweight='semibold', color='#1e293b', labelpad=10)

    ax.set_xticks(x)
    ax.set_xticklabels(sizes, fontsize=9.5, color='#334155')
    ax.tick_params(colors='#334155', labelsize=10)

    ax.grid(True, which='both', linestyle='--', alpha=0.3, color='#94a3b8')
    ax.set_ylim(0, 36000)

    ax.legend(loc='upper left', facecolor='#ffffff', edgecolor='#cbd5e1', fontsize=8.5, labelcolor='#0f172a', framealpha=0.95)

    plt.tight_layout()
    plt.savefig(output_filename, dpi=300)
    plt.close()
    print(f"Generated plot: {output_filename}")


def create_execution_time_plot(time_data, output_filename=''):
    plt.style.use('default')
    fig, ax = plt.subplots(figsize=(13, 7.5), dpi=300)

    fig.patch.set_facecolor('#ffffff')
    ax.set_facecolor('#ffffff')

    x = np.arange(len(sizes))

    cpu_x = [x_val for x_val, v in zip(x, time_data['cpu']) if v is not None]
    cpu_y = [v for v in time_data['cpu'] if v is not None]
    ax.plot(cpu_x, cpu_y, label='CPU (OpenBLAS Ref)', color=colors['CPU'], marker='o', linewidth=1.5, linestyle=':')

    ax.plot(x, time_data['naive'], label='Naive CUDA', color=colors['Naive CUDA'], marker='s', linewidth=1.8)
    ax.plot(x, time_data['naive_coalesced'], label='Naive Coalesced CUDA', color=colors['Naive Coalesced'], marker='^', linewidth=1.8)
    ax.plot(x, time_data['b1d_naive'], label='1D Blocktiling (Naive)', color=colors['1D Blocktiling (Naive)'], marker='v', linewidth=1.8)
    ax.plot(x, time_data['smem'], label='Shared Memory Tiling', color=colors['Shared Memory Tiling'], marker='d', linewidth=1.8)
    ax.plot(x, time_data['b1d_coalesced'], label='1D Blocktiling (Coalesced)', color=colors['1D Blocktiling (Coalesced)'], marker='p', linewidth=2.0)
    ax.plot(x, time_data['b2d'], label='2D Blocktiling', color=colors['2D Blocktiling'], marker='h', linewidth=2.2)
    ax.plot(x, time_data['vectorized_smem'], label='Vectorized Smem Loading', color=colors['Vectorized Smem Loading'], marker='D', linewidth=2.2)
    ax.plot(x, time_data['wraptiling'], label='Warp Tiling', color=colors['Warp Tiling'], marker='X', linewidth=2.5)
    ax.plot(x, time_data['wraptiling_float4'], label='Warp Tiling (float4 Load)', color=colors['Warp Tiling (float4 Load)'], marker='P', linewidth=2.5)
    ax.plot(x, time_data['cublas'], label='NVIDIA cuBLAS', color=colors['NVIDIA cuBLAS'], marker='*', linewidth=2.5, linestyle='--')

    ax.set_yscale('log')
    ax.set_title('CUDA GEMM Execution Time: Runtime (ms) vs Matrix Size (Log Scale)', fontsize=15, fontweight='bold', pad=15, color='#0f172a')
    ax.set_xlabel('Matrix Size (N x N)', fontsize=12, fontweight='semibold', color='#1e293b', labelpad=10)
    ax.set_ylabel('Execution Time (ms) [Log Scale]', fontsize=12, fontweight='semibold', color='#1e293b', labelpad=10)

    ax.set_xticks(x)
    ax.set_xticklabels(sizes, fontsize=9.5, color='#334155')
    ax.tick_params(colors='#334155', labelsize=10)

    ax.grid(True, which='both', linestyle='--', alpha=0.3, color='#94a3b8')

    ax.legend(loc='upper left', facecolor='#ffffff', edgecolor='#cbd5e1', fontsize=8.5, labelcolor='#0f172a', framealpha=0.95)

    plt.tight_layout()
    plt.savefig(output_filename, dpi=300)
    plt.close()
    print(f"Generated plot: {output_filename}")


def create_cublas_percentage_plot(gflops_data, output_filename=''):
    plt.style.use('default')
    fig, ax = plt.subplots(figsize=(13, 7.5), dpi=300)

    fig.patch.set_facecolor('#ffffff')
    ax.set_facecolor('#ffffff')

    x = np.arange(len(sizes))
    cublas = np.array(gflops_data['cublas'])

    pct_keys = [
        ('naive', 'Naive CUDA', colors['Naive CUDA'], 's', 1.8),
        ('naive_coalesced', 'Naive Coalesced CUDA', colors['Naive Coalesced'], '^', 1.8),
        ('b1d_naive', '1D Blocktiling (Naive)', colors['1D Blocktiling (Naive)'], 'v', 1.8),
        ('smem', 'Shared Memory Tiling', colors['Shared Memory Tiling'], 'd', 1.8),
        ('b1d_coalesced', '1D Blocktiling (Coalesced)', colors['1D Blocktiling (Coalesced)'], 'p', 2.0),
        ('b2d', '2D Blocktiling', colors['2D Blocktiling'], 'h', 2.2),
        ('vectorized_smem', 'Vectorized Smem Loading', colors['Vectorized Smem Loading'], 'D', 2.2),
        ('wraptiling', 'Warp Tiling', colors['Warp Tiling'], 'X', 2.5),
        ('wraptiling_float4', 'Warp Tiling (float4 Load)', colors['Warp Tiling (float4 Load)'], 'P', 2.5),
    ]

    for key, label, color, marker, lw in pct_keys:
        y_val = (np.array(gflops_data[key]) / cublas) * 100.0
        ax.plot(x, y_val, label=label, color=color, marker=marker, linewidth=lw)

    ax.axhline(y=100.0, color=colors['NVIDIA cuBLAS'], linestyle='--', alpha=0.85, linewidth=2.0, label='NVIDIA cuBLAS (100%)')

    ax.set_title('CUDA GEMM Efficiency: % of cuBLAS Performance vs Matrix Size', fontsize=15, fontweight='bold', pad=15, color='#0f172a')
    ax.set_xlabel('Matrix Size (N x N)', fontsize=12, fontweight='semibold', color='#1e293b', labelpad=10)
    ax.set_ylabel('Performance relative to cuBLAS (%)', fontsize=12, fontweight='semibold', color='#1e293b', labelpad=10)

    ax.set_xticks(x)
    ax.set_xticklabels(sizes, fontsize=9.5, color='#334155')
    ax.tick_params(colors='#334155', labelsize=10)

    ax.grid(True, which='both', linestyle='--', alpha=0.3, color='#94a3b8')
    ax.set_ylim(0, 115)

    ax.legend(loc='upper left', facecolor='#ffffff', edgecolor='#cbd5e1', fontsize=8.5, labelcolor='#0f172a', framealpha=0.95)

    plt.tight_layout()
    plt.savefig(output_filename, dpi=300)
    plt.close()
    print(f"Generated plot: {output_filename}")


def create_speedup_plot(gflops_data, output_filename=''):
    plt.style.use('default')
    fig, ax = plt.subplots(figsize=(13, 7.5), dpi=300)

    fig.patch.set_facecolor('#ffffff')
    ax.set_facecolor('#ffffff')

    x = np.arange(len(sizes))
    naive = np.array(gflops_data['naive'])

    speedup_keys = [
        ('naive_coalesced', 'Naive Coalesced CUDA', colors['Naive Coalesced'], '^', 1.8),
        ('b1d_naive', '1D Blocktiling (Naive)', colors['1D Blocktiling (Naive)'], 'v', 1.8),
        ('smem', 'Shared Memory Tiling', colors['Shared Memory Tiling'], 'd', 1.8),
        ('b1d_coalesced', '1D Blocktiling (Coalesced)', colors['1D Blocktiling (Coalesced)'], 'p', 2.0),
        ('b2d', '2D Blocktiling', colors['2D Blocktiling'], 'h', 2.2),
        ('vectorized_smem', 'Vectorized Smem Loading', colors['Vectorized Smem Loading'], 'D', 2.2),
        ('wraptiling', 'Warp Tiling', colors['Warp Tiling'], 'X', 2.5),
        ('wraptiling_float4', 'Warp Tiling (float4 Load)', colors['Warp Tiling (float4 Load)'], 'P', 2.5),
        ('cublas', 'NVIDIA cuBLAS', colors['NVIDIA cuBLAS'], '*', 2.5),
    ]

    for key, label, color, marker, lw in speedup_keys:
        y_val = np.array(gflops_data[key]) / naive
        ax.plot(x, y_val, label=label, color=color, marker=marker, linewidth=lw)

    ax.axhline(y=1.0, color=colors['Naive CUDA'], linestyle=':', alpha=0.8, linewidth=1.5, label='Naive CUDA Baseline (1.0x)')

    ax.set_title('CUDA GEMM Speedup vs Naive Baseline (x-factor)', fontsize=15, fontweight='bold', pad=15, color='#0f172a')
    ax.set_xlabel('Matrix Size (N x N)', fontsize=12, fontweight='semibold', color='#1e293b', labelpad=10)
    ax.set_ylabel('Speedup Multiplier (vs Naive Baseline)', fontsize=12, fontweight='semibold', color='#1e293b', labelpad=10)

    ax.set_xticks(x)
    ax.set_xticklabels(sizes, fontsize=9.5, color='#334155')
    ax.tick_params(colors='#334155', labelsize=10)

    ax.grid(True, which='both', linestyle='--', alpha=0.3, color='#94a3b8')

    ax.legend(loc='upper left', facecolor='#ffffff', edgecolor='#cbd5e1', fontsize=8.5, labelcolor='#0f172a', framealpha=0.95)

    plt.tight_layout()
    plt.savefig(output_filename, dpi=300)
    plt.close()
    print(f"Generated plot: {output_filename}")


if __name__ == '__main__':
    script_dir = os.path.dirname(os.path.abspath(__file__))
    benchmarks_dir = os.path.join(script_dir, 'benchmarks')
    plots_dir = os.path.join(script_dir, 'plots')
    gflops_data, time_data = load_benchmark_data(benchmarks_dir)

    create_gflops_plot(gflops_data, scaled_x=False, output_filename=os.path.join(plots_dir, 'gflops_vs_matrix_size.png'))
    create_gflops_plot(gflops_data, scaled_x=True, output_filename=os.path.join(plots_dir, 'gflops_vs_matrix_size_scaled.png'))
    create_execution_time_plot(time_data, output_filename=os.path.join(plots_dir, 'execution_time_vs_matrix_size.png'))
    create_cublas_percentage_plot(gflops_data, output_filename=os.path.join(plots_dir, 'cublas_percentage_vs_matrix_size.png'))
    create_speedup_plot(gflops_data, output_filename=os.path.join(plots_dir, 'speedup_vs_naive.png'))
