#!/usr/bin/env python3
# Copyright 2025 René Widera
# SPDX-License-Identifier: ISC

import torch
import time
import numpy as np
import matplotlib.pyplot as plt

# Load your custom op .so file
torch.ops.load_library("./vectoradd.so")

vectoradd_add = torch.ops.vectoradd.add

#@torch.jit.script
#def call_add(a, b):
#    return torch.ops.vectoradd.add(a, b)
@torch.jit.script
def call_add(a: torch.Tensor, b: torch.Tensor, out: torch.Tensor) -> torch.Tensor:
    return torch.ops.vectoradd.add(a, b, out)

# Warm-up
#vectoradd_add(torch.randn(1), torch.randn(1))

# Sizes from 1MiB to 16MiB (in bytes)
sizes_mb = [64,128,256,512]
sizes_bytes = [size * 1024 * 1024 for size in sizes_mb]

def measure_time(func, warmup=3, repeat=10, device='cpu'):
    # Warm-up runs
    for _ in range(warmup):
        out = func()

    times = []
    for _ in range(repeat):
        if device == 'cuda':
            torch.cuda.synchronize()
            start = time.time()
            with torch.no_grad():
                out = func()
            #print(out[0].item(),out[-1].item())
            torch.cuda.synchronize()
            end = time.time()
            #print("g ",end-start)
        else:
            start = time.time()
            with torch.no_grad():
                out = func()
            end = time.time()
           # print("c", end-start)
        times.append(end - start)
    return np.array(times)

def extract_median_and_error(times_list):
    medians = []
    error_bars = []
    for times in times_list:
        median = np.median(times) * 1e3       # convert to ms
        p1 = np.percentile(times, 1) * 1e3    # 1st percentile in ms
        medians.append(median)
        error_bars.append(median - p1)        # error bar downwards only
    return np.array(medians), np.array(error_bars)

cpu_native_times_all = []
cpu_custom_times_all = []

cuda_native_times_all = []
cuda_custom_times_all = []

for size_bytes in sizes_bytes:
    num_elements = size_bytes // 4  # float32 = 4 bytes

    print(f"\n--- Size: {size_bytes // (1024*1024)} MiB ---")

    # Create CPU tensors
    a_cpu = torch.arange(num_elements, dtype=torch.float32)
    b_cpu = torch.arange(num_elements, dtype=torch.float32)
    out_cpu = torch.empty_like(a_cpu)
    tout_cpu = torch.empty_like(a_cpu)

    # Create CUDA tensors
    a_cuda = a_cpu.to('cuda')
    b_cuda = b_cpu.to('cuda')
    out_cuda = torch.empty_like(a_cuda)
    tout_cuda = torch.empty_like(a_cuda)

    # CPU native add
    cpu_native = lambda: torch.add(a_cpu, b_cpu, out = tout_cpu)
    cpu_native_times = measure_time(cpu_native, device='cpu')

    # CPU custom op
    cpu_custom = lambda: torch.ops.vectoradd.add(a_cpu, b_cpu,out_cpu)
    cpu_custom_times = measure_time(cpu_custom, device='cpu')

    # CUDA native add
    cuda_native = lambda: torch.add(a_cuda, b_cuda, out = tout_cuda)
    cuda_native_times = measure_time(cuda_native, device='cuda')

    # CUDA custom op
    cuda_custom = lambda: torch.ops.vectoradd.add(a_cuda, b_cuda ,out_cuda)
    cuda_custom_times = measure_time(cuda_custom, device='cuda')

    print(f"CPU:    native = {cpu_native_times.mean()*1e3:.3f} ms | custom = {cpu_custom_times.mean()*1e3:.3f} ms")
    print(f"CUDA:   native = {cuda_native_times.mean()*1e3:.3f} ms | custom = {cuda_custom_times.mean()*1e3:.3f} ms")

    cpu_native_times_all.append(cpu_native_times)
    cpu_custom_times_all.append(cpu_custom_times)
    cuda_native_times_all.append(cuda_native_times)
    cuda_custom_times_all.append(cuda_custom_times)

# Compute medians and error bars for plotting
cpu_native_medians, cpu_native_errors = extract_median_and_error(cpu_native_times_all)
cpu_custom_medians, cpu_custom_errors = extract_median_and_error(cpu_custom_times_all)
cuda_native_medians, cuda_native_errors = extract_median_and_error(cuda_native_times_all)
cuda_custom_medians, cuda_custom_errors = extract_median_and_error(cuda_custom_times_all)

# Plot
# Create side-by-side subplots
fig, (ax_cpu, ax_cuda) = plt.subplots(ncols=2, figsize=(14, 6))

# --- CPU plot ---
ax_cpu.errorbar(sizes_mb, cpu_native_medians, yerr=cpu_native_errors, fmt='o-', label='CPU native', capsize=4)
ax_cpu.errorbar(sizes_mb, cpu_custom_medians, yerr=cpu_custom_errors, fmt='o--', label='CPU alpaka', capsize=4)
ax_cpu.set_title("CPU Vector Add")
ax_cpu.set_xlabel("Input size (MiB)")
ax_cpu.set_ylabel("Time per add (ms)")
ax_cpu.grid(True)
ax_cpu.legend()

# --- CUDA plot ---
ax_cuda.errorbar(sizes_mb, cuda_native_medians, yerr=cuda_native_errors, fmt='s-', label='CUDA native', capsize=4)
ax_cuda.errorbar(sizes_mb, cuda_custom_medians, yerr=cuda_custom_errors, fmt='s--', label='CUDA alpaka', capsize=4)
ax_cuda.set_title("CUDA Vector Add")
ax_cuda.set_xlabel("Input size (MiB)")
ax_cuda.grid(True)
ax_cuda.legend()

# --- Layout and save ---
plt.suptitle("Vector Add Performance (median with 1st percentile error bar)", fontsize=14)
plt.tight_layout(rect=[0, 0.03, 1, 0.95])  # Adjust space for the suptitle
plt.savefig("vector_add_cpu_vs_cuda.png")
print("Saved plot as vector_add_cpu_vs_cuda.png")

