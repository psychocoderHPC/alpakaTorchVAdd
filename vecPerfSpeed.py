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

@torch.jit.script
def call_add(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    return torch.ops.vectoradd.add(a, b)

# Sizes from 64MiB to 512MiB
sizes_mb = [64, 128, 256]
sizes_bytes = [size * 1024 * 1024 for size in sizes_mb]

def measure_time(func, warmup=3, repeat=20, device='cpu'):
    for _ in range(warmup):
        func()
    times = []
    for _ in range(repeat):
        if device == 'cuda':
            torch.cuda.synchronize()
            start = time.time()
            #with torch.no_grad():
            func()
            torch.cuda.synchronize()
            end = time.time()
        else:
            start = time.time()
            #with torch.no_grad():
            func()
            end = time.time()
        times.append(end - start)
    return np.array(times)

def compute_speedup(native_times_list, custom_times_list):
    speedups = []
    errors = []
    for native, custom in zip(native_times_list, custom_times_list):
        ratio = native / custom
        median = np.median(ratio)
        p1 = np.percentile(ratio, 1)
        speedups.append(median)
        errors.append(median - p1)
    return np.array(speedups), np.array(errors)

cpu_native_times_all = []
cpu_custom_times_all = []
cuda_native_times_all = []
cuda_custom_times_all = []

for size_bytes in sizes_bytes:
    num_elements = size_bytes // 4
    print(f"\n--- Size: {size_bytes // (1024*1024)} MiB ---")

    a_cpu = torch.arange(num_elements, dtype=torch.float32)
    b_cpu = torch.arange(num_elements, dtype=torch.float32)
    out_cpu = torch.empty_like(a_cpu)

    a_cuda = a_cpu.to('cuda')
    b_cuda = b_cpu.to('cuda')
    out_cuda = torch.empty_like(a_cuda)

    # CPU timings
    cpu_native_times = measure_time(lambda: a_cpu + b_cpu, device='cpu')
    cpu_custom_times = measure_time(lambda: call_add(a_cpu, b_cpu), device='cpu')

    # CUDA timings
    cuda_native_times = measure_time(lambda: a_cuda + b_cuda, device='cuda')
    cuda_custom_times = measure_time(lambda: call_add(a_cuda, b_cuda), device='cuda')

    print(f"CPU: native {cpu_native_times.mean()*1e3:.3f} ms | custom {cpu_custom_times.mean()*1e3:.3f} ms")
    print(f"CUDA: native {cuda_native_times.mean()*1e3:.3f} ms | custom {cuda_custom_times.mean()*1e3:.3f} ms")

    cpu_native_times_all.append(cpu_native_times)
    cpu_custom_times_all.append(cpu_custom_times)
    cuda_native_times_all.append(cuda_native_times)
    cuda_custom_times_all.append(cuda_custom_times)

# Compute speedups (native / custom)
cpu_speedup, cpu_err = compute_speedup(cpu_native_times_all, cpu_custom_times_all)
cuda_speedup, cuda_err = compute_speedup(cuda_native_times_all, cuda_custom_times_all)

# Plot
fig, (ax_cpu, ax_cuda) = plt.subplots(ncols=2, figsize=(14, 6))

# CPU speedup plot
ax_cpu.errorbar(sizes_mb, cpu_speedup, yerr=cpu_err, fmt='o-', capsize=4, label="alpaka CPU AMD EPYC 7452")
ax_cpu.axhline(1.0, color='gray', linestyle='--')
ax_cpu.set_title("CPU Speedup (native/alpaka)")
ax_cpu.set_xlabel("Input size (MiB)")
ax_cpu.set_ylabel("Speedup")
ax_cpu.grid(True)
ax_cpu.legend()

# CUDA speedup plot
ax_cuda.errorbar(sizes_mb, cuda_speedup, yerr=cuda_err, fmt='s-', capsize=4, label="alpaka CUDA A30")
ax_cuda.axhline(1.0, color='gray', linestyle='--')
ax_cuda.set_title("CUDA Speedup (native/alpaka)")
ax_cuda.set_xlabel("Input size (MiB)")
ax_cuda.set_ylabel("Speedup")
ax_cuda.grid(True)
ax_cuda.legend()

plt.suptitle("PyTorch with alpaka integration", fontsize=14)
plt.tight_layout(rect=[0, 0.03, 1, 0.95])
plt.savefig("vector_add_speedup.png")
print("Saved plot as vector_add_speedup.png")

