#!/usr/bin/env python3
# Copyright 2025 René Widera
# SPDX-License-Identifier: ISC


import torch
import time
import matplotlib.pyplot as plt
import numpy as np

# Load the custom op
torch.ops.load_library("./vectoradd.so")

def measure_time(func, warmup=3, repeat=10, device='cpu'):
    # Warm-up
    for _ in range(warmup):
        func()

    times = []
    for _ in range(repeat):
        if device == 'cuda':
            torch.cuda.synchronize()
            start = time.time()
            func()
            torch.cuda.synchronize()
            end = time.time()
        else:
            start = time.time()
            func()
            end = time.time()
        times.append(end - start)
    return np.mean(times), np.std(times)


sizes_mib = [1, 2, 4, 8, 16, 512]  # MiB sizes
sizes_bytes = [s * 1024 * 1024 for s in sizes_mib]
sizes_mb = sizes_mib

cpu_native_times = []
cpu_custom_times = []

cuda_native_times = []
cuda_custom_times = []

for size_bytes in sizes_bytes:
    num_elements = size_bytes // 4  # float32 = 4 bytes

    print(f"\n--- Size: {size_bytes // 1024 // 1024} MiB ---")

    # --- CPU ---
    a_cpu = torch.rand(num_elements, dtype=torch.float32)
    b_cpu = torch.rand(num_elements, dtype=torch.float32)
    
    # Native CPU
    native_cpu = lambda: a_cpu + b_cpu
    t_native_cpu, _ = measure_time(native_cpu, device='cpu')
    
    # Custom CPU
    custom_cpu = lambda: torch.ops.vectoradd.add(a_cpu, b_cpu)
    t_custom_cpu, _ = measure_time(custom_cpu, device='cpu')
    
    cpu_native_times.append(t_native_cpu * 1e3)  # to ms
    cpu_custom_times.append(t_custom_cpu * 1e3)

    print(f"CPU:    native = {t_native_cpu * 1e3:.3f} ms | custom = {t_custom_cpu * 1e3:.3f} ms")

    # --- CUDA ---
    a_cuda = torch.rand(num_elements, dtype=torch.float32, device='cuda')
    b_cuda = torch.rand(num_elements, dtype=torch.float32, device='cuda')

    # Native CUDA
    native_cuda = lambda: a_cuda + b_cuda
    t_native_cuda, _ = measure_time(native_cuda, device='cuda')
    
    # Custom CUDA
    custom_cuda = lambda: torch.ops.vectoradd.add(a_cuda, b_cuda)
    t_custom_cuda, _ = measure_time(custom_cuda, device='cuda')

    cuda_native_times.append(t_native_cuda * 1e3)
    cuda_custom_times.append(t_custom_cuda * 1e3)

    print(f"CUDA:   native = {t_native_cuda * 1e3:.3f} ms | custom = {t_custom_cuda * 1e3:.3f} ms")


# --- Plotting ---
plt.figure(figsize=(10, 6))

plt.plot(sizes_mb, cpu_native_times, 'o-', label='CPU native')
plt.plot(sizes_mb, cpu_custom_times, 'o--', label='CPU custom op')

plt.plot(sizes_mb, cuda_native_times, 's-', label='CUDA native')
plt.plot(sizes_mb, cuda_custom_times, 's--', label='CUDA custom op')

plt.xlabel("Input size (MiB)")
plt.ylabel("Time per add (ms)")
plt.title("Vector Add Performance: Native vs Custom Op")
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.savefig("vector_add_benchmark.png")

print("\n✅ Benchmark complete. Plot saved to vector_add_benchmark.png")

