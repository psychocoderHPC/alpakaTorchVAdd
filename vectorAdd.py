#!/usr/bin/env python3
# Copyright 2025 René Widera
# SPDX-License-Identifier: ISC

import torch


# Load the custom op
torch.ops.load_library("./vectoradd.so")

# Create two input tensors with values 0, 1, 2, ..., 9
a = torch.arange(0, 10, dtype=torch.float32)
b = torch.arange(0, 10, dtype=torch.float32)

# Add the two vectors
c = a + b

print("a:", a)
print("b:", b)
print("c (a + b):", c)

print("lets use alpaka")
print("cpu")
c2 = torch.ops.vectoradd.add(a, b)
print("c (a + b):", c)


cuda_a = torch.arange(0,10, device='cuda', dtype=torch.float32)
cuda_b = torch.arange(0,10, device='cuda', dtype=torch.float32)
print("gpu")
out = torch.ops.vectoradd.add(cuda_a, cuda_b)
print("c (a + b):", out)


