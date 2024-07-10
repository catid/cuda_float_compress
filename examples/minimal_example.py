import torch
import cuda_float_compress
import numpy as np

gpu_device = torch.device("cuda")

# Generate some random data
mean = 0
std = 1
num_floats = 32 * 1024
original_data = torch.tensor(np.random.normal(mean, std, num_floats), dtype=torch.float32).to(gpu_device)

# Compress the data, specifying the maximum error bound
error_bound = 0.0001
compressed_data = cuda_float_compress.cuszplus_compress(original_data, error_bound)

# --- Send the data over the network here ---

# Decompress the data
decompressed_data = cuda_float_compress.cuszplus_decompress(compressed_data, gpu_device)

# Verify that the decompressed data is the same as the original data
assert torch.allclose(original_data, decompressed_data, atol=error_bound)
print(f"Works! Compression Ratio: {original_data.numel() * 4.0 / compressed_data.numel():.2f}")
