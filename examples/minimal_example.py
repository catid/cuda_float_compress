import torch
import numpy as np
import cuda_float_compress

cuda_device = torch.device("cuda")

# Generate some random data
original_data = torch.tensor(np.random.normal(0, 1, 32 * 1024), dtype=torch.float32, device=cuda_device)

# Compress the data, specifying the maximum error bound
max_error = 0.0001
compressed_data = cuda_float_compress.cuszplus_compress(original_data, max_error)

# --- Send the data over the network here ---

# Decompress the data
decompressed_data = cuda_float_compress.cuszplus_decompress(compressed_data, cuda_device)

# Verify that the decompressed data is the same as the original data
assert torch.allclose(original_data, decompressed_data, atol=max_error)
print(f"Works! Compression Ratio: {original_data.numel() * 4.0 / compressed_data.numel():.2f}")
