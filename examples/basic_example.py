import torch
import cuda_float_compress
import time
import numpy as np

def main():
    gpu_device = torch.device("cuda:0")

    mean = 0
    std = 1
    num_floats = 32 * 1024
    original_data = torch.tensor(np.random.normal(mean, std, num_floats), dtype=torch.float32).to(gpu_device)

    print(f"original_data.shape: {original_data.shape}")
    print(f"original_data mean: {original_data.mean().item():.4f}, std: {original_data.std().item():.4f}")

    # Compression
    error_bound = 0.0001
    t0 = time.time()
    compressed_data = cuda_float_compress.cuszplus_compress(original_data, error_bound)
    t1 = time.time()

    print(f"compressed_data = {compressed_data.shape} {compressed_data.dtype} {compressed_data.device}")

    # Decompression
    t2 = time.time()
    decompressed_data = cuda_float_compress.cuszplus_decompress(compressed_data, gpu_device)
    t3 = time.time()

    # Calculate statistics
    max_err = torch.max(torch.abs(original_data - decompressed_data))
    mse = torch.mean((original_data - decompressed_data) ** 2)
    ratio = original_data.numel() * 4.0 / compressed_data.numel()

    print(f"MSE: {mse.item()}")
    print(f"Max Error: {max_err.item()}")
    print(f"Compression Ratio: {ratio:.2f}")
    print(f"Time to compress: {t1 - t0:.6f} s")
    print(f"Time to decompress: {t3 - t2:.6f} s")

if __name__ == "__main__":
    main()
