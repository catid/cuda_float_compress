import torch
import torchvision.models as models
import cuda_float_compress
import zstandard as zstd
import numpy as np

# Function to compress a tensor using Zstd
def compress_tensor_zstd(tensor, compression_level=1):
    # Convert tensor to bytes
    tensor_bytes = tensor.cpu().numpy().tobytes()
    
    # Create a Zstd compressor
    cctx = zstd.ZstdCompressor(level=compression_level)
    
    # Compress the tensor bytes
    compressed_data = cctx.compress(tensor_bytes)
    
    return compressed_data

# Function to decompress Zstd compressed data back to a tensor
def decompress_tensor_zstd(compressed_data, expected_bytes):
    # Create a Zstd decompressor
    dctx = zstd.ZstdDecompressor()
    
    # Decompress the data
    decompressed_bytes = dctx.decompress(compressed_data)

    if len(decompressed_bytes) != expected_bytes:
        return None

    tensor = torch.tensor(np.frombuffer(decompressed_bytes, dtype=np.uint8))

    return tensor

def main():
    # Load a pretrained model (e.g., ResNet18)
    model = models.resnet18(pretrained=True)

    # Verify decompression
    original_params = torch.cat([p.data.view(-1) for p in model.parameters()])
    print(f"oring_params.shape: {original_params.shape}")
    for p in model.parameters():
        data = p.data.view(-1)

        (d_min, d_max) = torch.aminmax(data.detach())
        range = (d_max - d_min).float()
        rescaled = (data - d_min).float() / range

        raw_data = rescaled

        print(f"raw_data = {raw_data}")

        # Set error bound for compression (adjust as needed)
        error_bound = 0.00001

        num_elements = raw_data.numel()
        compressed_params = cuda_float_compress.cuszplus_compress(raw_data, error_bound)

        print(f"compressed_params = {compressed_params}")

        zcomp = compress_tensor_zstd(compressed_params)

        dcomp = decompress_tensor_zstd(zcomp, compressed_params.numel())

        # Check if dcomp matches the input compressed_params
        if dcomp is not None:
            is_match = torch.all(dcomp == compressed_params)
            if is_match:
                print("Decompressed data matches the input compressed data.")
            else:
                print("WARNING: Decompressed data does not match the input compressed data!")
                # You can add more detailed comparison here if needed
                mismatch_count = torch.sum(dcomp != compressed_params)
                print(f"Number of mismatched elements: {mismatch_count}")
        else:
            print("ERROR: Decompression failed. Unable to compare.")

        decompressed_params = cuda_float_compress.cuszplus_decompress(compressed_params, num_elements, error_bound)

        mse = torch.mean((raw_data - decompressed_params) ** 2)
        ratio = raw_data.numel() * 4.0 / compressed_params.numel()
        zratio = raw_data.numel() * 4.0 / len(zcomp)
        # Emulate applying Zstd selectively (usually helps for larger tensors)
        if zratio > ratio:
            ratio = zratio
        print(f"Mean Squared Error after compression/decompression: {mse.item()} Ratio: {ratio:.2f}")

if __name__ == "__main__":
    main()
