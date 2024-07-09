import torch
import torchvision.models as models
import cuda_float_compress
import zstandard as zstd
import numpy as np
import copy

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

def flatten_params(model):
    return torch.cat([p.data.view(-1) for p in model.parameters() if p.dtype == torch.float32])

def unflatten_params(model, flat_params):
    offset = 0
    for p in model.parameters():
        if p.dtype == torch.float32:
            numel = p.numel()
            p.data.copy_(flat_params[offset:offset + numel].view_as(p.data))
            offset += numel

def main():
    gpu_device = torch.device("cuda:0")

    # Load a pretrained model (e.g., ResNet18)
    model = models.resnet18(pretrained=True)
    model.to(gpu_device)

    # Create a copy of the model
    original_model = copy.deepcopy(model.state_dict())

    # Flatten all parameters
    original_params = flatten_params(model)
    print(f"original_params.shape: {original_params.shape}")

    # Set error bound for compression (adjust as needed)
    error_bound = 0.00001

    # Compress the rescaled parameters
    compressed_params = cuda_float_compress.cuszplus_compress(original_params, error_bound)

    print(f"compressed_params = {compressed_params.shape} {compressed_params.dtype} {compressed_params.device}")

    # Decompress the parameters
    decompressed_params = cuda_float_compress.cuszplus_decompress(compressed_params, gpu_device)

    ratio = original_params.numel() * 4.0 / compressed_params.numel()
    print(f"Overall Compression Ratio: {ratio:.2f}")

    unflatten_params(model, decompressed_params)

    for name, data in model.state_dict().items():
        if data.dtype != torch.float32:
            continue
        print(f"{name} = {data.shape} {data.dtype} {data.device}")

        modified = data.view(-1)
        original = original_model[name].view(-1)

        max_err = torch.max(torch.abs(original - modified))
        mse = torch.mean((original - modified) ** 2)
        print(f"    MSE: {mse.item()} Max Error: {max_err.item()}")

if __name__ == "__main__":
    main()
