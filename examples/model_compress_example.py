import torch
import torchvision.models as models
import cuda_float_compress

def compress_model_params(model, error_bound):
    # Get all parameters from the model
    params = [p.data for p in model.parameters()]
    
    # Flatten and concatenate all parameters into a single tensor
    flattened_params = torch.cat([p.view(-1) for p in params])
    
    # Ensure the tensor is on CPU and in float32 format
    flattened_params = flattened_params.cpu().float()
    
    # Compress the flattened parameters
    compressed_params = cuda_float_compress.cuszp_compress(flattened_params, error_bound)
    
    return compressed_params, flattened_params.numel()

def decompress_model_params(compressed_params, num_elements, error_bound):
    # Decompress the parameters
    decompressed_params = cuda_float_compress.cuszp_decompress(compressed_params, num_elements, compressed_params.numel(), error_bound)
    
    return decompressed_params

def main():
    # Load a pretrained model (e.g., ResNet18)
    model = models.resnet18(pretrained=True)
    
    # Set error bound for compression (adjust as needed)
    error_bound = 1e-4
    
    # Compress model parameters
    compressed_params, num_elements = compress_model_params(model, error_bound)

    print(f"Original model size: {num_elements * 4 / 1000 / 1000:.2f} MB")
    print(f"Compressed model size: {compressed_params.numel() / 1000 / 1000:.2f} MB")

    # Decompress model parameters
    decompressed_params = decompress_model_params(compressed_params, num_elements, error_bound)
    
    # Verify decompression
    original_params = torch.cat([p.data.view(-1) for p in model.parameters()])
    mse = torch.mean((original_params - decompressed_params) ** 2)
    print(f"Mean Squared Error after compression/decompression: {mse.item()}")

if __name__ == "__main__":
    main()
