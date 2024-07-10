import torch
import torchvision.models as models
import cuda_float_compress
import numpy as np
import copy
import time

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
    model = models.regnet_y_32gf(pretrained=True)
    model.to(gpu_device)

    # Create a copy of the model
    original_model = copy.deepcopy(model.state_dict())

    t0 = time.time()
    error_bound = 0.00001
    original_params = flatten_params(model)
    compressed_params = cuda_float_compress.cuszplus_compress(original_params, error_bound)
    t1 = time.time()

    print(f"original_params.shape: {original_params.shape}")
    print(f"compressed_params = {compressed_params.shape} {compressed_params.dtype} {compressed_params.device}")

    t2 = time.time()
    decompressed_params = cuda_float_compress.cuszplus_decompress(compressed_params, gpu_device)
    unflatten_params(model, decompressed_params)
    t3 = time.time()

    for name, data in model.state_dict().items():
        if data.dtype != torch.float32:
            continue
        print(f"{name} = {data.shape} {data.dtype} {data.device}")

        modified = data.view(-1)
        original = original_model[name].view(-1)

        max_err = torch.max(torch.abs(original - modified))
        mse = torch.mean((original - modified) ** 2)
        print(f"    MSE: {mse.item()} Max Error: {max_err.item()}")
        print(f"    Delta: {(modified)}")

    ratio = original_params.numel() * 4.0 / compressed_params.numel()
    print(f"Overall Compression Ratio: {ratio:.2f}")
    print(f"Time to compress params: {t1 - t0:.2f} s")
    print(f"Time to decompress params: {t3 - t2:.2f} s")

if __name__ == "__main__":
    main()
