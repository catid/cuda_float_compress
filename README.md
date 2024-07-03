# cuda_float_compress

CUDA floating-point compression PIP package

## Setup

```bash
git clone https://github.com/catid/cuda_float_compress
cd cuda_float_compress

conda create -n cfc python=3.10 -y && conda activate cfc

pip install -e .
```

## Testing

After installing the package, you can run the example script:

```bash
cd examples

python model_compress_example.py
```
