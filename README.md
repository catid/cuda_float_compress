# cuda_float_compress [WIP]

WORK IN PROGRESS

Python package for compressing floating-point PyTorch tensors using the [cuSZp](https://github.com/szcompressor/cuSZp) library.


## Setup

```bash
git clone https://github.com/catid/cuda_float_compress
cd cuda_float_compress

conda create -n cfc python=3.10 -y && conda activate cfc

pip install .
```


## Testing

After installing the package, you can run the example script:

```bash
conda activate cfc

python examples/model_compress_example.py
```
