# cuda_float_compress [WIP]

WORK IN PROGRESS

Python package for compressing floating-point PyTorch tensors based on the [cuSZp](https://github.com/szcompressor/cuSZp) project.


## Setup

```bash
git clone https://github.com/catid/cuda_float_compress
cd cuda_float_compress
git submodule update --init --recursive

conda create -n cfc python=3.10 -y && conda activate cfc

pip install .
```


## Testing

After installing the package, you can run the example script:

```bash
conda activate cfc

python examples/model_compress_example.py
```
