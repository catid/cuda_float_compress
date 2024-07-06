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


TODO:

Each thread processes say N=32 items (64 may also work, try it!).  Variable-length data per thread (put these at the end of the global memory array).

Separate array for number of bits in each thread (8 bits), one for each thread.
This is 0..N, where 0 would mean no data from that thread.

Separate array for offset to largest item in each thread (8 bits), one for each thread.
This is 0..N-1.

Separate array for the high bits of the largest item in each thread (32 bits), one for each thread.
It might make sense to bit-interleave these.
