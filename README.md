# Anyprune
A neural network to prune 3D Gaussian Splats in a single forward pass.

## Installing the dependencies
We are going to need Python 3.10, PyTorch 2.2.0 and CUDA 12.1.

Here we provide instructions to setup the environment using Conda.
First, create and activate the environment:
```bash
conda create -y -n anyprune python=3.10
conda activate anyprune
```

Before going further, you can redirect the install caches to a scratch 
folder (epecially useful for HPC environments with a quota on the home
directory).
```bash
CACHE_DIR=/tmp
PIP_CACHE_DIR="$CACHE_DIR/pip"
TORCH_HOME="$CACHE_DIR/torch"
HF_HOME="$CACHE_DIR/huggingface"
TRITON_CACHE_DIR="$CACHE_DIR/triton"
```

