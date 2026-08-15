# Anyprune
A neural network to prune 3D Gaussian Splats in a single forward pass.

## Project setup

### Installing dependencies
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

Now, let's install all the required dependencies for this project and 
its git submodules. From the repo root:
```bash
git submodule update --init --recursive
pip install "setuptools<70" wheel
pip install torch==2.2.0 torchvision==0.17.0 torchaudio==2.2.0 --index-url https://download.pytorch.org/whl/cu121
pip install ./wheels/pytorch3d-*.whl
pip install --no-build-isolation "https://github.com/Dao-AILab/flash-attention/releases/download/v2.5.8/flash_attn-2.5.8%2Bcu122torch2.2cxx11abiFALSE-cp310-cp310-linux_x86_64.whl"
pip install torch-scatter torch-sparse torch-cluster torch-geometric -f https://data.pyg.org/whl/torch-2.2.0+cu121.html
pip install spconv-cu120
pip install anyprune/models/SplatFormer/Pointcept
# Install all SplatFormer dependencies apart from spconv
grep -ivE '^[[:space:]]*spconv' anyprune/models/SplatFormer/requirements.txt | pip install -r /dev/stdin
pip install -r requirements.txt
```

### Setting a Wandb API key
If you wish to use Wandb, it is suggested that you provide your API key
in a `.env` file. To do that, run:
```bash
echo "WANDB_API_KEY=<your_key_here>" > .env
```

### Downloading official Splatformer checkpoints
If you wish to use original, not fine-tuned checkpoints for SplatFormer,
you may download them from [here](https://drive.google.com/drive/folders/1WkrOexVd8S0lqbnr8Jx0wSqAiQQYVKdm) and move them manually into a  
`checkpoints/splatformer` directory under the root dir of the project.