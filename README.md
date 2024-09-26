# RoIPoly

The code is currently being organized and will be uploaded soon.

## Environment Setup

#### 1. Create Conda Environment (Recommended)
To begin, create and activate a new conda environment for this project:

```
conda create -n roipoly python=3.8
conda activate roipoly
```

### 2. Install PyTorch
Install the specific version of PyTorch compatible with CUDA 11.1:

```
pip install torch==1.9.0+cu111 torchvision==0.10.0+cu111 -f https://download.pytorch.org/whl/torch_stable.html
```

### 3. Install Detectron2
