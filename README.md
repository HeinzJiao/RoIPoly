# RoIPoly

The code is currently being organized and will be uploaded soon.

## Environment Setup

#### 1. Create Conda Environment (Recommended)
To begin, create and activate a new conda environment for this project:

```
conda create -n roipoly python=3.8
conda activate roipoly
```

#### 2. Install PyTorch
Install the specific version of PyTorch compatible with CUDA 11.1:

```
pip install torch==1.9.0+cu111 torchvision==0.10.0+cu111 -f https://download.pytorch.org/whl/torch_stable.html
```

#### 3. Install Detectron2
Follow the official [Detectron2 installation guide](https://github.com/facebookresearch/detectron2/tree/main) for installation instructions.

If you encounter any issues during installation, you can directly download the detectron2 folder and place it under the RoIPoly directory. The final structure should look like this:

```
RoIPoly/
│
├── detectron2/
├── other files...
```

#### 4. Install Required Packages
To install the remaining dependencies, run:
```
pip install -r requirements.txt
```

#### 5. Compile Deformable-Attention Modules (from [Deformable-DETR](https://github.com/fundamentalvision/Deformable-DETR))
To compile the deformable attention modules, navigate to the ops directory and execute the following:
```
cd roipoly/ops
sh make.sh
```


#### 5. Compile Deformable-Attention Modules (from Deformable-DETR)


