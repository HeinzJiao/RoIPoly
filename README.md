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

If you're using CUDA 12.x (e.g., CUDA 12.4), you'll need to install the nightly version of PyTorch, as the stable release doesn't yet support CUDA 12.4. Use the following command to install PyTorch with CUDA 12.1 support:

```
pip install torch torchvision torchaudio --pre --index-url https://download.pytorch.org/whl/nightly/cu121
```

#### 3. Install Detectron2
Follow the official [Detectron2 installation guide](https://github.com/facebookresearch/detectron2/tree/main) for installation instructions.

If you encounter any issues during installation, you can directly download the [detectron2](https://github.com/ywyue/RoomFormer/tree/main/detectron2) folder and place it under the RoIPoly directory. The final structure should look like this:

```
RoIPoly/
│
├── detectron2/
├── other files...
```

#### 4. Install Boundary IoU API
Follow the official [Boundary IoU API installation guide](https://github.com/bowenc0221/boundary-iou-api) for installation instructions.

#### 5. Install Required Packages
To install the remaining dependencies, run:
```
pip install -r requirements.txt
```

#### 6. Compile Deformable-Attention Modules (from [Deformable-DETR](https://github.com/fundamentalvision/Deformable-DETR))
To compile the deformable attention modules, navigate to the ops directory and execute the following:
```
cd roipoly/ops
sh make.sh
```

## Training the model
#### To train the model, use the following command:
```
CUDA_VISIBLE_DEVICES=<gpu_ids> python3 train_net.py --num-gpus <number_of_gpus> --config-file <path_to_config_file>
```

#### Arguments:
- CUDA_VISIBLE_DEVICES (optional): This environment variable is used to specify which GPU(s) to use. For example, CUDA_VISIBLE_DEVICES=1 makes the second GPU (index 1) available for training.
  - If you want to use multiple GPUs, you can specify them like CUDA_VISIBLE_DEVICES=0,1,2.
  - If not set, the first available GPU will be used by default.
- --num-gpus: This argument specifies how many GPUs to use during training. Set --num-gpus <number_of_gpus> to define the number of GPUs. For example, --num-gpus 1 for single-GPU training and --num-gpus 2 for two GPUs.
- --config-file: Path to the configuration file that contains model-specific parameters, dataset paths, and other training settings. Replace <path_to_config_file> with the actual path to your configuration file.

#### Example:
If you want to train a model using the **Swin-Base Transformer** as the backbone on the **CrowdAI Small-Medium dataset**, you can execute the following command:
```
CUDA_VISIBLE_DEVICES=0 python3 train_net.py --num-gpus 1 --config-file configs/roipoly.res50.34pro.aicrowd.yaml
```
