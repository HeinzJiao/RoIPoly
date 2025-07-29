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
CUDA_VISIBLE_DEVICES=<gpu_ids> python3 train_net.py --num-gpus <number_of_gpus> --config-file <path_to_config_file> --dataset-name <dataset_name> --train-json <path_to_annotation_json> --train-path <path_to_training_images>
```

#### Arguments:
- CUDA_VISIBLE_DEVICES (optional): This environment variable is used to specify which GPU(s) to use. For example, CUDA_VISIBLE_DEVICES=1 makes the second GPU (index 1) available for training.
  - If you want to use multiple GPUs, you can specify them like CUDA_VISIBLE_DEVICES=0,1,2.
  - If not set, the first available GPU will be used by default.
- --num-gpus: This argument specifies how many GPUs to use during training. Set --num-gpus <number_of_gpus> to define the number of GPUs. For example, --num-gpus 1 for single-GPU training and --num-gpus 2 for two GPUs.
- --config-file: Path to the configuration file that contains model-specific parameters, dataset paths, and other training settings. Replace <path_to_config_file> with the actual path to your configuration file.
- --dataset-name: The name of the dataset being used. Default is aicrowd_train, but you can change it to your own dataset name.
- --train-json: Path to the training dataset's annotation file in COCO format. Replace <path_to_annotation_json> with the path to your dataset's annotation JSON file.
- --train-path: Path to the folder containing training images. Replace <path_to_training_images> with the actual path to your dataset's images.

#### Example:
If you want to train a model using the **Swin-Base Transformer** as the backbone on the **CrowdAI Small-Medium dataset**, you can execute the following command:
```
CUDA_VISIBLE_DEVICES=1 python3 scripts/train_net.py --num-gpus 1 --config-file configs/roipoly.res50.34pro.aicrowd.yaml --dataset-name crowdai_train --train-json ./data/crowdai/train/annotation_sm_clean_us_index.json --train-path ./data/crowdai/train/images
```

## 🔍 Tips for Using This Repository

1. **ROI Generation Strategy**  
   - During training, we use **ground-truth bounding boxes** to provide region proposals (RoIs).  
   - During inference, RoIs are obtained from a **pretrained object detector**.

2. **Polygon Size Partitioning (for CrowdAI Dataset)**  
   - In training, we split the CrowdAI dataset into **small-medium** polygons and **large** polygons.  
   - This design is **purely due to limited computational resources**, not model performance considerations.  
   - Without this split, we would need to set a very large `num_proposal_vertices_per_polygon` to handle complex large polygons, which would significantly increase memory and computation cost.  
   - ⚠️ **If you have sufficient computing resources, we do not recommend applying this split.**

## Status

The remaining code is currently being organized and will be uploaded soon. Stay tuned for updates!

## Coming Soon

- Complete inference pipelines
