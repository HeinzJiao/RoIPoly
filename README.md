<div align="center">
  <h1>RoIPoly: Vectorized Building Outline Extraction Using Vertex and Logit Embeddings</h1>
  <p><strong>RoIPoly</strong> is a RoI query-based framework for efficient polygonal building outline extraction with strong vertex interaction and no redundant-vertex post-processing.</p>
  <p>
    <a href="https://doi.org/10.1016/j.isprsjprs.2025.03.030"><img src="https://img.shields.io/badge/ISPRS-2025-0A66C2" alt="ISPRS 2025"></a>
    <a href="https://github.com/HeinzJiao/RoIPoly"><img src="https://img.shields.io/badge/Code-GitHub-black" alt="GitHub"></a>
  </p>
</div>

## Overview

RoIPoly formulates each polygon vertex as a query and constrains attention to the most relevant Region-of-Interest features of each building. Together with the proposed learnable logit embedding, this design improves vertex classification quality and removes the need for redundant-vertex post-processing.

## Environment Setup

### 1. Create a Conda Environment

```bash
conda create -n roipoly python=3.8
conda activate roipoly
```

### 2. Install PyTorch

```bash
pip install torch==1.9.0+cu111 torchvision==0.10.0+cu111 -f https://download.pytorch.org/whl/torch_stable.html
```

### 3. Install Detectron2

Follow the official [Detectron2 installation guide](https://github.com/facebookresearch/detectron2/tree/main).

If installation is problematic, you can also download the [detectron2 fork from RoomFormer](https://github.com/ywyue/RoomFormer/tree/main/detectron2) and place it directly under the `RoIPoly` directory.

### 4. Install Boundary IoU API

Follow the official [Boundary IoU API installation guide](https://github.com/bowenc0221/boundary-iou-api).

### 5. Install Remaining Dependencies

```bash
pip install -r requirements.txt
```

### 6. Compile Deformable Attention Modules

```bash
cd roipoly/ops
sh make.sh
```

## Training

Train the model with:

```bash
CUDA_VISIBLE_DEVICES=<gpu_ids> python3 train_net.py --num-gpus <number_of_gpus> --config-file <path_to_config_file> --dataset-name <dataset_name> --train-json <path_to_annotation_json> --train-path <path_to_training_images>
```

Example:

```bash
CUDA_VISIBLE_DEVICES=1 python3 scripts/train_net.py --num-gpus 1 --config-file configs/roipoly.res50.34pro.aicrowd.yaml --dataset-name crowdai_train --train-json ./data/crowdai/train/annotation_sm_clean_us_index.json --train-path ./data/crowdai/train/images
```

## Practical Notes

1. During training, RoIPoly uses ground-truth bounding boxes as proposal RoIs.
2. For the results reported in the paper, proposal RoIs at inference time are provided by a Sparse R-CNN detector trained on the CrowdAI dataset.
3. For the results reported in the paper, the CrowdAI dataset is split into `small-medium` and `large` subsets, the model is trained separately on the two subsets, and the final predictions on the full dataset are obtained by merging the predictions from the two corresponding models.
4. This subset-based training strategy is adopted purely because of limited computational resources. If sufficient memory and compute are available, we recommend training directly on the complete CrowdAI dataset instead of splitting it into subsets.

## Inference

Inference and evaluation utilities are provided in `scripts/evaluate.py`.

## Citation

If you find RoIPoly useful in your research, please consider citing:

```bibtex
@article{jiao2025roipoly,
  title={RoIPoly: Vectorized building outline extraction using vertex and logit embeddings},
  author={Jiao, Weiqin and Cheng, Hao and Vosselman, George and Persello, Claudio},
  journal={ISPRS Journal of Photogrammetry and Remote Sensing},
  volume={224},
  pages={317--328},
  year={2025},
  publisher={Elsevier}
}
```

## Related Research

This repository belongs to a broader research line on polygonal vectorization from aerial imagery and large-scale topographic map generation. If RoIPoly is relevant to your work, you may also want to follow the companion papers below.

| Paper | Venue | Focus | Resources |
| --- | --- | --- | --- |
| **ACPV-Net** | CVPR 2026 | All-class polygonal vectorization and topology-aware vector basemap generation from a single aerial image. | [Paper](https://arxiv.org/abs/2603.16616) · [Code](https://github.com/HeinzJiao/ACPV-Net) · [Data](https://huggingface.co/datasets/HeinzJiao/Deventer-512) · [Weights](https://huggingface.co/HeinzJiao/deventer512_vmamba-s_m_vh-ldm_kl4_b8) |
| **LDPoly** | ISPRS 2025 | Latent diffusion for polygonal road outline extraction in topographic mapping. | [Paper](https://doi.org/10.1016/j.isprsjprs.2025.10.005) · [Code](https://github.com/HeinzJiao/LDPoly) · [Data and Weights](https://drive.google.com/drive/folders/1jsjuZxFdU9a8q-m0TNCj1MfX9rixTYJl?usp=sharing) · [Demo](https://colab.research.google.com/drive/1IW5AGfn3w3y9wSquYgXolGhcVwIWkoNd#scrollTo=eval_run) |
| **PolyR-CNN** | ISPRS 2024 | End-to-end polygonal building outline extraction with efficient RoI-based modeling. | [Paper](https://doi.org/10.1016/j.isprsjprs.2024.10.006) · [Code](https://github.com/HeinzJiao/PolyR-CNN) |

## Acknowledgements

This repository benefits from the excellent open-source contributions of [Sparse R-CNN](https://github.com/PeizeSun/SparseR-CNN), [Deformable DETR](https://github.com/fundamentalvision/Deformable-DETR), and [RoomFormer](https://github.com/ywyue/RoomFormer). We thank the authors for their great work.
