"""
detector.py

This script implements the main pipeline for the RoIPoly model using uniform sampling to pad polygons.

- The main model architecture is defined in `head.py`.
- The loss computation is detailed in `loss.py`.
"""
import torch
from torch import nn
from detectron2.modeling import META_ARCH_REGISTRY, build_backbone
from detectron2.structures import Boxes, ImageList, Instances
from .loss import SetCriterion, HungarianMatcher
from .head import DynamicHead
from util.misc import (NestedTensor, nested_tensor_from_tensor_list)

__all__ = ["RoIPoly"]


@META_ARCH_REGISTRY.register()
class RoIPoly(nn.Module):
    """
    Implement RoIPoly
    """

    def __init__(self, cfg):
        super().__init__()

        self.device = torch.device(cfg.MODEL.DEVICE)

        self.in_features = cfg.MODEL.ROI_HEADS.IN_FEATURES  # ["p2", "p3", "p4", "p5"]
        self.num_classes = cfg.MODEL.RoIPoly.NUM_CLASSES  # 1, valid or invalid corner
        self.num_proposals = cfg.MODEL.RoIPoly.NUM_PROPOSALS
        self.num_corners = cfg.MODEL.RoIPoly.NUM_CORNERS
        self.hidden_dim = cfg.MODEL.RoIPoly.HIDDEN_DIM  # 256

        # Build Backbone.
        self.backbone = build_backbone(cfg)
        self.size_divisibility = self.backbone.size_divisibility
        self.deep_supervision = cfg.MODEL.RoIPoly.DEEP_SUPERVISION  # False
        
        # Build Dynamic Head.
        self.head = DynamicHead(cfg=cfg, roi_input_shape=self.backbone.output_shape())

        # Build HungarianMatcher.
        matcher = HungarianMatcher(cost_class=2, cost_coords=5)

        weight_dict = {
            'loss_ce': 2,
            'loss_coords': 5,
            'loss_raster': 1
        }
        if self.deep_supervision:  # False
            aux_weight_dict = {}
            for i in range(5):
                aux_weight_dict.update({k + f"_{i}": v for k, v in weight_dict.items()})
            weight_dict.update(aux_weight_dict)

        losses = ['labels', 'polys']

        # Build Criterion.
        self.criterion = SetCriterion(self.num_classes, matcher, weight_dict, losses)  # This class computes the loss for RoIPoly.

        pixel_mean = torch.Tensor(cfg.MODEL.PIXEL_MEAN).to(self.device).view(3, 1, 1)
        pixel_std = torch.Tensor(cfg.MODEL.PIXEL_STD).to(self.device).view(3, 1, 1)
        self.normalizer = lambda x: (x - pixel_mean) / pixel_std
        self.to(self.device)

    def forward(self, batched_inputs):
        """
        Forward pass of the model.

        Args:
            batched_inputs (list): A list of batched outputs from `DatasetMapper`.
                Each item in the list is a dictionary containing:
                * image (Tensor): The image in (C, H, W) format.
                * instances (Instances): Ground truth instances.
                * file_name (str): The image file path (e.g., "./datasets/AIcrowd/train/images/....jpg").

        Returns:
            loss_dict (dict) if training, or outputs (dict) if in evaluation mode.
        """
        images, images_whwh = self.preprocess_image(batched_inputs)
        # images_whwh: torch.Tensor of shape [batch_size, 4], [[width, height, width, height], ...]
        if isinstance(images, (list, torch.Tensor)):  # False
            images = nested_tensor_from_tensor_list(images)

        # Feature Extraction.
        src = self.backbone(images.tensor)  # `images.tensor`: torch.Size([batch_size, channels, height, width])
        features = list()        
        for f in self.in_features:  # `self.in_features`: ["p2", "p3", "p4", "p5"]
            feature = src[f]
            features.append(feature)

        # Ground truth bounding boxes preparation (used in training).
        gt_instances = [x["instances"].to(self.device) for x in batched_inputs]
        proposal_boxes, box_lengths = pad_gt_boxes(gt_instances, self.num_proposals, self.device)
        # `proposal_boxes`: torch.Tensor([bs, num_proposals, 4]), absolute coordinates (x1, y1, x2, y2).
        # `box_lengths`: torch.Tensor([bs]), length of boxes for each image.

        # Prediction.
        outputs = self.head(features, proposal_boxes, images_whwh)
        # `outputs` contains dict_keys(['pred_logits', 'pred_coords']):
        # - `outputs['pred_logits']`: shape [bs, num_proposals, num_queries], classification scores.
        # - `outputs['pred_coords']`: shape [bs, num_proposals, num_queries, 2], predicted coordinates.

        if self.training:
            # Get file names for debugging or visualization purposes.
            gt_file_names = [x["file_name"] for x in batched_inputs]

            # Prepare the ground truth targets for the loss function.
            targets = prepare_targets(gt_instances, self.device)
            # `targets`: a dict containing 'coords' (absolute) and 'labels' (class 1 for annotated points, class 0 for padded points).

            # Optionally visualize targets (if required for debugging).
            # visualize_targets(targets, gt_file_names)

            # Compute the loss.
            loss_dict = self.criterion(outputs, targets, box_lengths)
            # Loss dictionary keys: ['loss_coords', 'loss_ce']
            weight_dict = self.criterion.weight_dict
            for k in loss_dict.keys():
                if k in weight_dict:
                    loss_dict[k] *= weight_dict[k]
            # Example loss dictionary:
            # {'loss_ce': tensor(0.0945, device='cuda:0', grad_fn=<MulBackward0>),
            #  'loss_coords': tensor(1.4065, device='cuda:0', grad_fn=<MulBackward0>),
            #  'cardinality_error': tensor(46.5000, device='cuda:0')}

            return loss_dict

        else:
            return outputs

    def preprocess_image(self, batched_inputs):
        """
        Normalize, pad and batch the input images.
        """
        images = [self.normalizer(x["image"].to(self.device)) for x in batched_inputs]  # (bs, c, h, w)
        images = ImageList.from_tensors(images, self.size_divisibility)  # self.size_divisibility=32
        # size_divisibility: add padding to ensure the common height and width is divisible by 'size_divisibility'.
        # This depends on the model and many models need a divisibility of 32.

        images_whwh = list()
        for bi in batched_inputs:
            h, w = bi["image"].shape[-2:]
            images_whwh.append(torch.tensor([w, h, w, h], dtype=torch.float32, device=self.device))
        images_whwh = torch.stack(images_whwh)  # (batch_size, 4)

        return images, images_whwh


def prepare_targets(gt_instances, device):
    """
    Prepares target data for training, converting ground truth instances into the appropriate format.
    """
    targets = []  # len(targets): batch size
    for gt_inst in gt_instances:  # per image
        h, w = gt_inst.image_size

        # Get bounding boxes (absolute, xyxy format) for each instance.
        boxes = gt_inst.gt_boxes.tensor.to(device) # torch.Size([num_instances, 4]), float32, xyxy format.

        # Get preprocessed masks (absolute, uniformly sampled polygon coordinates).
        gt_masks = gt_inst.gt_masks.to(device)  # torch.Size([num_instances, num_corners * 2]), int32.
        gt_masks = torch.clip(gt_masks, 0, h - 1)

        # Get corner classification labels (annotated points as class 1, sampled points as class 0).
        gt_corner_classes = gt_inst.gt_cor_cls_img.to(device).to(torch.float32)  # torch.Size([num_instances, num_corners])

        room_dict = {
            'coords': gt_masks,  # Polygon coordinates (absolute).
            'labels': gt_corner_classes,  # Corner classification labels.
            'room_labels': gt_inst.gt_classes, # Instance class labels
            #'boxes': torch.stack(boxes)  # Uncomment if bounding boxes are needed.
        }

        targets.append(room_dict)

    return targets


def visualize_targets(targets, gt_file_names):
    import numpy as np
    import cv2
    import os
    for k, room_dict in enumerate(targets):
        # Retrieve ground truth masks and corner classes
        gt_masks = room_dict["coords"].cpu().numpy()  # (num_instances, num_corners*2)
        gt_corner_classes = room_dict["labels"].cpu().numpy()  # (num_instances, num_corners)

        # Read the original image
        img = cv2.imread(gt_file_names[k])  # Example: '../../data/AIcrowd/train/images/000000059540.jpg'
        print(f"Processing image: {gt_file_names[k]}")

        # Loop through each instance's mask and corner labels
        for i in range(len(gt_masks)):
            polygon = gt_masks[i].reshape(-1, 2)
            index = np.nonzero(gt_corner_classes[i])  # Get the indices of corner points
            polygon = polygon[index]  # Filter polygon to include only corner points

            # Draw the polygon edges
            polygon = np.round(polygon).astype(np.int32).reshape((-1, 1, 2))
            img = cv2.polylines(img, [polygon], isClosed=True, color=(255, 255, 0), thickness=1)

            # Draw each corner of the polygon
            polygon = polygon.reshape((-1, 2))
            for cor in polygon:
                img = cv2.circle(img, tuple(cor), radius=2, color=(255, 255, 255), thickness=-1)

            # Highlight the first and second corners with different colors
            img = cv2.circle(img, tuple(polygon[0]), radius=2, color=(255, 255, 0),
                             thickness=-1)  # First corner (yellow)
            img = cv2.circle(img, tuple(polygon[1]), radius=2, color=(0, 255, 255),
                             thickness=-1)  # Second corner (cyan)

        # Save the modified image with the same filename in the current directory
        output_path = "./" + os.path.split(gt_file_names[k])[1]
        cv2.imwrite(output_path, img)


def pad_gt_boxes(gt_instances, num_proposals, device):
    """
    Pads ground truth bounding boxes for each image in the batch to match the number of proposals.
    """
    proposal_boxes = []  # per batch
    box_lengths = []

    for gt_inst in gt_instances:  # per image
        gt_boxes = gt_inst.gt_boxes.tensor  # Tensor containing ground truth bounding boxes (num_instances, 4)
        box_lengths.append(len(gt_boxes))

        # Create a padded box tensor for this image, filling extra proposals with zeros
        boxes_pad = torch.zeros([num_proposals, 4], device=device)
        boxes_pad[:len(gt_boxes)] = gt_boxes
        proposal_boxes.append(boxes_pad)

    proposal_boxes = torch.stack(proposal_boxes).view(-1, num_proposals, 4)  # (batch_size, num_proposals, 4)
    box_lengths = torch.tensor(box_lengths, device=device)

    return proposal_boxes, box_lengths
