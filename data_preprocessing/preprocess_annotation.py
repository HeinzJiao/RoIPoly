"""
This script preprocesses COCO-format polygon corner annotations, cleaning and padding polygons.

Parameters:
- `--json_path`: Path to the original COCO-format annotation file.
- `--save_path`: Path to save the preprocessed annotation file.
- `--num_corners`: Number of vertices to pad each polygon with (used only for training).
- `--image_size`: Size of the image to clip polygon coordinates.
- `--min_area`: Minimum area threshold for filtering out polygons.
- `--sampling_method`: Sampling method for polygon padding.

Usage:
```bash
python preprocess_annotation.py --json_path /path/to/annotations.json \
                                --save_path /path/to/save/preprocessed.json \
                                --num_corners 60 \
                                --image_size 300 \
                                --min_area 4 \
                                --sampling_method 'uniform_index'
"""
import numpy as np
import cv2
import json
import os
import shapely.geometry
import argparse
from tqdm import tqdm
from preprocess_utils import (remove_redundant_vertices, approximate_polygons, uniform_sampling_index,
                              uniform_sampling_euclidean, resort_corners, resort_corners_and_labels,
                              get_gt_bboxes)


def preprocess_annotation(json_path, save_path, num_corners, image_size, min_area, sampling_method):
    """
    Preprocesses the COCO-style annotation file by removing invalid/noisy polygons and optionally padding polygons.

    :param json_path: Path to the original annotation file
    :param save_path: Path to save the preprocessed annotation file
    :param num_corners: Number of uniform vertices to sample for polygon padding
    :param image_size: Size of the image (assuming square)
    :param min_area: Minimum polygon area to be considered valid
    :param sampling_method: Polygon padding method (None, 'uniform_euclidean', 'uniform_index')
    """
    print("Loading annotation file...")
    labels = json.load(open(json_path, "r"))
    print("Annotation file loaded.")

    annotations = labels["annotations"]
    indices_to_remove = []  # Polygons to remove due to invalid criteria

    for i, anno in enumerate(tqdm(annotations, desc="Processing annotations")):
        # Get the ground truth polygon vertices
        gt_pts = np.array(anno["segmentation"][0])

        # Clip the polygon points to the image size
        gt_pts = np.clip(gt_pts, 0.0, image_size - 1)

        # Remove redundant vertices and invalid polygons
        gt_pts = remove_redundant_vertices(gt_pts, epsilon=0.1)
        if gt_pts.shape[0] < 3 or shapely.geometry.Polygon(gt_pts).area < min_area:
            indices_to_remove.append(i)
            continue

        # Simplify the polygon and remove invalid polygons
        gt_pts = approximate_polygons(gt_pts, tolerance=0.01)
        if gt_pts.shape[0] < 3 or shapely.geometry.Polygon(gt_pts).area < min_area:
            indices_to_remove.append(i)
            continue

        # Compute the bounding box around the ground truth polygon and enlarge it by 20%.
        gt_bbox = get_gt_bboxes(gt_pts, image_size)

        if sampling_method is not None:
            # If processing the training set, apply uniform sampling.
            # Returns:
            # gt_pts (numpy.ndarray): Flattened array of sampled polygon points, shape (num_corners * 2).
            # gt_cor_cls (numpy.ndarray): Array of corner classification labels, shape (num_corners,).
            if sampling_method == 'uniform_euclidean':
                # Uniform sampling using the traditional Euclidean distance cost matrix
                gt_pts, gt_cor_cls = uniform_sampling_euclidean(gt_pts, num_corners, image_size, image_size)
            elif sampling_method == 'uniform_index':
                # Uniform sampling using the new method with index difference cost matrix
                gt_pts, gt_cor_cls = uniform_sampling_index(gt_pts, num_corners, image_size, image_size)

            # Resort the corners so that the first corner starts from upper-left and counterclockwise ordered in image
            gt_pts, gt_cor_cls = resort_corners_and_labels(gt_pts, gt_cor_cls)
            annotations[i]["segmentation"] = [[round(x) for x in gt_pts]]
            annotations[i]["cor_cls_poly"] = [int(c) for c in gt_cor_cls]
        else:
            # For test sets (sampling_method=None), only clean and process without sampling.
            gt_pts = resort_corners(gt_pts)
            annotations[i]["segmentation"] = [gt_pts.tolist()]  # Keep float values for test set

        annotations[i]["bbox"] = gt_bbox  # list, [x1, y1, w, h], float

    # Remove invalid polygons
    indices_to_remove = sorted(indices_to_remove)
    for i in reversed(indices_to_remove):
        annotations.pop(i)

    labels["annotations"] = annotations
    with open(save_path, 'w') as fp:
        json.dump(labels, fp)
    print("Preprocessing completed and saved to", save_path)


def parse_args():
    parser = argparse.ArgumentParser(
        description='Preprocess annotations by cleaning polygons and optionally applying uniform sampling.')
    parser.add_argument("--json_path", type=str, required=True, help="Path to the input annotation file")
    parser.add_argument("--save_path", type=str, required=True, help="Path to save the preprocessed annotation file")
    parser.add_argument("--num_corners", type=int, default=30, help="Number of corners to sample per polygon")
    parser.add_argument("--image_size", type=int, default=300, help="Size of the image (assuming square)")
    parser.add_argument("--min_area", type=float, default=4, help="Minimum polygon area threshold")
    parser.add_argument("--sampling_method", type=str, choices=[None, 'uniform_euclidean', 'uniform_index'], default=None,
                        help="Sampling method for polygon padding")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    preprocess_annotation(args.json_path, args.save_path, args.num_corners, args.image_size, args.min_area,
                          args.sampling_method)
