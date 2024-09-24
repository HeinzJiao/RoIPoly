"""
Usage:
```bash
python predictions_to_coco.py --json_path "./sparsercnn_swinb_iter1930059_predictions.json" \
                              --annotation_path "./annotation_preprocessed.json" \
                              --save_path "./annotation_sparsercnn_swinb_1930059_sm.json" \
                              --building_type "small_medium" \
                              --min_area 96 ** 2 * 1.44

"""
import json
import numpy as np
from shapely.geometry import Polygon
from tqdm import tqdm
import argparse


def predictions_to_coco(json_path, annotation_path, save_path, building_type, min_area=96 ** 2 * 1.44):
    """
    Convert object detector predictions to COCO-format annotations and filter based on building size.

    This function ensures that the output COCO-format annotations contain all the images from the
    given test annotation file, and filters bounding boxes based on their size (small/medium or large).

    Args:
        json_path (str): Path to the predictions JSON file from the object detector.
        annotation_path (str): Path to the COCO-format annotation JSON used for testing.
        save_path (str): Path to save the newly generated COCO-format annotation file.
        building_type (str): Specify whether to keep "small_medium" or "large" buildings.
        min_area (float, optional): Threshold to filter buildings by bounding box area. Defaults to 96**2 * 1.44.

    Returns:
        None
    """
    print(f"Loading predictions from {json_path}...")
    with open(json_path, 'r') as f:
        predictions = json.load(f)
    print("Predictions loaded.")

    # Load the annotation data used for testing (to ensure all images are included)
    print(f"Loading annotation from {annotation_path}...")
    with open(annotation_path, 'r') as f:
        annotation_data = json.load(f)

    # Prepare COCO-style labels dictionary for the output
    labels = {
        "info": {
            "contributor": "crowdAI.org",
            "about": "Dataset for crowdAI Mapping Challenge",
            "date_created": "07/03/2018",
            "description": "crowdAI mapping-challenge dataset",
            "url": "https://www.crowdai.org/challenges/mapping-challenge",
            "version": "1.0",
            "year": 2018
        },
        "categories": [{"id": 100, "name": "building", "supercategory": "building"}],
        "images": [],
        "annotations": []
    }

    # Set to keep track of processed image IDs
    processed_image_ids = set()

    # Process each prediction and filter based on building size
    for i, pred in enumerate(tqdm(predictions, desc="Processing predictions")):
        image_id = pred["image_id"]

        # Get bounding box dimensions (w, h)
        w, h = pred["bbox"][2], pred["bbox"][3]

        # Filter based on the specified building type and area threshold
        if building_type == "small_medium" and h * w >= min_area:
            continue
        elif building_type == "large" and h * w < min_area:
            continue

        # Keep track of image IDs that have valid buildings
        processed_image_ids.add(image_id)

        # Add annotation in COCO format
        annotation = {
            'id': i,
            'image_id': image_id,
            'area': h * w,
            'category_id': 100,
            'iscrowd': 0,
            'bbox': pred["bbox"]  # [x1, y1, w, h]
        }
        labels["annotations"].append(annotation)

    # Ensure all images in the test set are included
    labels["images"] = annotation_data['images']

    # Save the filtered annotations to the output path
    print(f"Saving filtered annotations to {save_path}...")
    with open(save_path, 'w') as f:
        json.dump(labels, f)

    print(f"Filtered annotations saved successfully to {save_path}.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Convert object detector predictions to COCO-format annotations.")
    parser.add_argument('--json_path', type=str, required=True, help='Path to the predictions JSON file')
    parser.add_argument('--annotation_path', type=str, required=True, help='Path to the test annotation JSON file')
    parser.add_argument('--save_path', type=str, required=True, help='Path to save the COCO-format annotation file')
    parser.add_argument('--type', type=str, required=True, choices=["small_medium", "large"],
                        help='Specify to keep "small_medium" or "large" buildings')
    parser.add_argument('--min_area', type=float, default=96 ** 2 * 1.44, help='Threshold to filter buildings by area')

    args = parser.parse_args()

    predictions_to_coco(args.json_path, args.annotation_path, args.save_path, args.type, args.min_area)
