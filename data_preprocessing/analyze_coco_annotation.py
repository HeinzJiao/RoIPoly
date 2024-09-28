"""
Usage:
```bash
python analyze_coco_annotation.py --json_path /path/to/annotation.json
python analyze_coco_annotation.py --json_path annotation_sm_us_euclidean.json
"""
import json
import argparse
from shapely.geometry import Polygon
from tqdm import tqdm
import numpy as np


def analyze_coco_annotation(json_path):
    # Load the COCO-format annotation file
    with open(json_path, 'r') as f:
        data = json.load(f)

    # Extract image and annotation information
    images = data['images']
    annotations = data['annotations']

    num_images = len(images)
    num_buildings = len(annotations)

    print(f"Total number of images: {num_images}")
    print(f"Total number of buildings: {num_buildings}")

    # Initialize variables to track the required information
    buildings_per_image = {}
    max_vertices = 0
    min_vertices = float('inf')

    # Process each annotation
    for annotation in tqdm(data['annotations'], desc="Analyzing annotations"):
        img_id = annotation['image_id']
        segmentation = annotation['segmentation'][0]
        num_vertices = len(np.array(segmentation).reshape(-1, 2))

        # Track the number of buildings per image
        if img_id not in buildings_per_image:
            buildings_per_image[img_id] = 0
        buildings_per_image[img_id] += 1

        # Track the max and min number of vertices per building
        max_vertices = max(max_vertices, num_vertices)
        min_vertices = min(min_vertices, num_vertices)

    # Get the image with the most buildings
    max_buildings = max(buildings_per_image.values())

    # Output the results
    print(f"\nResults:")
    print(f"Maximum number of buildings in a single image: {max_buildings}")
    print(f"Maximum number of vertices in a building polygon: {max_vertices}")
    print(f"Minimum number of vertices in a building polygon: {min_vertices}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Analyze COCO-format annotation JSON")
    parser.add_argument("--json_path", type=str, required=True, help="Path to COCO-format annotation.json file")

    args = parser.parse_args()

    analyze_coco_annotation(args.json_path)

    # annotation_sm_clean_us_index.json
    # Total number of images: 267542
    # Total number of buildings: 2308194
    # Maximum number of buildings in a single image: 34
    # Number of vertices in a building polygon: 30

    # annotation_sm_clean_us_euclidean.json
    # Total number of images: 267542
    # Total number of buildings: 2308194
    # Maximum number of buildings in a single image: 34
    # Number of vertices in a building polygon: 30

    # annotation_sm.json
    # Total number of images: 267542
    # Total number of buildings: 2338811
    # Maximum number of buildings in a single image: 34
    # Maximum number of vertices in a building polygon: 81
    # Minimum number of vertices in a building polygon: 4

    # annotation_large.json
    # Total number of images: 48852
    # Total number of buildings: 56742
    # Maximum number of buildings in a single image: 4
    # Maximum number of vertices in a building polygon: 263
    # Minimum number of vertices in a building polygon: 4

    # annotation_large_clean_us_index.json
    # Total number of images: 48852
    # Total number of buildings: 56742
    # Maximum number of buildings in a single image: 4
    # Number of vertices in a building polygon: 96
