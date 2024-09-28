import json
import numpy as np
import shapely.geometry
import argparse
from tqdm import tqdm


def filter_large_buildings(original_annotation, save_path, large_area_threshold=96 ** 2):
    """
    Filter out small/medium buildings (area < large_area_threshold) from images containing both large and small/medium buildings.
    Remove images that only contain small/medium buildings and save the new filtered annotation file.

    Args:
        original_annotation (str): Path to the original COCO-format annotation JSON file.
        save_path (str): Path to save the filtered annotations.
        large_area_threshold (float, optional): The threshold for a building to be considered "large." Defaults to 96**2.

    Returns:
        None
    """
    # Load the original annotation file
    print(f"Loading original annotation from {original_annotation}...")
    with open(original_annotation, 'r') as f:
        data = json.load(f)

    # Prepare new annotations list for large buildings
    new_annotations = []
    large_img_ids = set()  # To store image ids that have large buildings

    # Store the ids of images that contain only small/medium buildings
    small_medium_img_ids = set()

    # Create a map of image ids to annotations for easier lookup
    img_id_to_annotations = {img['id']: [] for img in data['images']}
    for annotation in data['annotations']:
        img_id_to_annotations[annotation['image_id']].append(annotation)

    # Iterate through each image and check the annotations
    for img in tqdm(data['images'], desc="Processing images"):
        img_id = img['id']
        annotations = img_id_to_annotations[img_id]

        large_buildings = []
        small_medium_buildings = []

        for annotation in annotations:
            segmentation = np.array(annotation['segmentation'][0]).reshape(-1, 2)
            polygon = shapely.geometry.Polygon(segmentation)
            area = polygon.area

            # Classify the building based on its area
            if area >= large_area_threshold:
                large_buildings.append(annotation)
            else:
                small_medium_buildings.append(annotation)

        # If the image has large buildings, keep only large ones and discard small/medium ones
        if large_buildings:
            new_annotations.extend(large_buildings)
            large_img_ids.add(img_id)
        else:
            small_medium_img_ids.add(img_id)

    # Remove images that only contained small/medium buildings
    new_images = [img for img in data['images'] if img['id'] in large_img_ids]

    # Update the original data structure with the new annotations and images
    data['annotations'] = new_annotations
    data['images'] = new_images

    # Save the filtered annotation file
    print(f"Saving filtered annotations to {save_path}...")
    with open(save_path, 'w') as f:
        json.dump(data, f)

    print(f"Filtered annotations saved successfully to {save_path}.")


def parse_args():
    parser = argparse.ArgumentParser(description="Filter large buildings from COCO-format annotations")
    parser.add_argument('--original_annotation', type=str, required=True, help="Path to the original COCO-format annotation JSON file")
    parser.add_argument('--save_path', type=str, required=True, help="Path to save the filtered annotations")
    parser.add_argument('--large_area_threshold', type=float, default=96**2, help="Threshold for a building to be considered large (default: 96^2 pixels)")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    filter_large_buildings(args.original_annotation, args.save_path, args.large_area_threshold)
