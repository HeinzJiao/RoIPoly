import json
import numpy as np
import shapely.geometry
import argparse
from tqdm import tqdm


def filter_small_medium_large_buildings(original_annotation, save_path, large_area_threshold=96 ** 2,
                                        small_area_threshold=32 ** 2):
    """
    Filter out large buildings (area >= large_area_threshold) from images containing both large and small/medium buildings.
    Remove images that only contain large buildings and save the new filtered annotation file.

    Args:
        original_annotation (str): Path to the original COCO-format annotation JSON file.
        save_path (str): Path to save the filtered annotations.
        large_area_threshold (float, optional): The threshold for a building to be considered "large." Defaults to 96**2.
        small_area_threshold (float, optional): The threshold for a building to be considered "small/medium." Defaults to 32**2.

    Returns:
        None
    """
    # Load the original annotation file
    print(f"Loading original annotation from {original_annotation}...")
    with open(original_annotation, 'r') as f:
        data = json.load(f)

    # Prepare a new annotations list for the filtered data
    new_annotations = []
    small_medium_img_ids = set()  # To store image ids that have small or medium buildings

    # Iterate over each annotation
    for i, annotation in enumerate(tqdm(data['annotations'], desc="Processing annotations")):
        img_id = annotation['image_id']

        # Calculate the area of the building polygon
        segmentation = np.array(annotation['segmentation'][0]).reshape(-1, 2)
        polygon = shapely.geometry.Polygon(segmentation)
        area = polygon.area

        # Classify the building based on its area
        if area < large_area_threshold:  # Keep small/medium buildings
            new_annotations.append(annotation)
            small_medium_img_ids.add(img_id)

    # Remove images that only contained large buildings
    new_images = [img for img in data['images'] if img['id'] in small_medium_img_ids]

    # Update the original data structure with the new annotations and images
    data['annotations'] = new_annotations
    data['images'] = new_images

    # Save the filtered annotation file
    print(f"Saving filtered annotations to {save_path}...")
    with open(save_path, 'w') as f:
        json.dump(data, f)

    print(f"Filtered annotations saved successfully to {save_path}.")


if __name__ == "__main__":
    # Define the argument parser
    parser = argparse.ArgumentParser(description="Filter small and medium buildings from COCO-format annotations.")

    parser.add_argument(
        "--original_annotation",
        type=str,
        required=True,
        help="Path to the original COCO-format annotation JSON file."
    )

    parser.add_argument(
        "--save_path",
        type=str,
        required=True,
        help="Path to save the filtered annotations JSON file."
    )

    parser.add_argument(
        "--large_area_threshold",
        type=float,
        default=96 ** 2,
        help="Area threshold for large buildings (default: 96^2)."
    )

    parser.add_argument(
        "--small_area_threshold",
        type=float,
        default=32 ** 2,
        help="Area threshold for small buildings (default: 32^2)."
    )

    # Parse arguments
    args = parser.parse_args()

    # Run the filtering function with parsed arguments
    filter_small_medium_large_buildings(
        original_annotation=args.original_annotation,
        save_path=args.save_path,
        large_area_threshold=args.large_area_threshold,
        small_area_threshold=args.small_area_threshold
    )
