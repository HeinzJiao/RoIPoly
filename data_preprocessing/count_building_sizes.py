import json
import numpy as np
import shapely.geometry
from tqdm import tqdm


def count_building_sizes(json_path, save_small_medium_path=None, save_large_path=None):
    """
    Count the number of small, medium, and large buildings in a COCO-format annotation file.

    Args:
        json_path (str): Path to the COCO-format annotation JSON file.
        save_large_path (str, optional): Path to save the image IDs of large buildings. Defaults to None.
        save_small_medium_path (str, optional): Path to save the image IDs of small and medium buildings. Defaults to None.

    Returns:
        tuple: A tuple containing the counts of small, medium, large buildings, and large bounding boxes.
    """
    print("Loading annotation file...")
    with open(json_path, "r") as f:
        labels = json.load(f)
    print("Annotation file loaded.")

    annotations = labels["annotations"]

    num_small, num_medium, num_large = 0, 0, 0
    small_medium_img_ids = set()  # IDs of images containing building smaller than 96**2
    large_building_img_ids = set()  # IDs of images containing buildings larger than or equal to 96^2

    for i, anno in enumerate(tqdm(annotations)):
        img_id = anno["image_id"]
        segmentation = np.array(anno["segmentation"][0]).reshape(-1, 2)  # Reshape the segmentation points
        polygon = shapely.geometry.Polygon(segmentation)
        area = polygon.area

        # Categorize by polygon area
        if area < 32 ** 2:
            num_small += 1
            small_medium_img_ids.add(img_id)
        elif 32 ** 2 <= area < 96 ** 2:
            num_medium += 1
            small_medium_img_ids.add(img_id)
        else:
            num_large += 1
            large_building_img_ids.add(img_id)

    print(f"Total small and medium buildings: {len(small_medium_img_ids)}")
    print(f"Total large buildings: {len(large_building_img_ids)}")

    # Save large building image IDs
    if save_large_path:
        with open(save_large_path, 'w') as fp:
            json.dump(list(large_building_img_ids), fp)

    # Save small and medium building image IDs
    if save_small_medium_path:
        with open(save_small_medium_path, 'w') as fp:
            json.dump(list(small_medium_img_ids), fp)

    return num_small, num_medium, num_large


if __name__ == "__main__":
    json_path = "annotation.json"  # Set the path to your annotation file here
    save_small_medium_path = "small_medium_building_img_ids.json"
    save_large_path = "large_building_img_ids.json"  # Optional: set path to save large image IDs
    small, medium, large = count_building_sizes(json_path, save_small_medium_path, save_large_path)

    print(f"Small buildings: {small}")
    print(f"Medium buildings: {medium}")
    print(f"Large buildings: {large}")
