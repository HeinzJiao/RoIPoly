import json
import argparse
from tqdm import tqdm


def convert_to_predictions(annotation_file, output_file):
    """
    Convert COCO-style annotation.json to predictions.json by filtering out sampled points.

    Args:
        annotation_file (str): Path to the COCO-style annotation file.
        output_file (str): Path to save the resulting predictions.json.
    """
    with open(annotation_file, "r") as f:
        data = json.load(f)

    predictions = []
    for annotation in tqdm(data["annotations"], desc="Converting annotations to predictions"):
        # Extract segmentation and cor_cls_poly (class labels for vertices)
        segmentation = annotation["segmentation"][0]
        cor_cls_poly = annotation["cor_cls_poly"]

        # Convert segmentation into numpy array and filter by the real vertex class (1 in cor_cls_poly)
        polygon = [segmentation[i:i + 2] for i, cls in enumerate(cor_cls_poly) if cls == 1]

        # Convert to the format required for predictions.json
        if len(polygon) >= 3:  # Ensure the polygon is valid
            bbox = annotation["bbox"]  # Assume bbox is already in x, y, w, h format
            prediction = {
                "image_id": annotation["image_id"],
                "category_id": annotation["category_id"],  # Assuming you have category_id = 100 for buildings
                "score": 1.0,  # You can assign a default score of 1.0 or modify it accordingly
                "segmentation": [list(sum(polygon, []))],  # Flatten the list of polygons
                "bbox": bbox  # Bbox stays the same
            }
            predictions.append(prediction)

    # Save the resulting predictions.json file
    with open(output_file, "w") as out_file:
        json.dump(predictions, out_file)

    print(f"Predictions saved to {output_file}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Convert COCO-style annotations to predictions format")
    parser.add_argument("--annotation_file", type=str, required=True, help="Path to COCO-style annotation.json file")
    parser.add_argument("--output_file", type=str, required=True, help="Path to save predictions.json file")
    args = parser.parse_args()

    convert_to_predictions(args.annotation_file, args.output_file)
