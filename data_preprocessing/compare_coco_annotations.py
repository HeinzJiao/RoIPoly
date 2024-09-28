import json
import argparse


def compare_annotations(file1, file2):
    # Load the JSON files
    with open(file1, 'r') as f1:
        data1 = json.load(f1)
    with open(file2, 'r') as f2:
        data2 = json.load(f2)

    # Check if image sets are identical
    images1 = {img['id']: img['file_name'] for img in data1['images']}
    images2 = {img['id']: img['file_name'] for img in data2['images']}

    images1_set = set(images1.keys())
    images2_set = set(images2.keys())

    print(f"Total images in {file1}: {len(images1)}")
    print(f"Total images in {file2}: {len(images2)}")

    if images1_set == images2_set:
        print("Both datasets contain the same images.")
    else:
        print("The datasets contain different images.")
        print(f"Images in {file1} but not in {file2}: {images1_set - images2_set}")
        print(f"Images in {file2} but not in {file1}: {images2_set - images1_set}")

    # Compare annotations
    annotations1 = {ann['image_id']: [] for ann in data1['annotations']}
    annotations2 = {ann['image_id']: [] for ann in data2['annotations']}

    for ann in data1['annotations']:
        annotations1[ann['image_id']].append(ann)

    for ann in data2['annotations']:
        annotations2[ann['image_id']].append(ann)

    diff_images = []
    for img_id in images1_set.intersection(images2_set):
        anns1 = sorted(annotations1.get(img_id, []), key=lambda x: x['id'])
        anns2 = sorted(annotations2.get(img_id, []), key=lambda x: x['id'])

        if len(anns1) != len(anns2):
            diff_images.append(img_id)
            print(f"Different number of annotations for image_id {img_id} ({images1[img_id]}):")
            print(f"{file1} has {len(anns1)} annotations, {file2} has {len(anns2)} annotations.")

    if diff_images:
        print(f"\nImages with differing annotations: {len(diff_images)}")
    else:
        print("All images have the same number of annotations.")


def parse_args():
    parser = argparse.ArgumentParser(description="Compare two COCO-format annotation files.")
    parser.add_argument("--file1", type=str, required=True, help="Path to the first annotation file")
    parser.add_argument("--file2", type=str, required=True, help="Path to the second annotation file")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    compare_annotations(args.file1, args.file2)
