The preprocessed val set based on the CrowdAI dataset can be downloaded [here](https://drive.google.com/drive/folders/1PUmvptNLJGTKxwmSTpMn2OlZabfICSdw?usp=sharing).

---

## 📄 File Descriptions

### `annotation.json`
- This is the raw annotation file.

### `sparsercnn_swinb_iter1930059_predictions.json`
Bounding box predictions generated using SparseR-CNN with a Swin-Base transformer backbone, trained on the AIcrowd dataset for 1,930,059 iterations.

### `annotation_clean.json`
Generated with the following command:
```
python preprocess_annotation.py --json_path annotation.json \
                                --save_path annotation_clean.json \
                                --image_size 300 \
                                --min_area 10 \
```
Changes made:
- cleaned polygons and removed invalid or noisy polygons.

### `annotation_sparsercnn_swinb_1930059_sm.json`
Generated with the following command:
```
python predictions_to_coco.py --json_path "./sparsercnn_swinb_iter1930059_predictions.json" \
                              --annotation_path "./annotation_clean.json" \
                              --save_path "./annotation_sparsercnn_swinb_1930059_sm.json" \
                              --building_type "small_medium" \
                              --min_area 96 ** 2 * 1.44
```
Changes made:
- Converted sparsercnn_swinb_iter1930059_predictions.json to COCO-format.
- Filtered out bounding boxes for small and medium buildings (area < 96**2 * 1.44).
- Ensured the image IDs match those in annotation_clean.json.

### `annotation_sparsercnn_swinb_1930059_large.json`
Generated with the following command:
```
python predictions_to_coco.py --json_path "./sparsercnn_swinb_iter1930059_predictions.json" \
                              --annotation_path "./annotation_clean.json" \
                              --save_path "./annotation_sparsercnn_swinb_1930059_large.json" \
                              --building_type "large" \
                              --min_area 96 ** 2 * 1.44
```
Changes made:
- Converted sparsercnn_swinb_iter1930059_predictions.json to COCO-format.
- Filtered out bounding boxes for large buildings (area ≥ 96**2 * 1.44).
- Ensured the image IDs match those in annotation_clean.json.

## Note
As the original CrowdAI dataset contains some degraded polygon annotations (e.g., with less than 3 vertices or very small area), we followed the common practice in the evaluation code of FFL [here] (https://github.com/Lydorn/Polygonization-by-Frame-Field-Learning) and applied minimal filtering to reduce the impact of annotation noise on evaluation.
