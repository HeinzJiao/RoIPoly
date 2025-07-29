# CrowdAI Processed Annotations

The preprocessed training set based on the CrowdAI dataset can be downloaded [here] (https://drive.google.com/drive/folders/1dNcNe_ah6SFLVDFeE3-34p0af87jJBRN?usp=sharing).

For detailed information on the polygon padding process, please refer to Section 3.1 *"Reformulation of Building Polygon Prediction"* in the paper.

---

## 📄 File Descriptions

### `annotation.json`
- This is the raw annotation file.

### `annotation_sm.json`
Generated with the following command:
```
python filter_small_medium_buildings.py --original_annotation annotation.json \
        --save_path annotation_sm.json \
        --large_area_threshold 9216 \
        --small_area_threshold 1024
```
Changes made:
- Removed all images containing only large buildings (buildings with an area greater than or equal to 96^2 pixels).
- For images containing both large and small/medium buildings, only the small/medium building annotations were kept.

### `annotation_sm_clean_us_euclidean.json`
Generated with the following command:
```
python preprocess_annotation.py --json_path annotation_sm.json \
                                --save_path annotation_sm_clean_us_euclidean.json \
                                --num_corners 30 \
                                --image_size 300 \
                                --min_area 4 \
                                --sampling_method 'uniform_euclidean'
```
Changes made:
- cleaned polygons, removed invalid or noisy polygons.
- padded polygons to a fixed number of vertices using uniform sampling with a Euclidean distance-based cost matrix for bipartite matching.

### `annotation_sm_clean_us_index.json`
Generated with the following command:
```
python preprocess_annotation.py --json_path annotation_sm.json \
                                --save_path annotation_sm_clean_us_index.json \
                                --num_corners 30 \
                                --image_size 300 \
                                --min_area 4 \
                                --sampling_method 'uniform_index'
```
Changes made:
- cleaned polygons, removed invalid or noisy polygons.
- padded polygons to a fixed number of vertices using uniform sampling with a index difference-based cost matrix for bipartite matching.
✅ This is the default annotation file used for small/medium buildings in the experiments.
It corresponds to the ordering-based matching method described in Section 3.1 of the paper.

### `annotation_large.json`
Generated with the following command:
```
python filter_large_buildings.py --original_annotation annotation.json \
        --save_path annotation_large.json \
        --large_area_threshold 9216 \
```
Changes made:
- Removed all images containing only small/medium buildings (buildings with an area smaller than 96^2 pixels).
- For images containing both large and small/medium buildings, only the large building annotations were kept.

### `annotation_large_clean_us_index.json`
Generated with the following command:
```
python preprocess_annotation.py --json_path annotation_large.json \
                                --save_path annotation_large_clean_us_index.json \
                                --num_corners 96 \
                                --image_size 300 \
                                --min_area 4 \
                                --sampling_method 'uniform_index'
```
Changes made:
- cleaned polygons, removed invalid or noisy polygons.
- padded polygons to a fixed number of vertices using uniform sampling with a index difference-based cost matrix for bipartite matching.
✅ This is the default annotation file used for large buildings in the experiments.
It corresponds to the ordering-based matching method described in Section 3.1 of the paper.

# Note
- All cleaned annotations follow a fixed-vertex polygon representation.
- The choice to split buildings into small/medium and large categories is for memory efficiency only.
- Users with sufficient computational resources are encouraged to use the unified original annotation and increase num_proposal_vertices accordingly.
