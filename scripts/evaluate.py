"""
This script is used to generate predicted building polygons using a pre-trained model.

During the prediction process, known building bounding boxes (bbox) are input into the model.
These bboxes can either be ground truth bboxes or bboxes predicted by a pre-trained object
detection model. The source of the bboxes depends on the `TEST_JSON` parameter passed in.

In the prediction process, since the bboxes are known, we do not use Hungarian matching
during training. Instead, each region of interest (RoI) feature corresponds to a single
polygon prediction. We directly map the top `num_gt_boxes` predictions to the known bboxes.
Therefore, during the prediction process, we only need to take the top `num_gt_boxes`
predicted results for each image.
"""
import json
import time
import numpy as np
from tqdm import tqdm
import multiprocessing as mp
import torch
from detectron2.modeling import build_model
from detectron2.checkpoint import DetectionCheckpointer
from detectron2.utils.logger import setup_logger
from detectron2.data import build_detection_test_loader
from detectron2.data import DatasetCatalog, MetadataCatalog
from detectron2.data.datasets.coco import load_coco_json
from roipoly import RoIPolyDatasetMapper, add_roipoly_config
import shapely.geometry
from detectron2.config import get_cfg
import argparse
import os


def setup_cfg(args):
    """Set up the configuration for the Detectron2 model."""
    cfg = get_cfg()
    add_roipoly_config(cfg)
    cfg.merge_from_file(args.config_file)
    cfg.merge_from_list(args.opts)
    cfg.CORNER_THRESHOLD = args.corner_threshold
    cfg.OUTPUT_DIRPATH = args.output
    cfg.ITER = args.iter
    cfg.IMAGE_ID = args.image_id
    cfg.freeze()
    return cfg


def get_parser():
    """Set up argument parser for command line options."""
    parser = argparse.ArgumentParser(description="Detectron2 demo for builtin models")
    parser.add_argument("--config-file", default="configs/roipoly.res50.100pro.3x_inference_acc_test.yaml", metavar="FILE", help="Path to config file")
    parser.add_argument("--output", help="Directory to save output visualizations")
    parser.add_argument("--gt-path", help="Directory of the annotation file")
    parser.add_argument("--corner-threshold", type=float, default=0.4, help="Threshold for vertex predictions to be classified as corners")
    parser.add_argument("--num-gpus", type=int, default=1, help="Number of GPUs to use")
    parser.add_argument("--iter", type=int, default=2842656, help="Checkpoint iteration number")
    parser.add_argument("--image-id", type=int, default=1, help="ID of the input image")
    parser.add_argument("--train-json", help="Path to the COCO-format annotation file")
    parser.add_argument("--train-path", help="Path to the images directory")
    parser.add_argument("--opts", default=[], nargs=argparse.REMAINDER, help="Modify config options using the command-line 'KEY VALUE' pairs")
    return parser


def single_annotation(image_id, poly, bbox, score):
    """Create a single annotation result dictionary."""
    return {
        "image_id": int(image_id),
        "category_id": 100,
        "score": score,
        "segmentation": poly,
        "bbox": [bbox[0], bbox[1], bbox[2] - bbox[0], bbox[3] - bbox[1]]  # Convert (x1, y1, x2, y2) to (x1, y1, width, height)
    }


def prediction(cfg):
    """Perform predictions on the test dataset."""
    mapper = RoIPolyDatasetMapper(cfg, is_train=False)
    dataloader = build_detection_test_loader(DatasetCatalog.get("aicrowd_test"), mapper=mapper)
    test_iterator = tqdm(dataloader)

    # Load the RoIPoly model
    model = build_model(cfg)
    model.eval()
    checkpointer = DetectionCheckpointer(model)
    checkpointer.load(cfg.MODEL.WEIGHTS)

    speed = []  # List to keep track of model speed
    predictions = []  # List to store prediction results

    # Iterate over the test dataset
    for i_batch, batched_inputs in enumerate(test_iterator):
        # Process per image (batch_size = 1)
        h, w = batched_inputs[0]["height"], batched_inputs[0]["width"]
        instances = batched_inputs[0]["instances"]
        gt_boxes = instances.gt_boxes.tensor  # Ground truth bounding boxes
        num_gt_boxes = gt_boxes.shape[0]  # Number of ground truth boxes

        # Perform inference with the model
        with torch.no_grad():
            t0 = time.time()
            outputs = model(batched_inputs)
            t1 = time.time()

        # Process model outputs
        pred_logits = outputs["pred_logits"][0]  # [num_proposals, num_corners]
        pred_coords = outputs["pred_coords"][0]  # [num_proposals, num_corners, 2]
        fg_mask = torch.sigmoid(pred_logits) > cfg.CORNER_THRESHOLD  # [num_proposals, num_corners]

        polys_image = []  # List to store valid polygons
        scores_image = []  # List to store average scores per polygon
        boxes_image = []  # List to store bounding boxes for valid polygons

        # We only need to take the top `num_gt_boxes` predicted results for each image.
        for j in range(num_gt_boxes):
            fg_mask_per_poly = fg_mask[j]  # [num_corners]
            valid_coords_per_poly = pred_coords[j][fg_mask_per_poly]  # [num_valid_corners_per_poly, 2]
            valid_scores_per_poly = torch.sigmoid(pred_logits[j])[fg_mask_per_poly]  # [num_valid_corners_per_poly]

            if len(valid_coords_per_poly) > 0:
                coords = valid_coords_per_poly.cpu().numpy()  # [num_valid_corners_per_poly, 2]
                if len(coords) >= 3 and shapely.geometry.Polygon(coords).area >= 10:
                    polys_image.append(coords)
                    scores_image.append(
                        torch.mean(valid_scores_per_poly).cpu().numpy())  # Average score per polygon
                    boxes_image.append(np.array([np.min(coords[:, 0]), np.min(coords[:, 1]),
                                                 np.max(coords[:, 0]), np.max(coords[:, 1])]))

        # Store predictions for the current image
        image_id = batched_inputs[0]["image_id"]
        for j, poly in enumerate(polys_image):
            box = list(np.array(boxes_image[j]).flatten().astype(np.float64))
            score = float(scores_image[j])
            poly = list(np.array(poly).flatten().astype(np.float64))

            # polygons with 2 vertices will be mistaken as bounding boxes and cause error in eval_coco.py
            if len(poly) > 4:
                predictions.append(single_annotation(image_id, [poly], box, score))

        # Calculate model speed
        speed.append(t1 - t0)

    # Print average model speed
    print("Average model speed: ", np.mean(speed), " [s / image]")

    # Save predictions to a JSON file
    output_path = os.path.join(cfg.OUTPUT_DIRPATH, f"predictions_{int(cfg.ITER)}.json")
    with open(output_path, "w") as fp:
        json.dump(predictions, fp)


def register_my_dataset(dataset_name="aicrowd_test", TEST_JSON="../../data/AIcrowd/val/annotation.json",
                        TEST_PATH="../../data/AIcrowd/val/images"):
    """Register your own COCO-format dataset.

       usage::
       from detectron2.data import DatasetCatalog, MetadataCatalog
       from detectron2.data.datasets.coco import load_coco_json

    :param TRAIN_JSON: the file path of the annotation
    :param TRAIN_PATH: the folder path of the images
    """
    DatasetCatalog.register(dataset_name, lambda: load_coco_json(TEST_JSON, TEST_PATH, dataset_name))
    MetadataCatalog.get(dataset_name).set(json_file=TEST_JSON, image_root=TEST_PATH)


if __name__ == '__main__':
    mp.set_start_method("spawn", force=True)
    args = get_parser().parse_args()

    setup_logger(name="fvcore")
    logger = setup_logger()
    logger.info("Arguments: " + str(args))

    cfg = setup_cfg(args)  # default config, args.config_file, args.opts
    register_my_dataset(dataset_name="aicrowd_test", TEST_JSON=args.train_json,
                        TEST_PATH=args.train_path)
    prediction(cfg)
